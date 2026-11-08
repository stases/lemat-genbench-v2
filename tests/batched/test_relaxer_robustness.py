"""Bounded CPU checks for relaxer admission, recovery, and thread lifetime."""

import sys
import threading
import weakref
from contextlib import closing
from functools import wraps

import pytest
import torch
from ase import Atoms
from test_relaxer import (
    EMTModel,
    assert_matches_reference,
    by_index,
    make_structures,
    reference_relax,
)

from lemat_genbench.batched import relaxer

pytestmark = pytest.mark.filterwarnings("ignore:logm result may be inaccurate")
TIMEOUT = 20


def bounded(test):
    """Run the entire scenario under a timeout, including cleanup and assertions."""

    @wraps(test)
    def run(*args, **kwargs):
        errors = []

        def target():
            try:
                test(*args, **kwargs)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(TIMEOUT)
        assert not thread.is_alive(), f"Scenario exceeded {TIMEOUT} s (deadlock?)"
        if errors:
            if not isinstance(errors[0], Exception):
                # an interrupt must fail this test, not stop the whole session
                pytest.fail(f"scenario raised {errors[0]!r}")
            raise errors[0]

    return run


def assert_threads_stopped(baseline):
    for thread in set(threading.enumerate()) - baseline:
        assert thread.daemon
        thread.join(1)
    assert set(threading.enumerate()) == baseline


@pytest.mark.parametrize(
    "mode", ["complete", "close", "consumer_error", "source_error"]
)
@bounded
def test_threads_return_to_baseline(mode):
    baseline = set(threading.enumerate())
    structures = make_structures(4)

    def jobs():
        for i, atoms in enumerate(structures):
            if mode == "source_error" and i == 2:
                raise ValueError("source failed")
            yield i, atoms

    with closing(
        relaxer.relax_stream(jobs(), EMTModel(), steps=2, max_in_flight=3)
    ) as stream:
        if mode == "complete":
            outcomes = list(stream)
            assert len(outcomes) == len(structures)
            assert all(o.error is None for o in outcomes)
        elif mode == "close":
            next(stream)
            stream.close()
        elif mode == "consumer_error":
            with pytest.raises(ValueError, match="consumer failed"):
                with closing(stream):
                    for _ in stream:
                        raise ValueError("consumer failed")
        else:
            with pytest.raises(ValueError, match="source failed"):
                list(stream)
    assert_threads_stopped(baseline)


@pytest.mark.parametrize("extra", [False, True], ids=["too_few", "too_many"])
@pytest.mark.parametrize("bad_singleton", [False, True])
@bounded
def test_wrong_prediction_count_is_isolated(extra, bad_singleton):
    structures = make_structures(3)
    for i, atoms in enumerate(structures):
        atoms.info["job"] = i

    class WrongCountModel(EMTModel):
        def predict(self, atoms):
            predictions = super().predict(atoms)
            if len(atoms) > 1 or (bad_singleton and atoms[0].info["job"] == 0):
                return predictions + predictions[:1] if extra else predictions[:-1]
            return predictions

    outcomes = by_index(
        list(relaxer.relax_stream(enumerate(structures), WrongCountModel(), steps=1))
    )
    assert set(outcomes) == set(range(len(structures)))
    for i, atoms in enumerate(structures):
        if bad_singleton and i == 0:
            outcome = outcomes[i]
            assert outcome.error == "Model must return one prediction per structure"
            assert outcome.atoms is outcome.energy is None
            assert not outcome.converged and outcome.steps == outcome.model_calls == 0
        else:
            assert_matches_reference(
                outcomes[i], reference_relax(atoms, EMTModel(), steps=1)
            )


@pytest.mark.parametrize("chained", [False, True], ids=["direct", "chained"])
@pytest.mark.parametrize("error_type", [torch.OutOfMemoryError, RuntimeError])
@bounded
def test_failed_forward_is_released_before_cache_clear_and_retry(
    monkeypatch, error_type, chained
):
    structures = make_structures(4)
    for i, atoms in enumerate(structures):
        atoms.info["job"] = i
    allocations = []
    cleared = []

    class Allocation:
        pass

    def failing_forward():
        allocation = Allocation()
        allocations.append(weakref.ref(allocation))
        raise RuntimeError("inner forward failure")

    class FailingModel(EMTModel):
        def predict(self, atoms):
            assert all(ref() is None for ref in allocations), "Forward frame retained"
            if any(a.info["job"] == 0 for a in atoms):
                if not chained:
                    allocation = Allocation()
                    allocations.append(weakref.ref(allocation))
                    raise error_type("injected forward failure")
                # an adapter that wraps the error keeps the forward's frames in the chain
                try:
                    failing_forward()
                except RuntimeError as inner:
                    raise error_type("injected forward failure") from inner
            return super().predict(atoms)

    def empty_cache():
        assert sys.exc_info()[0] is None, "Cache cleared inside an exception handler"
        assert all(ref() is None for ref in allocations), "Forward frame retained"
        cleared.append(True)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    outcomes = by_index(
        list(relaxer.relax_stream(enumerate(structures), FailingModel(), steps=0))
    )
    # The four-item batch, its left half, and its first singleton each fail.
    assert len(allocations) == 3
    assert all(ref() is None for ref in allocations)
    assert len(cleared) == (3 if error_type is torch.OutOfMemoryError else 0)
    assert outcomes[0].error == "injected forward failure"
    assert outcomes[0].model_calls == 0 and outcomes[0].atoms is None
    for i in range(1, len(structures)):
        assert_matches_reference(
            outcomes[i], reference_relax(structures[i], EMTModel(), steps=0)
        )


@bounded
def test_at_most_one_job_is_read_ahead():
    admitted = []
    pulled = []
    source_ids = set()
    observations = []
    lock = threading.Lock()

    class SourceAtoms(Atoms):
        def copy(self):
            copied = super().copy()
            if id(self) in source_ids:
                with lock:
                    admitted.append(self.info["job"])
            return copied

    structures = [SourceAtoms(a) for a in make_structures(6)]
    source_ids.update(map(id, structures))
    for i, atoms in enumerate(structures):
        atoms.info["job"] = i

    def jobs():
        for i, atoms in enumerate(structures):
            with lock:
                pulled.append(i)
                assert len(pulled) <= len(admitted) + 1
            yield i, atoms

    class ObservedModel(EMTModel):
        def predict(self, atoms):
            with lock:
                observations.append((len(pulled), len(admitted)))
            return super().predict(atoms)

    outcomes = list(
        relaxer.relax_stream(
            jobs(), ObservedModel(), steps=0, max_in_flight=3, max_atoms=8
        )
    )
    assert admitted == pulled == list(range(len(structures)))
    assert any(p == a + 1 for p, a in observations), "Budget never blocked admission"
    assert len(outcomes) == len(structures) and all(o.error is None for o in outcomes)


@bounded
def test_refill_keeps_every_round_full():
    structures = make_structures(7)
    for i, atoms in enumerate(structures):
        atoms.info["job"] = i
    completed = []
    rounds = []
    capacity = 3

    class ObservedModel(EMTModel):
        def predict(self, atoms):
            rounds.append(([a.info["job"] for a in atoms], len(completed)))
            return super().predict(atoms)

    for outcome in relaxer.relax_stream(
        enumerate(structures), ObservedModel(), steps=2, max_in_flight=capacity
    ):
        assert outcome.error is None
        completed.append(outcome.index)

    assert sorted(completed) == list(range(len(structures)))
    seen = set()
    refills = 0
    for batch, n_finished in rounds:
        assert len(batch) == min(capacity, len(structures) - n_finished)
        assert len(batch) == len(set(batch))
        if seen and set(batch) - seen:
            refills += 1
        seen.update(batch)
    assert refills > 0 and len(seen) == len(structures)


@bounded
def test_reused_input_is_copied_before_next_job(monkeypatch):
    baseline = set(threading.enumerate())
    released = threading.Event()

    class DelayedThread(threading.Thread):
        def run(self):
            assert released.wait(5)
            super().run()

    monkeypatch.setattr(relaxer, "Thread", DelayedThread)
    (shared,) = make_structures(1)
    originals = [shared.copy()]

    def jobs():
        yield 0, shared
        # Workers cannot copy the first geometry until after this mutation.
        shared.positions[0, 0] += 0.2
        originals.append(shared.copy())
        released.set()
        yield 1, shared

    try:
        outcomes = by_index(list(relaxer.relax_stream(jobs(), EMTModel(), steps=1)))
    finally:
        released.set()
    for i, atoms in enumerate(originals):
        assert_matches_reference(
            outcomes[i], reference_relax(atoms, EMTModel(), steps=1)
        )
    assert_threads_stopped(baseline)


@pytest.mark.parametrize("name", ["max_in_flight", "max_requests"])
@pytest.mark.parametrize("value", [None, 0, -1, 1.5, "2", True])
@bounded
def test_request_and_flight_limits_must_be_positive_integers(name, value):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        list(
            relaxer.relax_stream(
                enumerate(make_structures(1)), EMTModel(), **{name: value}
            )
        )


@pytest.mark.parametrize("mode", ["close", "throw", "consumer_error"])
@bounded
def test_cleanup_preserves_the_propagating_exception(monkeypatch, mode):
    baseline = set(threading.enumerate())
    ready = threading.Event()
    released = threading.Event()

    class InterruptingProxy(relaxer._ProxyCalculator):
        def __init__(self, condition, state, *args):
            super().__init__(condition, state, *args)
            if state.index == 0:
                raise RuntimeError("first outcome")
            ready.set()
            assert released.wait(5)
            raise KeyboardInterrupt("interrupt during cleanup")

    monkeypatch.setattr(relaxer, "_ProxyCalculator", InterruptingProxy)
    stream = relaxer.relax_stream(enumerate(make_structures(2)), EMTModel())
    try:
        assert next(stream).error == "first outcome"
        assert ready.wait(5)
        released.set()
        if mode == "close":
            stream.close()
        elif mode == "throw":
            with pytest.raises(ValueError, match="consumer failed"):
                stream.throw(ValueError("consumer failed"))
        else:
            with pytest.raises(ValueError, match="consumer failed"):
                with closing(stream):
                    raise ValueError("consumer failed")
    finally:
        released.set()
        stream.close()
    assert_threads_stopped(baseline)


@bounded
def test_worker_interrupt_propagates(monkeypatch):
    baseline = set(threading.enumerate())

    class InterruptingProxy(relaxer._ProxyCalculator):
        def __init__(self, *args):
            super().__init__(*args)
            raise KeyboardInterrupt("worker interrupted")

    monkeypatch.setattr(relaxer, "_ProxyCalculator", InterruptingProxy)
    with pytest.raises(KeyboardInterrupt, match="worker interrupted"):
        list(relaxer.relax_stream(enumerate(make_structures(2)), EMTModel()))
    assert_threads_stopped(baseline)
