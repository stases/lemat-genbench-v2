"""Batch model requests from independent, unmodified ASE relaxations."""

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from numbers import Integral
from threading import Condition, Lock, Thread

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes, compare_atoms
from ase.filters import FrechetCellFilter
from ase.optimize import BFGSLineSearch

from lemat_genbench.batched.types import BatchedModel, Prediction, RelaxOutcome


@dataclass(eq=False)
class _State:
    index: int
    atoms: Atoms
    wake: Condition
    request: Atoms | None = None
    reply: Prediction | Exception | None = None
    outcome: RelaxOutcome | BaseException | None = None
    model_calls: int = 0
    cache_hits: int = 0
    stopped: bool = False
    thread: Thread | None = None


class _ProxyCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, condition, state, geometry_tol, max_requests):
        super().__init__()
        self.condition = condition
        self.state = state
        self.geometry_tol = geometry_tol
        self.max_requests = max_requests
        self.requests = 0
        # ASE clears its cache before calculate(), so retain the last evaluation.
        self.evaluated_atoms = None
        self.prediction = None

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        with self.condition:
            if self.state.stopped:
                raise RuntimeError("relax_stream closed")
            if self.requests >= self.max_requests:
                raise RuntimeError(f"max_requests ({self.max_requests}) exceeded")
            self.requests += 1

        if self.geometry_tol is not None and not compare_atoms(
            self.evaluated_atoms,
            self.atoms,
            tol=self.geometry_tol,
            excluded_properties=set(self.ignored_changes),
        ):
            prediction = self.prediction
            with self.condition:
                self.state.cache_hits += 1
        else:
            with self.condition:
                self.state.request = self.atoms.copy()
                self.condition.notify()
                self.state.wake.wait_for(
                    lambda: self.state.reply is not None or self.state.stopped
                )
                if self.state.stopped:
                    raise RuntimeError("relax_stream closed")
                prediction = self.state.reply
                self.state.reply = None
            if isinstance(prediction, Exception):
                raise prediction.with_traceback(None)
            if not all(
                np.isfinite(value).all()
                for value in (prediction.energy, prediction.forces, prediction.stress)
            ):
                raise ValueError("Model prediction was not finite")
            if self.geometry_tol is not None:
                self.evaluated_atoms = self.atoms.copy()
                self.prediction = prediction

        self.results = {
            "energy": prediction.energy,
            "free_energy": prediction.energy,
            "forces": prediction.forces,
            "stress": prediction.stress,
        }


def _without_tracebacks(exc):
    """Drop the tracebacks along an exception chain, so the failed call's frames are freed."""
    seen = set()
    link = exc
    while link is not None and id(link) not in seen:
        seen.add(id(link))
        link.__traceback__ = None
        link = link.__cause__ or link.__context__
    return exc


def _evaluate(model, geometries):
    """Bisect failed calls until each request has a prediction or exception."""
    try:
        predictions = model.predict([atoms.copy() for atoms in geometries])
        if len(predictions) != len(geometries):
            raise ValueError("Model must return one prediction per structure")
        return predictions
    except Exception as exc:
        error = _without_tracebacks(exc)
    # Release the failed forward's frames before freeing CUDA memory or retrying.
    if isinstance(error, torch.OutOfMemoryError) and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if len(geometries) == 1:
        return [error]
    midpoint = len(geometries) // 2
    return _evaluate(model, geometries[:midpoint]) + _evaluate(
        model, geometries[midpoint:]
    )


def relax_stream(
    jobs: Iterable[tuple[int, Atoms]],
    model: BatchedModel,
    *,
    fmax: float = 0.02,
    steps: int = 50,
    max_in_flight: int = 32,
    max_atoms: int | None = None,
    geometry_tol: float | None = None,
    max_requests: int = 1000,
) -> Iterator[RelaxOutcome]:
    """Yield completed ASE relaxations with FIFO refill; cache hits count toward max_requests.

    If the job source raises or a thread raises a non-Exception BaseException,
    outcomes still in flight are dropped and the exception propagates to the caller.
    A thread's BaseException that has not reached the caller when the generator is
    closed or fails is dropped.
    """
    for name, value in (
        ("max_in_flight", max_in_flight),
        ("max_requests", max_requests),
    ):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    lock = Lock()
    condition = Condition(lock)
    finished = deque()
    active = {}
    iterator = iter(jobs)
    next_job = None
    exhausted = False
    in_flight_atoms = 0

    def relax(state):
        dyn = None
        endpoint = energy = error = fatal = None
        converged = False
        nsteps = 0
        try:
            atoms = state.atoms
            atoms.calc = _ProxyCalculator(condition, state, geometry_tol, max_requests)
            dyn = BFGSLineSearch(FrechetCellFilter(atoms), logfile=None)
            converged = dyn.run(fmax=fmax, steps=steps)
            energy = atoms.get_potential_energy()
            nsteps = dyn.get_number_of_steps()
            endpoint = atoms.copy()
            endpoint.calc = None
        except Exception as exc:
            endpoint = energy = None
            converged = False
            error = str(exc)
            nsteps = dyn.get_number_of_steps() if dyn is not None else 0
        except BaseException as exc:
            # Transport interrupts to the iterator's thread instead of losing them.
            fatal = exc
        finally:
            with condition:
                state.outcome = (
                    fatal
                    if fatal is not None
                    else RelaxOutcome(
                        state.index,
                        endpoint,
                        energy,
                        nsteps,
                        bool(converged),
                        error,
                        state.model_calls,
                        state.cache_hits,
                    )
                )
                finished.append(state)
                condition.notify()

    try:
        while True:
            with condition:
                if finished:
                    state = finished.popleft()
                    del active[state]
                    in_flight_atoms -= len(state.atoms)
                    outcome = state.outcome
                else:
                    outcome = None
            if outcome is not None:
                if isinstance(outcome, BaseException):
                    raise outcome
                yield outcome
                continue

            while len(active) < max_in_flight and not exhausted:
                if next_job is None:
                    try:
                        next_job = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                index, atoms = next_job
                if (
                    max_atoms is not None
                    and active
                    and in_flight_atoms + len(atoms) > max_atoms
                ):
                    break
                state = _State(index, atoms.copy(), wake=Condition(lock))
                state.thread = Thread(target=relax, args=(state,), daemon=True)
                active[state] = None
                in_flight_atoms += len(atoms)
                next_job = None
                state.thread.start()

            with condition:
                if not active:
                    break
                condition.wait_for(
                    lambda: finished
                    or all(state.request is not None for state in active)
                )
                if finished:
                    continue
                batch = [(state, state.request) for state in active]
                for state, _ in batch:
                    state.request = None

            replies = _evaluate(model, [atoms for _, atoms in batch])
            with condition:
                for (state, _), reply in zip(batch, replies):
                    state.model_calls += not isinstance(reply, Exception)
                    state.reply = reply
                    state.wake.notify()
    finally:
        with condition:
            for state in active:
                state.stopped = True
                state.wake.notify_all()
        for state in active:
            if state.thread.ident is not None:
                state.thread.join()
