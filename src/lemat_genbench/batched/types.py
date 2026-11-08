"""Types shared by the batched model adapters, the relaxer and the stage."""

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from ase import Atoms


@dataclass
class Prediction:
    """Model output for one structure, in ASE conventions."""

    energy: float  # total energy, eV
    forces: np.ndarray  # (n_atoms, 3), eV/Å
    stress: np.ndarray  # (6,), Voigt order xx, yy, zz, yz, xz, xy, eV/Å³


class BatchedModel(Protocol):
    def predict(self, atoms: Sequence[Atoms]) -> list[Prediction]:
        """Evaluate all structures in one call; one prediction per input, in input order.

        Raises if the call as a whole fails (e.g. ``torch.OutOfMemoryError``). Non-finite
        values for a single structure are returned as they are; callers check them.
        """


@dataclass
class RelaxOutcome:
    """Result of relaxing one structure; mirrors ``BaseMLIPCalculator.relax_structure``."""

    index: int  # the index the job was submitted with
    atoms: Atoms | None  # endpoint positions and cell; None if the relaxation failed
    energy: float | None  # endpoint total energy, eV; None if the relaxation failed
    steps: int  # optimizer steps taken, as dyn.get_number_of_steps()
    converged: bool
    error: str | None  # None on success; the failure text otherwise
    model_calls: int  # predictions the model computed for this structure
    cache_hits: int  # requests answered by the geometry-tolerance check
