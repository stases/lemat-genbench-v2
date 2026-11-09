"""Memory failure classification and the atom budget shared by a worker's phases."""

from collections.abc import Callable
from dataclasses import dataclass

import torch


def memory_error(error) -> bool:
    """Recognize allocator failures, including CUDA library handle allocation."""
    if isinstance(error, torch.OutOfMemoryError):
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cublas_status_alloc_failed",
            "cublas_status_not_initialized",
            "cusolver_status_alloc_failed",
            "cudnn_status_alloc_failed",
        )
    ) or ("cusolver_status_internal_error" in message and "cusolverdncreate" in message)


@dataclass
class MemoryBudget:
    max_atoms: int | None = None
    on_error: Callable | None = None
    adaptive: bool = True

    def record_error(self, error):
        if memory_error(error) and self.on_error is not None:
            self.on_error(error)

    def lower(self, atoms):
        if self.adaptive and atoms:
            self.max_atoms = (
                atoms if self.max_atoms is None else min(self.max_atoms, atoms)
            )
