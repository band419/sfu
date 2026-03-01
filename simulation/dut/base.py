from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


Mode = Literal["sin", "cos"]


@dataclass(frozen=True)
class DUTResult:
    """Unified return payload from a DUT model."""

    y_bits: int
    meta: dict[str, str] | None = None


class DUTAdapter(Protocol):
    """Interface contract for all DUT models.

    Each DUT must implement deterministic bit-level output for a reduced FP16 input.
    """

    name: str

    def eval(self, mode: Mode, x_bits: int) -> DUTResult:
        """Compute DUT output bits for one sample."""
        ...
