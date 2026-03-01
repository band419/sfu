"""DUT adapter interfaces and registry for golden comparison."""

from .base import DUTAdapter, DUTResult
from .registry import (
    get_dut,
    list_duts,
    register_dut,
)

__all__ = [
    "DUTAdapter",
    "DUTResult",
    "register_dut",
    "get_dut",
    "list_duts",
]
