from __future__ import annotations

from typing import Dict

from .base import DUTAdapter


_DUT_REGISTRY: Dict[str, DUTAdapter] = {}


def register_dut(adapter: DUTAdapter) -> None:
    name = adapter.name.strip()
    if not name:
        raise ValueError("DUT adapter name must be non-empty")
    if name in _DUT_REGISTRY:
        raise ValueError(f"DUT adapter already registered: {name}")
    _DUT_REGISTRY[name] = adapter


def get_dut(name: str) -> DUTAdapter:
    try:
        return _DUT_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_DUT_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown DUT adapter '{name}'. Available: {available}") from exc


def list_duts() -> list[str]:
    return sorted(_DUT_REGISTRY)
