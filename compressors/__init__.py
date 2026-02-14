"""Compressors package — auto-discovers all BaseCompressor subclasses."""

import importlib
import pkgutil
from pathlib import Path

from compressors.base import BaseCompressor

# ── Registry ────────────────────────────────────────────────────────────────
_registry: dict[str, type[BaseCompressor]] = {}


def _discover() -> None:
    """Import every module in this package so subclasses get registered."""
    package_dir = Path(__file__).resolve().parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name == "base":
            continue
        importlib.import_module(f"compressors.{module_info.name}")

    # Collect all concrete subclasses of BaseCompressor
    for cls in BaseCompressor.__subclasses__():
        if cls.name:
            _registry[cls.name] = cls


def get_compressor(name: str) -> BaseCompressor:
    """Return an instance of the compressor with the given name."""
    if not _registry:
        _discover()
    if name not in _registry:
        available = ", ".join(_registry.keys())
        raise ValueError(f"Unknown compressor '{name}'. Available: {available}")
    return _registry[name]()


def list_compressors() -> list[str]:
    """Return the names of all available compressors."""
    if not _registry:
        _discover()
    return list(_registry.keys())
