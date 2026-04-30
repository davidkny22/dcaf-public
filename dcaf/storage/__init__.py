"""Storage APIs for DCAF run artifacts.

Exports are lazy to avoid importing torch/transformers when callers only import
the package namespace.
"""

_LAZY_EXPORTS = {
    "DeltaMetadata": ("dcaf.storage.delta_store", "DeltaMetadata"),
    "DeltaStore": ("dcaf.storage.delta_store", "DeltaStore"),
    "CheckpointManager": ("dcaf.storage.checkpoint", "CheckpointManager"),
    "NamedCheckpoint": ("dcaf.storage.checkpoint", "NamedCheckpoint"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)

    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
