"""Read and query the frozen EVOP instance catalogue."""

from __future__ import annotations

import json
from pathlib import Path


def load_instance_bank(path: Path) -> tuple[dict, dict]:
    with path.open("r", encoding="utf-8") as file:
        bank = json.load(file)

    if "model" not in bank or "instances" not in bank:
        raise ValueError("Expected an instance bank with 'model' and 'instances'.")

    return bank["model"], bank["instances"]


def keys_with_size(raw_instances: dict, size: int) -> tuple[str, ...]:
    keys = [
        key
        for key, raw in raw_instances.items()
        if len(raw["customers"]) + len(raw["stations"]) == size
    ]
    return tuple(sorted(keys))


def development_key_for(
    size: int,
    raw_instances: dict,
    development_instances: dict[str, str],
) -> str:
    try:
        key = development_instances[str(size)]
    except KeyError as error:
        raise KeyError(
            f"No development instance is declared for N={size}."
        ) from error

    if key not in raw_instances:
        raise KeyError(f"Development instance {key!r} is not in the instance bank.")

    raw = raw_instances[key]
    actual_size = len(raw["customers"]) + len(raw["stations"])
    if actual_size != size:
        raise ValueError(
            f"Development instance {key} has N={actual_size}, expected N={size}."
        )

    return key


def transfer_keys_for(
    size: int,
    development_key: str,
    raw_instances: dict,
) -> tuple[str, ...]:
    return tuple(
        key
        for key in keys_with_size(raw_instances, size)
        if key != development_key
    )
