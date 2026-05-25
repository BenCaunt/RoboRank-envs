from __future__ import annotations

import copy
import json
from functools import lru_cache
from importlib.resources import files
from typing import Any


CatalogEntry = dict[str, Any]


class CatalogError(RuntimeError):
    """Raised when packaged challenge data is malformed."""


def list_challenges() -> list[CatalogEntry]:
    return [copy.deepcopy(challenge) for challenge in _load_catalog()]


def get_challenge(challenge_id: str) -> CatalogEntry | None:
    for challenge in _load_catalog():
        if challenge["id"] == challenge_id:
            return copy.deepcopy(challenge)
    return None


def validate_catalog() -> list[CatalogEntry]:
    return list_challenges()


@lru_cache(maxsize=1)
def _load_catalog() -> tuple[CatalogEntry, ...]:
    resource = files("roborank_envs").joinpath("catalog_data.json")
    with resource.open("r", encoding="utf-8") as catalog_file:
        raw_catalog = json.load(catalog_file)

    if not isinstance(raw_catalog, list):
        raise CatalogError("catalog_data.json must contain a list of challenges.")

    ids: set[str] = set()
    catalog: list[CatalogEntry] = []
    required_keys = {
        "id",
        "title",
        "difficulty",
        "description",
        "robot",
        "sensors",
        "actuators",
        "objective",
        "success_conditions",
        "scoring",
        "defaults",
    }

    for index, challenge in enumerate(raw_catalog):
        if not isinstance(challenge, dict):
            raise CatalogError(f"Catalog entry {index} must be an object.")
        missing = required_keys - set(challenge)
        if missing:
            challenge_id = challenge.get("id", f"entry {index}")
            raise CatalogError(f"Challenge {challenge_id!r} is missing required keys: {sorted(missing)}.")
        challenge_id = challenge["id"]
        if not isinstance(challenge_id, str) or not challenge_id:
            raise CatalogError(f"Catalog entry {index} has an invalid id.")
        if challenge_id in ids:
            raise CatalogError(f"Duplicate challenge id {challenge_id!r}.")
        ids.add(challenge_id)
        catalog.append(challenge)

    return tuple(catalog)
