# RoboRank Envs

Private staging repo for RoboRank robotics challenge environments.

This package is intentionally small at first: it exposes the challenge catalog as
plain Python dictionaries that RoboRank validates against its private
`ChallengeSpec` schema. Simulation runners, scorer internals, and local replay
execution will move here after catalog parity is proven.

## Setup

```bash
uv sync
uv run roborank-envs validate
uv run roborank-envs list
```

## Use From RoboRank

From the RoboRank backend checkout:

```bash
uv pip install --python .venv/bin/python -e ../../RoboRank-envs
ROBORANK_ENV_CATALOG_MODULE=roborank_envs.catalog uv run python -m app.environment_registry
ROBORANK_ENV_CATALOG_MODULE=roborank_envs.catalog uv run pytest
```

Leaving `ROBORANK_ENV_CATALOG_MODULE` unset keeps RoboRank on its in-repo catalog.

## Current Scope

- `src/roborank_envs/catalog.py` exposes `list_challenges()` and
  `get_challenge()`.
- `src/roborank_envs/catalog_data.json` is a snapshot of the current RoboRank
  challenge catalog.
- `docs/challenges/` carries public-facing challenge docs.
- `samples/policies/` carries sample policies used for parity checks.

## Next Extraction Step

Move the runner-facing public APIs and simulation modules here, then add
`roborank-envs run <challenge_id> --policy <path>` for local challenge execution
without the private RoboRank app.
