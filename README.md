# RoboRank Envs

Private staging repo for RoboRank robotics challenge environments.

This package exposes the RoboRank challenge catalog, policy-facing API, and
simulation runners so environments can be validated and run outside the private
RoboRank application.

## Setup

```bash
uv sync
uv run roborank-envs validate
uv run roborank-envs list
ROBORANK_DISABLE_RERUN_EXPORT=1 uv run roborank-envs run diff_drive_reach_target --policy samples/policies/pure_pursuit.py
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
- `src/roborank_envs/simulation/` contains the environment runners, robot
  primitives, MuJoCo world builders, rendering helpers, and Rerun export path.
- `src/roborank_envs/policy_api.py` and `policy_loader.py` contain the public
  policy API and local policy loading helpers.
- `src/roborank_envs/runner.py` runs a local policy file against a packaged
  challenge.
- `docs/challenges/` carries public-facing challenge docs.
- `samples/policies/` carries sample policies used for parity checks.

## Next Extraction Step

Teach the private RoboRank backend to import runner classes from this package
behind a feature flag, then remove the duplicated runner code from RoboRank only
after a canary deploy proves the package path in production.
