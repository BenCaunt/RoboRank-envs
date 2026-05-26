# RoboRank Envs

Open-source package for RoboRank robotics challenge environments.

This package exposes the RoboRank challenge catalog, policy-facing API, and
simulation runners so environments can be validated and run outside the main
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
uv sync
uv run python -m app.environment_registry
uv run pytest
```

The RoboRank backend pins this package by git commit in `backend/pyproject.toml`
and reports the active package version and git SHA from `/health` and run
metadata.

## Current Scope

- `.codex/skills/create-challenge/` contains the repo-local challenge authoring
  skill, including the brief template and authoring spec.
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

## Integration Status

The private RoboRank backend can execute submitted policies through this package.
The duplicated internal runner code should stay in RoboRank until a canary deploy
has proven the package path in production.

## Authoring Challenges

Use `.codex/skills/create-challenge/SKILL.md` as the agent workflow for adding
or modifying environments. It points to:

- `.codex/skills/create-challenge/references/challenge-authoring-spec.md`
- `.codex/skills/create-challenge/assets/challenge-brief-template.md`
