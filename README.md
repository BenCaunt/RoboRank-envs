# RoboRank Envs

Open-source package for [RoboRank](https://roborank.dev) robotics challenge
environments.

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

## Public CLI

The PyPI distribution is published as `roborank` and installs both the public
`roborank` CLI and the lower-level `roborank-envs` environment runner:

```bash
pip install roborank
```

The `roborank` CLI is the agent-friendly entry point for
[roborank.dev](https://roborank.dev) and for local challenge execution. It can:

- print a machine-readable workflow primer with `roborank prime`
- authenticate against the hosted app and persist a project-local token
- list, inspect, and run packaged robotics evals locally
- submit packaged evals to the hosted API
- search, read, create, and update RoboRank resource records
- initialize, validate, and upload evidence bundles with Rerun recordings
- fetch and validate environment-specific metrics schemas

Start by asking the CLI for the current command index:

```bash
uv run roborank prime --agent --json
```

Authenticate against [roborank.dev](https://roborank.dev) when you need hosted
resource, evidence, or eval submission workflows:

```bash
uv run roborank auth login
uv run roborank auth status --json
```

`roborank auth login` opens the hosted token page, prompts for the generated
token, and saves it in the current project at `.roborank/auth.json`. Later CLI
commands load that file automatically when no `--token`, `ROBORANK_TOKEN`, or
profile token is configured.

Run packaged evals locally before submitting them:

```bash
uv run roborank eval list --json
uv run roborank eval show diff_drive_reach_target --json
uv run roborank eval run diff_drive_reach_target --policy-source samples/policies/pure_pursuit.py --out runs/local-001 --json
```

Submit the same policy to the hosted RoboRank API:

```bash
uv run roborank eval submit diff_drive_reach_target --policy-source robot_policy.py --yes --non-interactive --json
```

`roborank prime` is intended as the model-facing index for standard RoboRank
procedures. It defines resource and eval boundaries, then points agents to the
commands they should use to discover canonical IDs, resource README markdown,
metrics schemas, and upload steps.

Resource management commands cover the normal read/write loop:

```bash
uv run roborank resources search --kind robot --query "flat disk" --json
uv run roborank resources read robot benkant/flat-disk-robot --json
uv run roborank resources readme robot benkant/flat-disk-robot --out README.md
uv run roborank resources create robot benkant/flat-disk-robot --title "Flat Disk Robot" --markdown README.md --yes --non-interactive --json
uv run roborank resources update robot benkant/flat-disk-robot --markdown README.md --yes --non-interactive --json
```

Metrics and evidence commands cover the upload path for Rerun recordings:

```bash
uv run roborank metrics schema --environment roborank/diff-drive-reach-target --json
uv run roborank metrics validate --environment roborank/diff-drive-reach-target metrics.json --json
uv run roborank evidence validate --from runs/local-001/evidence.json --json
uv run roborank evidence upload --from runs/local-001/evidence.json --yes --non-interactive --json
```

`roborank eval run` uses the packaged `roborank_envs.runner` local execution
path. Install the `visualization` extra when you need upload-ready Rerun `.rrd`
bundles:

```bash
uv sync --extra visualization
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
- `src/roborank/cli.py` exposes the public RoboRank CLI, including resource,
  metrics, evidence, hosted eval submit, and local eval run commands.
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
