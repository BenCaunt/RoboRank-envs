from __future__ import annotations

import argparse
import json
import sys

from roborank_envs.catalog import CatalogError, get_challenge, list_challenges, validate_catalog
from roborank_envs.runner import EnvironmentRunError, run_policy_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roborank-envs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="validate packaged challenge data")
    subparsers.add_parser("list", help="list challenge ids")

    show_parser = subparsers.add_parser("show", help="print one challenge as JSON")
    show_parser.add_argument("challenge_id")

    export_parser = subparsers.add_parser("export", help="print the full catalog as JSON")
    export_parser.add_argument("--indent", type=int, default=2)

    run_parser = subparsers.add_parser("run", help="run a local policy file against an environment")
    run_parser.add_argument("challenge_id")
    run_parser.add_argument("--policy", required=True, help="path to a Python file defining RobotPolicy")
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--max-steps", type=int)
    run_parser.add_argument("--json", action="store_true", help="print the full RunResult JSON")

    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            challenges = validate_catalog()
            print(f"Loaded {len(challenges)} challenges.")
            return 0
        if args.command == "list":
            for challenge in list_challenges():
                print(challenge["id"])
            return 0
        if args.command == "show":
            challenge = get_challenge(args.challenge_id)
            if challenge is None:
                print(f"Unknown challenge id: {args.challenge_id}", file=sys.stderr)
                return 1
            print(json.dumps(challenge, indent=2))
            return 0
        if args.command == "export":
            print(json.dumps(list_challenges(), indent=args.indent))
            return 0
        if args.command == "run":
            result = run_policy_file(
                challenge_id=args.challenge_id,
                policy_path=args.policy,
                seed=args.seed,
                max_steps=args.max_steps,
            )
            if args.json:
                print(result.model_dump_json(indent=2))
            else:
                print(
                    f"{result.challenge_id}: status={result.metrics.status} "
                    f"success={str(result.metrics.success).lower()} score={result.metrics.score:.2f}"
                )
            return 0
    except (CatalogError, EnvironmentRunError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
