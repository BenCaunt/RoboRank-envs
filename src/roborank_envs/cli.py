from __future__ import annotations

import argparse
import json
import sys

from roborank_envs.catalog import CatalogError, get_challenge, list_challenges, validate_catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roborank-envs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="validate packaged challenge data")
    subparsers.add_parser("list", help="list challenge ids")

    show_parser = subparsers.add_parser("show", help="print one challenge as JSON")
    show_parser.add_argument("challenge_id")

    export_parser = subparsers.add_parser("export", help="print the full catalog as JSON")
    export_parser.add_argument("--indent", type=int, default=2)

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
    except CatalogError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
