from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

from roborank_envs.catalog import get_challenge, list_challenges
from roborank_envs.runner import EnvironmentRunError, run_policy_file


MAX_RRD_BYTES = 40 * 1024 * 1024
MAX_METRICS_BYTES = 1024 * 1024
DEFAULT_API_URL = "https://roborank.dev"
DEFAULT_LICENSE = "CC-BY-4.0"
API_USER_AGENT = "RoboRank CLI/0.1.0 (+https://roborank.dev)"
ALLOWED_LICENSES = {"CC-BY-4.0", "CC0-1.0"}
RESOURCE_KINDS = {"robot", "environment", "policy", "policy_family"}
EVIDENCE_VISIBILITIES = {"public", "unlisted", "private"}
SOURCE_LINK_KINDS = {
    "source_repo",
    "source_commit",
    "model",
    "dataset",
    "robot_docs",
    "environment_docs",
    "policy_docs",
    "run_script",
    "paper",
    "video",
    "other",
}
SOURCE_LINK_KIND_ALIASES = {
    "github": "source_repo",
    "github_repo": "source_repo",
    "repo": "source_repo",
    "hf": "model",
    "huggingface": "model",
}
API_TOKEN_SCOPES = {
    "resources:read",
    "resources:write",
    "evidence:read",
    "evidence:write",
    "metrics:read",
    "metrics:write",
    "eval:run",
    "migration:admin",
}
CHALLENGE_RESOURCE_CACHE: dict[str, tuple[str, str]] = {
    "diff_drive_reach_target": ("roborank/differential-drive-cube-v1", "roborank/diff-drive-reach-target"),
    "diff_drive_odometry": ("roborank/differential-drive-cube-v1", "roborank/diff-drive-odometry"),
    "diff_drive_state_estimation": ("roborank/differential-drive-cube-v1", "roborank/diff-drive-state-estimation"),
    "diff_drive_2d_slam": ("roborank/differential-drive-cube-v1", "roborank/diff-drive-2d-slam"),
    "dock_to_charger": ("roborank/differential-drive-cube-v1", "roborank/dock-to-charger"),
    "warehouse_aisle_avoidance": ("roborank/differential-drive-cube-v1", "roborank/warehouse-aisle-avoidance"),
    "quadrotor_gate_sequence": ("roborank/quadrotor-x500-v1", "roborank/quadrotor-gate-sequence"),
    "imu_collision_detection": ("roborank/mobile-probe-v1", "roborank/imu-collision-detection"),
    "motor_torque_scale_control": ("roborank/motor-stand-scale-v1", "roborank/motor-torque-scale-control"),
    "trapezoidal_motion_profile": ("roborank/profiled-cart-1d-v1", "roborank/trapezoidal-motion-profile"),
    "inverse_kinematics_1": ("roborank/planar-2link-arm-v1", "roborank/inverse-kinematics-1"),
    "inverse_kinematics_2": ("roborank/turret-2link-arm-v1", "roborank/inverse-kinematics-2"),
    "inverse_kinematics_3": ("roborank/five-bar-scara-v1", "roborank/inverse-kinematics-3"),
    "cart_pole": ("roborank/cart-pole-cart-v1", "roborank/cart-pole"),
    "cart_pole_minimum_phase": ("roborank/cart-pole-cart-v1", "roborank/cart-pole-minimum-phase"),
    "diff_drive_lidar_maze": ("roborank/differential-drive-cube-v1", "roborank/diff-drive-lidar-maze"),
}
MIGRATION_RERUN_STATE_VERSION = "roborank.migration_rerun_state.v0"
D1_RESTORE_REQUIRED_TABLES = {
    "users",
    "sessions",
    "submissions",
    "resources",
    "resource_aliases",
    "evidence_runs",
    "run_artifacts",
    "run_metrics",
    "run_resource_tags",
    "run_source_links",
    "metrics_schemas",
}
GO_NO_GO_REQUIRED_SMOKE_CHECKS = {
    "auth",
    "problem_list",
    "problem_submit_evidence_creation",
    "direct_evidence_creation",
    "artifact_download",
    "explorer_anonymous_limit",
    "explorer_facets",
    "explorer_signed_in_pagination",
    "run_detail",
    "resource_page",
    "environment_leaderboard",
    "legacy_leaderboard",
    "legacy_submission_detail",
}
GO_NO_GO_REQUIRED_BROWSER_SURFACES = {
    "problem_page",
    "explorer_desktop",
    "explorer_mobile",
    "run_detail",
    "compare",
    "resource_page",
    "environment_leaderboard",
}
GO_NO_GO_REQUIRED_PRIVACY_CHECKS = {
    "explorer_public",
    "run_detail_public",
    "resource_page_public",
    "compare_anonymous_denied",
    "compare_authenticated",
}
GO_NO_GO_REQUIRED_SCHEMA_GUARD_CHECKS = {
    "metrics_schema_active",
    "missing_metrics_rejected",
    "invalid_metrics_rejected",
}
GO_NO_GO_REQUIRED_TRUST_ACTION_CHECKS = {
    "resource_page",
    "run_detail",
    "endorse",
    "endorsed_resource_context",
    "dispute",
    "disputed_resource_context",
    "trust_labels",
}
GO_NO_GO_REQUIRED_RESOURCE_GUARD_CHECKS = {
    "known_resources_validate",
    "missing_robot_rejected",
    "missing_environment_rejected",
}
GO_NO_GO_REQUIRED_SIGNOFF_ROLES = {"data_storage", "backend", "frontend", "product"}
CURATED_EVIDENCE_SCENARIOS = {
    "schema_backed": "valid schema-backed simulation",
    "missing_optional_metrics": "schema-backed run with missing optional metrics",
    "std_unknown": "std/unknown environment evidence",
    "failed_result": "failed robotics result evidence",
    "viewer_incompatible": "viewer-incompatible retained artifact",
}
CURATED_EVIDENCE_SCENARIO_ALIASES = {
    "valid_schema_backed_simulation": "schema_backed",
    "schema_backed_simulation": "schema_backed",
    "schema_backed": "schema_backed",
    "missing_optional_metrics": "missing_optional_metrics",
    "std_unknown": "std_unknown",
    "std_unknown_environment": "std_unknown",
    "failed_robotics_result": "failed_result",
    "failed_result": "failed_result",
    "viewer_incompatible_retained_artifact": "viewer_incompatible",
    "viewer_incompatible": "viewer_incompatible",
}
LAUNCH_WATCH_STAT_LABELS = {
    "uploadFailures7d": "upload failures",
    "metricsValidationFailures7d": "metrics validation failures",
    "storageFailures7d": "storage failures",
    "viewerFailures7d": "viewer load failures",
    "explorerSlowEvents7d": "slow explorer events",
    "migrationFailures7d": "migration failures",
    "rateLimitEvents7d": "rate limit events",
}
LAUNCH_PREFLIGHT_COMMANDS = {
    "npm_test": ["npm", "run", "test"],
    "cloudflare_typecheck": ["npm", "--prefix", "cloudflare", "run", "typecheck"],
    "cloudflare_build": ["npm", "run", "cf:build"],
    "d1_schema_import": ["sqlite3", ":memory:"],
}
GLOBAL_VALUE_FLAGS = {"--api-url", "--token", "--format", "--profile"}
GLOBAL_BOOL_FLAGS = {
    "--json",
    "--quiet",
    "--verbose",
    "--no-color",
    "--non-interactive",
    "--yes",
    "--fail-on-warning",
    "--dry-run",
}


class RoboRankError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "command_failed",
        exit_code: int = 1,
        details: dict[str, Any] | None = None,
        suggested_commands: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code
        self.details = details or {}
        self.suggested_commands = suggested_commands or []


@dataclass
class GlobalOptions:
    api_url: str
    token: str | None = None
    profile: str = "default"
    output_format: str = "human"
    quiet: bool = False
    verbose: bool = False
    no_color: bool = False
    non_interactive: bool = False
    yes: bool = False
    fail_on_warning: bool = False
    dry_run: bool = False


@dataclass
class Context:
    options: GlobalOptions
    stdout: TextIO
    stderr: TextIO
    warnings: list[dict[str, str]] = field(default_factory=list)

    @property
    def api_url(self) -> str:
        return self.options.api_url

    @property
    def json_output(self) -> bool:
        return self.options.output_format == "json"

    @property
    def yaml_output(self) -> bool:
        return self.options.output_format == "yaml"

    def warn(self, code: str, message: str) -> None:
        self.warnings.append({"code": code, "message": message})


class ApiClient:
    def __init__(self, api_url: str, token: str | None = None) -> None:
        self.api_url = api_url.rstrip("/")
        self.token = token

    def request_json(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self._url(path, params)
        payload = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json", "User-Agent": API_USER_AGENT}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, data=payload, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise api_error_from_http(exc) from exc
        except urllib.error.URLError as exc:
            raise RoboRankError(str(exc), code="api_unavailable", exit_code=7) from exc
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def request_bytes(self, path_or_url: str) -> bytes:
        url = path_or_url if path_or_url.startswith(("http://", "https://")) else self._url(path_or_url)
        headers = {"Accept": "application/octet-stream", "User-Agent": API_USER_AGENT}
        if self.token and url.startswith(self.api_url):
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise api_error_from_http(exc) from exc
        except urllib.error.URLError as exc:
            raise RoboRankError(str(exc), code="artifact_download_failed", exit_code=7) from exc

    def multipart(
        self,
        path: str,
        *,
        fields: dict[str, str],
        files: dict[str, tuple[Path, str]],
        attempts: int = 3,
    ) -> dict[str, Any]:
        boundary = f"roborank-{uuid.uuid4().hex}"
        parts: list[bytes] = []
        for name, value in fields.items():
            parts.append(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                    f"{value}\r\n"
                ).encode("utf-8")
            )
        for name, (path_value, content_type) in files.items():
            filename = path_value.name
            parts.append(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode("utf-8")
            )
            parts.append(path_value.read_bytes())
            parts.append(b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        payload = b"".join(parts)
        headers = {
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": API_USER_AGENT,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        last_error: RoboRankError | None = None
        for attempt in range(max(1, attempts)):
            request = urllib.request.Request(self._url(path), data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    raw = response.read()
                return json.loads(raw.decode("utf-8")) if raw else {}
            except urllib.error.HTTPError as exc:
                error = api_error_from_http(exc)
                last_error = error
                if not should_retry_http_error(error) or attempt + 1 >= attempts:
                    raise error from exc
            except urllib.error.URLError as exc:
                last_error = RoboRankError(str(exc), code="upload_failed", exit_code=7)
                if attempt + 1 >= attempts:
                    raise last_error from exc
            time.sleep(0.25 * (attempt + 1))
        if last_error:
            raise last_error
        raise RoboRankError("Upload failed.", code="upload_failed", exit_code=7)

    def _url(self, path: str, params: dict[str, Any] | None = None) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        url = f"{self.api_url}{normalized_path}"
        if params:
            clean_params = {key: value for key, value in params.items() if value not in (None, "")}
            if clean_params:
                url = f"{url}?{urllib.parse.urlencode(clean_params, doseq=True)}"
        return url


def api_error_from_http(exc: urllib.error.HTTPError) -> RoboRankError:
    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except Exception:
        payload = {}
    message = str(payload.get("detail") or payload.get("message") or f"API returned {exc.code}")
    if exc.code in (401, 403):
        return RoboRankError(message, code="auth_failed", exit_code=3, details={"status": exc.code})
    if exc.code == 404:
        return RoboRankError(message, code="resource_not_found", exit_code=4, details={"status": exc.code})
    return RoboRankError(message, code="api_error", exit_code=7, details={"status": exc.code})


def should_retry_http_error(error: RoboRankError) -> bool:
    status = error.details.get("status")
    return isinstance(status, int) and (status == 429 or status >= 500)


def config_path() -> Path:
    configured = os.environ.get("ROBORANK_CONFIG")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".config" / "roborank" / "config.toml"

def load_toml_file(path: Path, *, code: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise RoboRankError(f"{path} is not valid TOML: {exc}", code=code, exit_code=2) from exc
    if not isinstance(data, dict):
        return {}
    return data


def load_cli_config() -> dict[str, Any]:
    return load_toml_file(config_path(), code="invalid_config")


def project_config(base_dir: Path | None = None) -> dict[str, Any]:
    return load_toml_file((base_dir or Path.cwd()) / "roborank.toml", code="invalid_project_config")


def evidence_project_defaults(base_dir: Path | None = None) -> dict[str, Any]:
    evidence = project_config(base_dir).get("evidence")
    return evidence if isinstance(evidence, dict) else {}


def profile_config(config: dict[str, Any], profile: str) -> dict[str, Any]:
    result = {key: value for key, value in config.items() if key != "profiles"}
    profiles = config.get("profiles")
    if isinstance(profiles, dict):
        selected = profiles.get(profile)
        if isinstance(selected, dict):
            result.update(selected)
    return result


def config_string(config: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def config_bool(config: dict[str, Any], key: str) -> bool | None:
    value = config.get(key)
    if isinstance(value, bool):
        return value
    return None


def env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.lower() in {"1", "true", "yes", "on"}


def option_bool(flag: str, bools: set[str], config: dict[str, Any], key: str, env_name: str | None = None) -> bool:
    if flag in bools:
        return True
    if env_name:
        env_value = env_bool(env_name)
        if env_value is not None:
            return env_value
    config_value = config_bool(config, key)
    return bool(config_value)


def extract_global_options(argv: list[str]) -> tuple[GlobalOptions, list[str]]:
    values: dict[str, str] = {}
    bools: set[str] = set()
    filtered: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in GLOBAL_VALUE_FLAGS:
            if index + 1 >= len(argv):
                raise RoboRankError(f"{arg} requires a value.", code="usage_error", exit_code=2)
            values[arg] = argv[index + 1]
            index += 2
            continue
        if any(arg.startswith(f"{flag}=") for flag in GLOBAL_VALUE_FLAGS):
            flag, value = arg.split("=", 1)
            values[flag] = value
            index += 1
            continue
        if arg in GLOBAL_BOOL_FLAGS:
            bools.add(arg)
            index += 1
            continue
        filtered.append(arg)
        index += 1

    raw_config = load_cli_config()
    profile = values.get("--profile") or os.environ.get("ROBORANK_PROFILE") or config_string(raw_config, "profile") or "default"
    config = profile_config(raw_config, profile)

    output_format = values.get("--format") or os.environ.get("ROBORANK_FORMAT") or config_string(config, "format", "output_format") or "human"
    if "--json" in bools:
        output_format = "json"
    if output_format == "markdown":
        output_format = "human"
    if output_format not in {"human", "json", "yaml"}:
        raise RoboRankError("--format must be human, json, or yaml.", code="usage_error", exit_code=2)

    return (
        GlobalOptions(
            api_url=values.get("--api-url") or os.environ.get("ROBORANK_API_URL") or config_string(config, "api_url", "api-url") or DEFAULT_API_URL,
            token=values.get("--token") or os.environ.get("ROBORANK_TOKEN") or config_string(config, "token"),
            profile=profile,
            output_format=output_format,
            quiet=option_bool("--quiet", bools, config, "quiet"),
            verbose=option_bool("--verbose", bools, config, "verbose"),
            no_color=option_bool("--no-color", bools, config, "no_color", "ROBORANK_NO_COLOR"),
            non_interactive=option_bool("--non-interactive", bools, config, "non_interactive", "ROBORANK_NON_INTERACTIVE"),
            yes=option_bool("--yes", bools, config, "yes"),
            fail_on_warning=option_bool("--fail-on-warning", bools, config, "fail_on_warning"),
            dry_run=option_bool("--dry-run", bools, config, "dry_run"),
        ),
        filtered,
    )


def redact_token(value: Any, token: str | None) -> Any:
    if token is None:
        return value
    if isinstance(value, str):
        return value.replace(token, "[redacted]")
    if isinstance(value, list):
        return [redact_token(item, token) for item in value]
    if isinstance(value, dict):
        return {key: redact_token(item, token) for key, item in value.items()}
    return value


def yaml_key(value: Any) -> str:
    text = str(value)
    if text and all(char.isalnum() or char in "_-" for char in text):
        return text
    return json.dumps(text)


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return json.dumps(str(value))


def yaml_value(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, item in value.items():
            key_prefix = f"{prefix}{yaml_key(key)}:"
            if isinstance(item, (dict, list)) and item:
                lines.append(key_prefix)
                lines.extend(yaml_value(item, indent + 2))
            else:
                lines.append(f"{key_prefix} {yaml_value(item, 0)[0].strip()}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{prefix}-")
                lines.extend(yaml_value(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_value(item, 0)[0].strip()}")
        return lines
    return [f"{prefix}{yaml_scalar(value)}"]


def dump_yaml(value: Any) -> str:
    return "\n".join(yaml_value(value)) + "\n"


def emit_success(ctx: Context, command: str, result: dict[str, Any], warnings: list[dict[str, str]] | None = None) -> int:
    all_warnings = [*ctx.warnings, *(warnings or [])]
    if ctx.json_output:
        payload = {
            "ok": True,
            "command": command,
            "api_url": ctx.api_url,
            "warnings": all_warnings,
            "result": result,
        }
        print(json.dumps(redact_token(payload, ctx.options.token), indent=2, sort_keys=True), file=ctx.stdout)
    elif ctx.yaml_output:
        payload = {
            "ok": True,
            "command": command,
            "api_url": ctx.api_url,
            "warnings": all_warnings,
            "result": result,
        }
        print(dump_yaml(redact_token(payload, ctx.options.token)), file=ctx.stdout, end="")
    elif not ctx.options.quiet:
        if result:
            print(json.dumps(redact_token(result, ctx.options.token), indent=2, sort_keys=True), file=ctx.stdout)
        for warning in all_warnings:
            print(f"warning: {warning['message']}", file=ctx.stderr)
    if all_warnings and ctx.options.fail_on_warning:
        return 9
    return 0


def emit_error(ctx: Context, command: str, error: RoboRankError) -> int:
    payload = {
        "ok": False,
        "command": command,
        "error": {
            "code": error.code,
            "message": str(error),
            "details": error.details,
            "suggested_commands": error.suggested_commands,
        },
        "warnings": ctx.warnings,
    }
    if ctx.json_output:
        print(json.dumps(redact_token(payload, ctx.options.token), indent=2, sort_keys=True), file=ctx.stdout)
    elif ctx.yaml_output:
        print(dump_yaml(redact_token(payload, ctx.options.token)), file=ctx.stdout, end="")
    else:
        print(f"error: {error}", file=ctx.stderr)
    return error.exit_code


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RoboRankError(f"{path} does not exist.", code="file_missing", exit_code=2) from exc
    except json.JSONDecodeError as exc:
        raise RoboRankError(f"{path} is not valid JSON: {exc}", code="invalid_json", exit_code=2) from exc
    if not isinstance(data, dict):
        raise RoboRankError(f"{path} must contain a JSON object.", code="invalid_json", exit_code=2)
    return data


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_id_parts(resource_id: str) -> tuple[str, str]:
    parts = resource_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise RoboRankError(
            "Resource IDs must use <namespace>/<slug>.",
            code="invalid_resource_id",
            exit_code=2,
            details={"resource_id": resource_id},
        )
    return parts[0], parts[1]


def client(ctx: Context) -> ApiClient:
    return ApiClient(ctx.api_url, ctx.options.token)


def schema_from_response(payload: dict[str, Any]) -> dict[str, Any] | None:
    schema = payload.get("schema")
    if not isinstance(schema, dict):
        return None
    raw = schema.get("jsonSchema") or schema.get("json_schema")
    return raw if isinstance(raw, dict) else None


def template_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    output: dict[str, Any] = {}
    fields = [field for field in required if isinstance(field, str)]
    for field, spec in properties.items():
        if field not in fields and isinstance(spec, dict) and "default" in spec:
            fields.append(field)
    for field in fields:
        if not isinstance(field, str):
            continue
        spec = properties.get(field) if isinstance(properties.get(field), dict) else {}
        if "default" in spec:
            output[field] = spec["default"]
        elif spec.get("type") == "object":
            output[field] = {}
        elif spec.get("type") == "array":
            output[field] = []
        else:
            output[field] = None
    return output


def validate_json_schema(data: Any, schema: dict[str, Any], pointer: str = "") -> list[str]:
    errors: list[str] = []
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        types = schema_type
    else:
        types = [schema_type] if schema_type else []
    if types and not any(json_type_matches(data, schema_type_item) for schema_type_item in types):
        errors.append(f"{pointer or '/'} expected {types[0]}")
        return errors
    if isinstance(data, dict):
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        for field in required:
            if isinstance(field, str) and field not in data:
                errors.append(f"{pointer}/{field} is required")
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for field, child_schema in properties.items():
            if field in data and isinstance(child_schema, dict):
                errors.extend(validate_json_schema(data[field], child_schema, f"{pointer}/{field}"))
    if "enum" in schema and isinstance(schema["enum"], list) and data not in schema["enum"]:
        errors.append(f"{pointer or '/'} must be one of {schema['enum']}")
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and data < minimum:
            errors.append(f"{pointer or '/'} must be >= {minimum}")
        if isinstance(maximum, (int, float)) and data > maximum:
            errors.append(f"{pointer or '/'} must be <= {maximum}")
    return errors


def json_type_matches(value: Any, expected: Any) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def parse_set_value(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise RoboRankError("--set must use /json/pointer=value.", code="usage_error", exit_code=2)
    pointer, value = raw.split("=", 1)
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        parsed_value = value
    return [part for part in pointer.split("/") if part], parsed_value


def schema_has_property(schema: dict[str, Any] | None, name: str) -> bool:
    properties = schema.get("properties") if isinstance(schema, dict) and isinstance(schema.get("properties"), dict) else {}
    return name in properties


def apply_set(data: dict[str, Any], raw: str, schema: dict[str, Any] | None = None) -> None:
    parts, value = parse_set_value(raw)
    if schema is not None and parts[:1] == ["metrics"] and not schema_has_property(schema, "metrics"):
        parts = parts[1:]
    if not parts:
        raise RoboRankError("--set cannot target the document root.", code="usage_error", exit_code=2)
    current: dict[str, Any] = data
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def merge_metric_defaults(defaults: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in data.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_metric_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


def metrics_setup_source_data(path: Path, schema: dict[str, Any] | None) -> dict[str, Any]:
    raw = load_json(path)
    metrics = raw.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    metrics_path = raw.get("metrics_path") or raw.get("metricsPath")
    if isinstance(metrics_path, str) and metrics_path:
        resolved = Path(metrics_path)
        if not resolved.is_absolute():
            resolved = path.parent / resolved
        return load_json(resolved)
    return raw


def command_prime(ctx: Context, args: argparse.Namespace) -> int:
    primer = {
        "primer_version": "roborank-agent-primer-v0",
        "model": {
            "primary_artifact": "recording.rrd",
            "required_tags": ["robot", "environment", "policy"],
            "optional_tags": ["policy_family"],
            "resource_id_grammar": "<namespace>/<slug>",
            "max_rrd_bytes": MAX_RRD_BYTES,
            "default_license": DEFAULT_LICENSE,
            "allowed_licenses": sorted(ALLOWED_LICENSES),
        },
        "rules": [
            "One evidence run is one Rerun .rrd recording.",
            "Do not fabricate RoboRank resource IDs; search or resolve them through the API.",
            "Validate metrics.json before upload when the environment has an active schema.",
            "Do not upload policy source code as public evidence through evidence upload.",
        ],
        "commands": {
            "resource_search": 'roborank resources search --kind robot --query "<name>" --json',
            "resource_resolve": "roborank resources resolve robot <namespace>/<slug> --json",
            "metrics_schema": "roborank metrics schema --environment <namespace>/<slug> --json",
            "metrics_init": "roborank metrics init --environment <namespace>/<slug> --out metrics.json",
            "metrics_validate": "roborank metrics validate --environment <namespace>/<slug> metrics.json --json",
            "evidence_upload": (
                "roborank evidence upload --rrd recording.rrd --metrics metrics.json "
                "--robot <namespace>/<slug> --environment <namespace>/<slug> --policy <namespace>/<slug> "
                "--license CC-BY-4.0 --yes --non-interactive --json"
            ),
        },
        "task": args.task,
        "environment": args.environment,
        "challenge": args.challenge,
    }
    if ctx.json_output or ctx.yaml_output:
        return emit_success(ctx, "prime", primer)
    text = f"""# RoboRank Agent Primer

One evidence run is one Rerun `.rrd` recording. Required tags are robot,
environment, and exact policy. Policy family is optional but recommended.

Do not fabricate RoboRank resource IDs. Use:

```text
roborank resources search --kind robot --query "<name>" --json
roborank resources resolve robot <namespace>/<slug> --json
```

Before uploading evidence, ensure the Rerun file is a single `.rrd` under 40 MB,
check the environment metrics schema, validate metrics when required, and upload
with an explicit public artifact license.
"""
    print(textwrap.dedent(text).strip(), file=ctx.stdout)
    return 0


def command_auth(ctx: Context, args: argparse.Namespace) -> int:
    if args.auth_command == "status":
        payload = client(ctx).request_json("GET", "/api/auth/cli/status")
        return emit_success(
            ctx,
            "auth.status",
            {
                "authenticated": bool(payload.get("authenticated")),
                "token_configured": bool(ctx.options.token),
                "api_url": ctx.api_url,
                "profile": ctx.options.profile,
                "authSource": payload.get("authSource"),
                "user": payload.get("user"),
            },
        )
    if args.auth_command == "token_create":
        invalid_scopes = sorted(set(args.scope) - API_TOKEN_SCOPES)
        if invalid_scopes:
            raise RoboRankError(
                "Unsupported token scope.",
                code="invalid_token_scope",
                exit_code=2,
                details={"invalid_scopes": invalid_scopes, "allowed_scopes": sorted(API_TOKEN_SCOPES)},
            )
        if not args.scope:
            raise RoboRankError(
                "auth token create requires at least one --scope.",
                code="missing_token_scope",
                exit_code=2,
                details={"allowed_scopes": sorted(API_TOKEN_SCOPES)},
            )
        payload = client(ctx).request_json(
            "POST",
            "/api/auth/tokens",
            body={"name": args.name, "scopes": sorted(set(args.scope))},
        )
        return emit_success(ctx, "auth.token.create", payload)
    if args.auth_command == "logout":
        return emit_success(ctx, "auth.logout", {"message": "Unset ROBORANK_TOKEN or remove it from your profile config."})
    login_url = f"{ctx.api_url.rstrip('/')}/api/auth/cli/login"
    opened_browser = False
    if not getattr(args, "no_browser", False):
        opened_browser = webbrowser.open(login_url)
    return emit_success(
        ctx,
        "auth.login",
        {
            "message": "Create a personal access token in the browser, then pass it with --token or ROBORANK_TOKEN.",
            "login_url": login_url,
            "no_browser": bool(getattr(args, "no_browser", False)),
            "opened_browser": opened_browser,
        },
    )


def resource_result_command(ctx: Context, args: argparse.Namespace, command: str) -> int:
    params = {
        "kind": getattr(args, "kind", None),
        "namespace": getattr(args, "namespace", None),
        "q": getattr(args, "query", None),
        "limit": getattr(args, "limit", 25),
    }
    payload = client(ctx).request_json("GET", "/api/resources", params=params)
    return emit_success(ctx, command, payload)


def command_resources(ctx: Context, args: argparse.Namespace) -> int:
    api = client(ctx)
    if args.resources_command in {"search", "list"}:
        return resource_result_command(ctx, args, f"resources.{args.resources_command}")
    if args.resources_command in {"show", "resolve"}:
        namespace, slug = canonical_id_parts(args.resource_id)
        if args.kind not in RESOURCE_KINDS:
            raise RoboRankError("Invalid resource kind.", code="usage_error", exit_code=2)
        path = (
            "/api/resources/resolve"
            if args.resources_command == "resolve"
            else f"/api/resources/{args.kind}/{namespace}/{slug}"
        )
        params = {"kind": args.kind, "id": args.resource_id} if args.resources_command == "resolve" else None
        payload = api.request_json("GET", path, params=params)
        return emit_success(ctx, f"resources.{args.resources_command}", payload)
    namespace, slug = canonical_id_parts(args.resource_id)
    body_namespace, body_slug = canonical_id_parts(args.new_id) if getattr(args, "new_id", None) else (namespace, slug)
    if ctx.options.non_interactive and not ctx.options.yes:
        raise RoboRankError(
            "Non-interactive resource mutation requires --yes.",
            code="confirmation_required",
            exit_code=2,
        )
    body = {
        "kind": args.kind,
        "namespace": body_namespace,
        "slug": body_slug,
        "displayName": args.title,
        "summary": args.summary,
    }
    if args.markdown:
        body["markdown"] = Path(args.markdown).read_text()
    if args.resources_command == "create":
        payload = api.request_json("POST", "/api/resources", body=body)
    else:
        payload = api.request_json("PATCH", f"/api/resources/{args.kind}/{namespace}/{slug}", body=body)
    return emit_success(ctx, f"resources.{args.resources_command}", payload)


def fetch_metrics_schema(ctx: Context, environment: str) -> dict[str, Any]:
    namespace, slug = canonical_id_parts(environment)
    return client(ctx).request_json("GET", f"/api/environments/{namespace}/{slug}/metrics-schema")


def command_metrics(ctx: Context, args: argparse.Namespace) -> int:
    if args.metrics_command == "schema":
        payload = fetch_metrics_schema(ctx, args.environment)
        schema = schema_from_response(payload)
        if schema is None:
            ctx.warn(
                "environment_has_no_metrics_schema",
                "metrics.json is optional and this run will not be eligible for a schema-backed leaderboard.",
            )
        if args.out and schema is not None:
            write_json(Path(args.out), schema)
        return emit_success(ctx, "metrics.schema", payload)

    payload = fetch_metrics_schema(ctx, args.environment)
    schema = schema_from_response(payload)
    if args.metrics_command == "init":
        result = template_from_schema(schema) if schema else {}
        write_json(Path(args.out), result)
        if args.instructions and schema:
            Path(args.instructions).write_text(metrics_instructions(schema))
        return emit_success(ctx, "metrics.init", {"path": args.out, "metrics_required": schema is not None})

    if args.metrics_command == "setup":
        defaults = template_from_schema(schema) if schema else {}
        source_data = metrics_setup_source_data(Path(args.from_path), schema) if args.from_path else {}
        data = merge_metric_defaults(defaults, source_data)
        for item in args.set_values or []:
            apply_set(data, item, schema)
        if schema:
            errors = validate_json_schema(data, schema)
            if errors:
                raise RoboRankError(
                    "metrics.json is missing required values or is invalid.",
                    code="metrics_schema_validation_failed",
                    exit_code=5,
                    details={"errors": errors, "environment": args.environment},
                )
        write_json(Path(args.out), data)
        return emit_success(ctx, "metrics.setup", {"path": args.out, "metrics_required": schema is not None})

    if args.metrics_command in {"validate", "explain"}:
        data = load_json(Path(args.metrics_path)) if args.metrics_path else {}
        if schema is None:
            ctx.warn(
                "environment_has_no_metrics_schema",
                "metrics.json is optional and this run will not be eligible for a schema-backed leaderboard.",
            )
            return emit_success(ctx, f"metrics.{args.metrics_command}", {"valid": True, "schema": None})
        schema_info = payload.get("schema") if isinstance(payload.get("schema"), dict) else {}
        if args.schema_hash and args.schema_hash != schema_info.get("schemaHash"):
            raise RoboRankError(
                "Active metrics schema hash does not match --schema-hash.",
                code="metrics_schema_hash_mismatch",
                exit_code=5,
                details={"expected": args.schema_hash, "actual": schema_info.get("schemaHash")},
            )
        errors = validate_json_schema(data, schema)
        result = {
            "valid": not errors,
            "errors": errors,
            "schema": schema_info,
            "required": schema.get("required", []),
        }
        if errors:
            raise RoboRankError(
                "metrics.json does not validate against the active environment schema.",
                code="metrics_schema_validation_failed",
                exit_code=5,
                details=result,
                suggested_commands=[
                    f"roborank metrics explain --environment {args.environment} {args.metrics_path or 'metrics.json'} --json"
                ],
            )
        return emit_success(ctx, f"metrics.{args.metrics_command}", result)
    raise RoboRankError("Unknown metrics command.", code="usage_error", exit_code=2)


def metrics_instructions(schema: dict[str, Any]) -> str:
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    lines = ["# RoboRank metrics template", "", "Fill measured values for required fields:"]
    for field in required if isinstance(required, list) else []:
        spec = properties.get(field, {}) if isinstance(properties, dict) else {}
        description = spec.get("description", "") if isinstance(spec, dict) else ""
        lines.append(f"- `{field}` {description}".rstrip())
    return "\n".join(lines) + "\n"


def first_string(*values: Any, default: str | None = None) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return default


def envelope_tags(envelope: dict[str, Any]) -> dict[str, Any]:
    tags = envelope.get("tags")
    return tags if isinstance(tags, dict) else {}


def evidence_from_args(args: argparse.Namespace) -> tuple[dict[str, Any], Path | None, Path | None]:
    if args.from_path:
        from_path = Path(args.from_path)
        envelope = load_json(from_path)
        base_dir = from_path.parent
    else:
        envelope = {}
        base_dir = Path.cwd()
    defaults = evidence_project_defaults(Path.cwd())
    tags = envelope_tags(envelope)
    rrd_raw = args.rrd or envelope.get("rrd") or envelope.get("recording_path", "")
    metrics_raw = args.metrics or envelope.get("metrics") or envelope.get("metrics_path", "")
    rrd = resolve_bundle_path(base_dir, rrd_raw) if rrd_raw else None
    metrics = resolve_bundle_path(base_dir, metrics_raw) if metrics_raw else None
    license_value = first_string(args.license, envelope.get("license"), defaults.get("license"), default=DEFAULT_LICENSE)
    visibility = validate_evidence_visibility(
        first_string(args.visibility, envelope.get("visibility"), defaults.get("visibility"), default="public")
    )
    client_upload_id = args.client_upload_id or envelope.get("client_upload_id") or envelope.get("clientUploadId")
    metadata = {
        "schema_version": "roborank.rerun_evidence.v0",
        "client_upload_id": client_upload_id,
        "title": first_string(args.title, envelope.get("title"), defaults.get("title")),
        "summary": first_string(args.summary, envelope.get("summary"), defaults.get("summary")),
        "notes": first_string(args.notes, envelope.get("notes"), defaults.get("notes")),
        "superseded_by_run_id": first_string(
            args.superseded_by_run_id,
            envelope.get("superseded_by_run_id"),
            envelope.get("supersededByRunId"),
            defaults.get("superseded_by_run_id"),
            defaults.get("supersededByRunId"),
        ),
        "tags": {
            "robot": first_string(args.robot, tags.get("robot"), envelope.get("robot"), defaults.get("robot")),
            "environment": first_string(args.environment, tags.get("environment"), envelope.get("environment"), defaults.get("environment")),
            "policy": first_string(args.policy, tags.get("policy"), envelope.get("policy"), defaults.get("policy")),
            "policy_family": first_string(
                args.policy_family,
                tags.get("policy_family"),
                tags.get("policyFamily"),
                envelope.get("policy_family"),
                envelope.get("policyFamily"),
                defaults.get("policy_family"),
                defaults.get("policyFamily"),
            ),
        },
        "run": {
            "mode": first_string(args.run_mode, envelope.get("run_mode"), defaults.get("run_mode"), default="unknown"),
            "result_status": first_string(
                args.result_status,
                envelope.get("result_status"),
                defaults.get("result_status"),
                default="unknown",
            ),
        },
        "license": {
            "artifact": license_value,
            "metadata": license_value,
            "confirmed": False,
        },
        "visibility": visibility,
        "allow_new_policy": bool(args.allow_new_policy or envelope.get("allow_new_policy") or envelope.get("allowNewPolicy")),
        "allow_new_policy_family": bool(
            args.allow_new_policy_family or envelope.get("allow_new_policy_family") or envelope.get("allowNewPolicyFamily")
        ),
        "links": evidence_links(args.source_link or [], envelope, defaults),
    }
    migration_fields = {
        "source_kind": (envelope.get("source_kind"), envelope.get("sourceKind")),
        "legacy_submission_id": (envelope.get("legacy_submission_id"), envelope.get("legacySubmissionId")),
        "legacy_score": (envelope.get("legacy_score"), envelope.get("legacyScore")),
        "legacy_metrics_json": (
            envelope.get("legacy_metrics_json"),
            envelope.get("legacyMetricsJson"),
            envelope.get("legacy_metrics"),
            envelope.get("legacyMetrics"),
        ),
        "regenerated_score": (envelope.get("regenerated_score"), envelope.get("regeneratedScore")),
        "regenerated_metrics_json": (
            envelope.get("regenerated_metrics_json"),
            envelope.get("regeneratedMetricsJson"),
            envelope.get("regenerated_metrics"),
            envelope.get("regeneratedMetrics"),
        ),
        "migration_method": (
            envelope.get("migration_method"),
            envelope.get("migrationMethod"),
            envelope.get("migration_provenance"),
            envelope.get("migrationProvenance"),
        ),
        "migration_status": (envelope.get("migration_status"), envelope.get("migrationStatus")),
        "migration_notes": (envelope.get("migration_notes"), envelope.get("migrationNotes")),
        "legacy_code_hash": (envelope.get("legacy_code_hash"), envelope.get("legacyCodeHash")),
    }
    for key, values in migration_fields.items():
        for value in values:
            if value is None or value == "":
                continue
            metadata[key] = value
            break
    return metadata, rrd, metrics


def resolve_bundle_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def validate_evidence_visibility(value: str | None) -> str:
    visibility = value or "public"
    if visibility not in EVIDENCE_VISIBILITIES:
        raise RoboRankError(
            "Evidence visibility must be public, unlisted, or private.",
            code="usage_error",
            exit_code=2,
            details={"visibility": visibility, "allowed": sorted(EVIDENCE_VISIBILITIES)},
        )
    return visibility


def validate_metrics_artifact_path(path: Path) -> None:
    if not path.exists():
        raise RoboRankError(f"{path} does not exist.", code="metrics_missing", exit_code=5)
    if path.suffix.lower() != ".json":
        raise RoboRankError("metrics artifact must be a .json file.", code="metrics_invalid", exit_code=5)
    size_bytes = path.stat().st_size
    if size_bytes > MAX_METRICS_BYTES:
        raise RoboRankError(
            "metrics.json exceeds the 1 MB upload limit.",
            code="metrics_too_large",
            exit_code=5,
            details={"size_bytes": size_bytes, "max_bytes": MAX_METRICS_BYTES},
        )


def source_links(values: Iterable[str]) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for value in values:
        if "=" in value:
            kind, url = value.split("=", 1)
        else:
            kind, url = "other", value
        kind = normalize_source_link_kind(kind)
        validate_source_link_url(url)
        links.append({"kind": kind, "label": kind, "url": url})
    return links


def normalize_source_link_kind(value: Any) -> str:
    kind = first_string(value, default="other") or "other"
    kind = SOURCE_LINK_KIND_ALIASES.get(kind, kind)
    if kind not in SOURCE_LINK_KINDS:
        raise RoboRankError(
            "Unsupported source link kind.",
            code="usage_error",
            exit_code=2,
            details={"kind": kind, "allowed": sorted(SOURCE_LINK_KINDS), "aliases": SOURCE_LINK_KIND_ALIASES},
        )
    return kind


def validate_source_link_url(value: str) -> None:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RoboRankError("Source links must be valid http(s) URLs.", code="usage_error", exit_code=2)


def source_links_from_objects(values: Any) -> list[dict[str, str | None]]:
    if not isinstance(values, list):
        return []
    links: list[dict[str, str | None]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        url = first_string(value.get("url"))
        if not url:
            continue
        validate_source_link_url(url)
        kind = normalize_source_link_kind(value.get("kind"))
        links.append(
            {
                "kind": kind,
                "label": first_string(value.get("label"), default=kind),
                "url": url,
                "revision": first_string(value.get("revision")),
            }
        )
    return links


def evidence_links(cli_values: Iterable[str], envelope: dict[str, Any], defaults: dict[str, Any]) -> list[dict[str, str | None]]:
    cli_links = source_links(cli_values)
    if cli_links:
        return cli_links
    envelope_links = source_links_from_objects(envelope.get("links") or envelope.get("source_links"))
    if envelope_links:
        return envelope_links
    return source_links_from_objects(defaults.get("source_links"))


def missing_source_revision_warnings(links: Iterable[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for link in links:
        revision = link.get("revision")
        if isinstance(revision, str) and revision:
            continue
        url = link.get("url")
        if not isinstance(url, str):
            continue
        hostname = urllib.parse.urlparse(url).hostname or ""
        hostname = hostname.lower()
        if hostname == "github.com" or hostname.endswith(".github.com"):
            warnings.append(f"source link {url} is not pinned to a GitHub commit SHA.")
        elif hostname == "huggingface.co" or hostname.endswith(".huggingface.co"):
            warnings.append(f"source link {url} is not pinned to a Hugging Face revision.")
    return warnings


def preflight_evidence(ctx: Context, args: argparse.Namespace, *, require_confirmation: bool) -> dict[str, Any]:
    if getattr(args, "policy_source", None):
        raise RoboRankError(
            "evidence upload does not accept policy source code; use eval submit for challenge submissions.",
            code="policy_source_not_allowed",
            exit_code=2,
        )
    metadata, rrd, metrics_path = evidence_from_args(args)
    if require_confirmation and ctx.options.non_interactive and not ctx.options.yes:
        raise RoboRankError("Evidence upload requires --yes in non-interactive mode.", code="confirmation_required", exit_code=2)
    metadata["license"]["confirmed"] = bool(ctx.options.yes)
    if not rrd:
        raise RoboRankError("--rrd or evidence recording_path is required.", code="rerun_artifact_missing", exit_code=6)
    if not rrd.exists():
        raise RoboRankError(f"{rrd} does not exist.", code="rerun_artifact_missing", exit_code=6)
    if rrd.suffix.lower() != ".rrd":
        raise RoboRankError("Rerun artifact must have a .rrd extension.", code="rerun_artifact_invalid", exit_code=6)
    size_bytes = rrd.stat().st_size
    if size_bytes > MAX_RRD_BYTES:
        raise RoboRankError(
            "Rerun artifact exceeds the 40 MB v0 upload limit.",
            code="rerun_artifact_too_large",
            exit_code=6,
            details={"size_bytes": size_bytes, "max_bytes": MAX_RRD_BYTES},
        )
    tags = metadata["tags"]
    for kind in ("robot", "environment", "policy"):
        if not tags.get(kind):
            raise RoboRankError(f"{kind} tag is required.", code="missing_required_tag", exit_code=2)
    license_value = metadata["license"]["artifact"]
    if license_value not in ALLOWED_LICENSES:
        raise RoboRankError("Artifact license must be CC-BY-4.0 or CC0-1.0.", code="invalid_license", exit_code=2)

    api = client(ctx)
    for kind in ("robot", "environment"):
        resolve_resource(api, kind, tags[kind])
    try:
        resolve_resource(api, "policy", tags["policy"])
    except RoboRankError:
        if not metadata.get("allow_new_policy"):
            raise
        ctx.warn("policy_will_be_created", "Exact policy resource will be created during upload if the server permits it.")
    if tags.get("policy_family"):
        try:
            resolve_resource(api, "policy_family", tags["policy_family"])
        except RoboRankError:
            if not metadata.get("allow_new_policy_family"):
                raise
            ctx.warn("policy_family_will_be_created", "Policy family resource will be created during upload if the server permits it.")
    else:
        ctx.warn("policy_family_missing", "policy_family is omitted; explorer grouping will be less useful.")
    for warning in missing_source_revision_warnings(metadata.get("links", [])):
        ctx.warn("source_link_revision_missing", warning)

    schema_payload = fetch_metrics_schema(ctx, tags["environment"])
    schema = schema_from_response(schema_payload)
    metrics: dict[str, Any] | None = None
    if metrics_path:
        validate_metrics_artifact_path(metrics_path)
    if schema is not None:
        if not metrics_path:
            raise RoboRankError(
                "metrics.json is required for this schema-backed environment.",
                code="metrics_missing",
                exit_code=5,
                details={"environment": tags["environment"]},
            )
        metrics = load_json(metrics_path)
        errors = validate_json_schema(metrics, schema)
        if errors:
            raise RoboRankError(
                "metrics.json does not validate against the active environment schema.",
                code="metrics_schema_validation_failed",
                exit_code=5,
                details={"errors": errors, "environment": tags["environment"]},
            )
    elif metrics_path:
        metrics = load_json(metrics_path)
        ctx.warn(
            "environment_has_no_metrics_schema",
            "metrics.json will be uploaded as self-reported metrics and is not leaderboard-eligible.",
        )
    else:
        ctx.warn(
            "environment_has_no_metrics_schema",
            "environment has no active metrics schema; metrics.json is optional.",
        )
    if tags["environment"] == "std/unknown":
        ctx.warn("std_unknown_environment", "std/unknown limits reproducibility and leaderboard usability.")
    preflight = {
        "metadata": metadata,
        "rrd_path": str(rrd),
        "metrics_path": str(metrics_path) if metrics_path else None,
        "recording": {
            "sha256": sha256_file(rrd),
            "size_bytes": size_bytes,
        },
        "metrics": {
            "present": metrics is not None,
            "schema": schema_payload.get("schema"),
        },
        "tags": tags,
        "license": license_value,
    }
    preflight["server_validation"] = server_validate_evidence(ctx, metadata, metrics)
    return preflight


def server_validate_evidence(
    ctx: Context,
    metadata: dict[str, Any],
    metrics: dict[str, Any] | None,
) -> dict[str, Any] | None:
    try:
        return client(ctx).request_json(
            "POST",
            "/api/evidence-runs/validate",
            body={"metadata": metadata, "metrics": metrics},
        )
    except RoboRankError as exc:
        if exc.details.get("status") == 404:
            ctx.warn("server_validation_unavailable", "Server does not expose /api/evidence-runs/validate.")
            return None
        raise


def resolve_resource(api: ApiClient, kind: str, resource_id: str) -> dict[str, Any]:
    namespace, slug = canonical_id_parts(resource_id)
    try:
        return api.request_json("GET", "/api/resources/resolve", params={"kind": kind, "id": f"{namespace}/{slug}"})
    except RoboRankError as exc:
        if exc.exit_code == 4:
            raise RoboRankError(
                f"{kind} resource {resource_id} was not found.",
                code="resource_not_found",
                exit_code=4,
                details={"kind": kind, "resource_id": resource_id},
            ) from exc
        raise


def command_evidence(ctx: Context, args: argparse.Namespace) -> int:
    if args.evidence_command == "init":
        output = Path(args.out)
        defaults = evidence_project_defaults(Path.cwd())
        envelope = {
            "schema_version": "roborank.rerun_evidence.v0",
            "title": first_string(args.title, defaults.get("title")),
            "summary": first_string(args.summary, defaults.get("summary")),
            "robot": first_string(args.robot, defaults.get("robot")),
            "environment": first_string(args.environment, defaults.get("environment")),
            "policy": first_string(args.policy, defaults.get("policy")),
            "policy_family": first_string(args.policy_family, defaults.get("policy_family"), defaults.get("policyFamily")),
            "recording_path": first_string(defaults.get("recording_path"), default="recording.rrd"),
            "metrics_path": first_string(defaults.get("metrics_path"), default="metrics.json"),
            "run_mode": first_string(args.run_mode, defaults.get("run_mode"), default="unknown"),
            "result_status": first_string(args.result_status, defaults.get("result_status"), default="unknown"),
            "license": first_string(args.license, defaults.get("license"), default=DEFAULT_LICENSE),
            "visibility": validate_evidence_visibility(first_string(args.visibility, defaults.get("visibility"), default="public")),
        }
        links = source_links_from_objects(defaults.get("source_links"))
        if links:
            envelope["source_links"] = links
        write_json(output, envelope)
        return emit_success(ctx, "evidence.init", {"path": str(output)})

    if args.evidence_command == "show":
        if args.client_upload_id and args.run_id:
            raise RoboRankError("Use either a run ID or --client-upload-id, not both.", code="usage_error", exit_code=2)
        if args.client_upload_id:
            payload = client(ctx).request_json("GET", "/api/evidence-runs", params={"client_upload_id": args.client_upload_id})
        elif args.run_id:
            payload = client(ctx).request_json("GET", f"/api/evidence-runs/{args.run_id}")
        else:
            raise RoboRankError("evidence show requires a run ID or --client-upload-id.", code="usage_error", exit_code=2)
        return emit_success(ctx, "evidence.show", payload)

    if args.evidence_command == "upload" and not args.client_upload_id:
        args.client_upload_id = f"upload_{uuid.uuid4().hex}"
    preflight = preflight_evidence(ctx, args, require_confirmation=args.evidence_command == "upload")
    if args.evidence_command == "validate" or ctx.options.dry_run:
        return emit_success(ctx, f"evidence.{args.evidence_command}", {"valid": True, **preflight})

    metadata = preflight["metadata"]
    files: dict[str, tuple[Path, str]] = {
        "recording": (Path(preflight["rrd_path"]), "application/octet-stream"),
    }
    if preflight["metrics_path"]:
        files["metrics"] = (Path(preflight["metrics_path"]), "application/json")
    payload = client(ctx).multipart(
        "/api/evidence-runs",
        fields={"metadata": json.dumps(metadata)},
        files=files,
    )
    result = {
        "run_id": payload.get("runId") or payload.get("run_id"),
        "run_url": payload.get("runUrl") or payload.get("run_url"),
        "client_upload_id": metadata.get("client_upload_id"),
        "recording": preflight["recording"],
        "metrics": preflight["metrics"],
        "tags": preflight["tags"],
        "license": preflight["license"],
        "server": payload,
    }
    return emit_success(ctx, "evidence.upload", result, warnings=payload.get("warnings") if isinstance(payload.get("warnings"), list) else None)


def command_eval(ctx: Context, args: argparse.Namespace) -> int:
    if args.eval_command == "list":
        return emit_success(ctx, "eval.list", list_challenges())
    if args.eval_command == "show":
        challenge = get_challenge(args.challenge_id)
        if challenge is None:
            raise RoboRankError("Unknown challenge id.", code="challenge_not_found", exit_code=2, details={"challenge_id": args.challenge_id})
        return emit_success(ctx, "eval.show", challenge)
    if args.eval_command == "run":
        policy_path = Path(args.policy_source)
        try:
            policy_source = policy_path.read_text()
        except FileNotFoundError as exc:
            raise RoboRankError(f"{policy_path} does not exist.", code="policy_source_missing", exit_code=2) from exc
        result = write_local_eval_bundle(ctx, args, policy_path, policy_source)
        if args.require_result and args.require_result != result["result_status"]:
            raise RoboRankError(
                "Run result did not meet the requested result-status gate.",
                code="result_gate_failed",
                exit_code=10,
                details={"required": args.require_result, "actual": result["result_status"], "bundle_dir": result["bundle_dir"]},
            )
        return emit_success(ctx, "eval.run", result)
    api = client(ctx)
    if ctx.options.non_interactive and not ctx.options.yes:
        raise RoboRankError("eval submit requires --yes in non-interactive mode.", code="confirmation_required", exit_code=2)
    policy_path = Path(args.policy_source)
    try:
        policy_source = policy_path.read_text()
    except FileNotFoundError as exc:
        raise RoboRankError(f"{policy_path} does not exist.", code="policy_source_missing", exit_code=2) from exc
    payload = api.request_json(
        "POST",
        "/api/runs",
        body={
            "challengeId": args.challenge_id,
            "challenge_id": args.challenge_id,
            "language": "python",
            "code": policy_source,
            "evidenceLicenseAccepted": True,
        },
    )
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    result_status = payload.get("status") or metrics.get("status")
    if args.require_result and args.require_result != result_status:
        raise RoboRankError(
            "Run result did not meet the requested result-status gate.",
            code="result_gate_failed",
            exit_code=10,
            details={"required": args.require_result, "actual": result_status},
        )
    return emit_success(
        ctx,
        "eval.submit",
        {
            "challenge_id": args.challenge_id,
            "evidence_run_id": payload.get("evidence_run_id") or payload.get("evidenceRunId"),
            "evidence_url": payload.get("evidence_url") or payload.get("evidenceUrl"),
            "score": payload.get("score"),
            "metrics": payload.get("metrics", {}),
            "response": payload,
        },
    )


def command_migration(ctx: Context, args: argparse.Namespace) -> int:
    if args.migration_command == "inventory":
        params = {"limit": args.limit, "offset": args.offset, "scope": "all"}
        if args.challenge:
            params["challenge_id"] = args.challenge
        payload = client(ctx).request_json("GET", "/api/submissions", params=params)
        submissions = payload.get("submissions", []) if isinstance(payload.get("submissions"), list) else []
        sanitized = []
        for item in submissions:
            if not isinstance(item, dict):
                continue
            sanitized.append(
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"code", "logs"}
                }
            )
        report = {
            "schema_version": "roborank.migration_inventory.v0",
            "source": "api/submissions",
            "limit": payload.get("limit", args.limit),
            "offset": payload.get("offset", args.offset),
            "has_more": bool(payload.get("hasMore")),
            "submissions": sanitized,
            "counts": {
                "total_returned": len(sanitized),
                "with_evidence": sum(1 for item in sanitized if item.get("evidenceRunId")),
                "legacy_unmapped": sum(1 for item in sanitized if not item.get("evidenceRunId")),
            },
        }
        if args.out:
            write_json(Path(args.out), report)
        return emit_success(ctx, "migration.inventory", report)
    if args.migration_command == "map":
        report = command_migration_map_report(ctx, args)
        if args.out:
            write_json(Path(args.out), report)
        return emit_success(ctx, "migration.map", report)
    if args.migration_command == "rerun":
        return command_migration_rerun(ctx, args)
    if args.migration_command == "recover":
        if not args.submission_id:
            raise RoboRankError("migration recover requires --submission-id.", code="usage_error", exit_code=2)
        if not args.artifact:
            raise RoboRankError("migration recover requires --artifact <path-or-url>.", code="usage_error", exit_code=2)
        submission_payload = client(ctx).request_json("GET", f"/api/submissions/{args.submission_id}")
        submission = submission_payload.get("submission")
        if not isinstance(submission, dict):
            raise RoboRankError(f"Submission {args.submission_id} was not returned.", code="submission_missing", exit_code=4)
        result = recover_legacy_submission_artifact(ctx, client(ctx), submission, args.artifact, Path(args.out or "migration"))
        return emit_success(
            ctx,
            "migration.recover",
            {
                "schema_version": "roborank.migration_recover.v0",
                "result": result,
            },
        )
    if args.migration_command == "upload":
        if not args.from_path:
            raise RoboRankError("migration upload requires --from <bundle-dir>.", code="usage_error", exit_code=2)
        bundle_dir = Path(args.from_path)
        evidence_path = bundle_dir / "evidence.json" if bundle_dir.is_dir() else bundle_dir
        upload_args = argparse.Namespace(
            evidence_command="upload",
            from_path=str(evidence_path),
            rrd=None,
            metrics=None,
            robot=None,
            environment=None,
            policy=None,
            policy_family=None,
            title=None,
            summary=None,
            notes=None,
            superseded_by_run_id=None,
            run_mode=None,
            result_status=None,
            license=None,
            visibility=None,
            source_link=[],
            client_upload_id=None,
            allow_new_policy=True,
            allow_new_policy_family=True,
            allow_new_robot=False,
            allow_new_environment=False,
            policy_source=None,
        )
        return command_evidence(ctx, upload_args)
    if args.migration_command == "verify":
        return command_migration_verify(ctx, args)
    if args.migration_command == "report":
        report = command_migration_dry_run_report(ctx, args)
        if args.out:
            write_json(Path(args.out), report)
        privacy_failures = [check for check in report["privacy_checks"] if check.get("status") == "fail"]
        validation_failures = report["validation_failures"]
        if privacy_failures or (args.strict and validation_failures):
            raise RoboRankError(
                "Migration dry-run report contains launch-blocking failures.",
                code="migration_report_failed",
                exit_code=9,
                details=report,
            )
        return emit_success(ctx, "migration.report", report)
    if args.migration_command == "progress":
        report = command_migration_progress_report(ctx, args)
        if args.out:
            write_json(Path(args.out), report)
        if report["decision"] == "pause":
            raise RoboRankError("Migration progress requires pausing controlled batches.", code="migration_progress_pause", exit_code=9, details=report)
        return emit_success(ctx, "migration.progress", report)
    raise RoboRankError("Unknown migration command.", code="usage_error", exit_code=2)


def command_migration_rerun(ctx: Context, args: argparse.Namespace) -> int:
    api = client(ctx)
    out_root = Path(args.out or "migration")
    if args.resume and not args.all:
        raise RoboRankError("migration rerun --resume is only supported with --all.", code="usage_error", exit_code=2)
    submission_ids: list[str] = []
    if args.submission_id:
        submission_ids.append(args.submission_id)
    elif args.all:
        return command_migration_rerun_all(ctx, args, api, out_root)
    else:
        raise RoboRankError("migration rerun requires --submission-id or --all.", code="usage_error", exit_code=2)

    results = []
    for submission_id in submission_ids:
        submission_payload = api.request_json("GET", f"/api/submissions/{submission_id}")
        submission = submission_payload.get("submission")
        if not isinstance(submission, dict):
            raise RoboRankError(f"Submission {submission_id} was not returned.", code="submission_missing", exit_code=4)
        results.append(rerun_legacy_submission(ctx, api, submission, out_root))
    return emit_success(
        ctx,
        "migration.rerun",
        {
            "schema_version": "roborank.migration_rerun.v0",
            "count": len(results),
            "results": results,
        },
    )


def command_migration_rerun_all(ctx: Context, args: argparse.Namespace, api: ApiClient, out_root: Path) -> int:
    state_path = Path(args.state) if args.state else out_root / "migration-state.json"
    state = load_migration_rerun_state(state_path, args, out_root) if args.resume and state_path.exists() else new_migration_rerun_state(args, out_root)
    processed = ensure_state_mapping(state, "processed")
    failed = ensure_state_mapping(state, "failed")
    offset = int(state.get("next_offset", args.offset) or 0)
    limit = int(args.limit)

    while True:
        params = {"limit": limit, "offset": offset, "scope": "all"}
        if args.challenge:
            params["challenge_id"] = args.challenge
        payload = api.request_json("GET", "/api/submissions", params=params)
        submissions = payload.get("submissions", []) if isinstance(payload.get("submissions"), list) else []
        state["next_offset"] = offset
        state["last_page"] = {
            "offset": offset,
            "limit": payload.get("limit", limit),
            "returned": len(submissions),
            "has_more": bool(payload.get("hasMore")),
        }
        save_migration_rerun_state(state_path, state)

        if not submissions:
            state["completed"] = True
            state["next_offset"] = offset
            save_migration_rerun_state(state_path, state)
            break

        for item in submissions:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            submission_id = str(item["id"])
            if submission_id in processed or submission_id in failed:
                continue
            state["current_submission_id"] = submission_id
            save_migration_rerun_state(state_path, state)
            try:
                submission_payload = api.request_json("GET", f"/api/submissions/{submission_id}")
                submission = submission_payload.get("submission")
                if not isinstance(submission, dict):
                    raise RoboRankError(f"Submission {submission_id} was not returned.", code="submission_missing", exit_code=4)
                processed[submission_id] = rerun_legacy_submission(ctx, api, submission, out_root)
            except RoboRankError as error:
                failed[submission_id] = migration_failure_payload(error)
            finally:
                state.pop("current_submission_id", None)
                save_migration_rerun_state(state_path, state)

        offset += len(submissions)
        state["next_offset"] = offset
        state["completed"] = not bool(payload.get("hasMore"))
        save_migration_rerun_state(state_path, state)
        if not payload.get("hasMore"):
            break

    results = list(processed.values())
    failures = failed
    return emit_success(
        ctx,
        "migration.rerun",
        {
            "schema_version": "roborank.migration_rerun.v0",
            "state_path": str(state_path),
            "status": "partial" if failures else "complete",
            "count": len(results),
            "failed_count": len(failures),
            "next_offset": state.get("next_offset"),
            "results": results,
            "failures": failures,
        },
    )


def command_migration_map_report(ctx: Context, args: argparse.Namespace) -> dict[str, Any]:
    source = "api/submissions"
    source_payload: dict[str, Any] = {}
    if args.from_path:
        source = str(args.from_path)
        source_payload = load_json(Path(args.from_path))
        raw_submissions = source_payload.get("submissions", [])
        if not isinstance(raw_submissions, list):
            raise RoboRankError(
                "Migration inventory file must contain a submissions array.",
                code="invalid_migration_inventory",
                exit_code=2,
                details={"path": source},
            )
    else:
        params = {"limit": args.limit, "offset": args.offset, "scope": "all"}
        if args.challenge:
            params["challenge_id"] = args.challenge
        source_payload = client(ctx).request_json("GET", "/api/submissions", params=params)
        raw_submissions = source_payload.get("submissions", [])
        if not isinstance(raw_submissions, list):
            raw_submissions = []

    mappings = [migration_mapping_from_submission(item) for item in raw_submissions if isinstance(item, dict)]
    mapped = sum(1 for item in mappings if item["mapping_status"] == "mapped")
    unmapped = sum(1 for item in mappings if item["mapping_status"] == "legacy_unmapped")
    return {
        "schema_version": "roborank.migration_mapping.v0",
        "source": source,
        "source_schema_version": source_payload.get("schema_version"),
        "limit": source_payload.get("limit", args.limit),
        "offset": source_payload.get("offset", args.offset),
        "has_more": bool(source_payload.get("hasMore") or source_payload.get("has_more")),
        "challenge": args.challenge,
        "mappings": mappings,
        "counts": {
            "total": len(mappings),
            "mapped": mapped,
            "legacy_unmapped": unmapped,
        },
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_migration_rerun_state(args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    return {
        "schema_version": MIGRATION_RERUN_STATE_VERSION,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "request": {
            "challenge": args.challenge,
            "limit": args.limit,
            "initial_offset": args.offset,
            "out": str(out_root),
        },
        "next_offset": args.offset,
        "completed": False,
        "processed": {},
        "failed": {},
    }


def load_migration_rerun_state(path: Path, args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    if not path.exists():
        return new_migration_rerun_state(args, out_root)
    state = load_json(path)
    if state.get("schema_version") != MIGRATION_RERUN_STATE_VERSION:
        raise RoboRankError(
            "Migration state file has an unsupported schema_version.",
            code="invalid_migration_state",
            exit_code=2,
            details={"path": str(path), "schema_version": state.get("schema_version")},
        )
    state.setdefault("processed", {})
    state.setdefault("failed", {})
    return state


def ensure_state_mapping(state: dict[str, Any], key: str) -> dict[str, Any]:
    value = state.get(key)
    if not isinstance(value, dict):
        value = {}
        state[key] = value
    return value


def migration_failure_payload(error: RoboRankError) -> dict[str, Any]:
    return {
        "code": error.code,
        "message": str(error),
        "details": error.details,
        "suggested_commands": error.suggested_commands,
    }


def save_migration_rerun_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    write_json(path, state)


def submission_challenge_id(submission: dict[str, Any]) -> str:
    return str(submission.get("challengeId") or submission.get("challenge_id") or "")


def submission_field(submission: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in submission:
            return submission.get(name)
    return None


def migration_mapping_from_submission(submission: dict[str, Any]) -> dict[str, Any]:
    submission_id = str(submission_field(submission, "id", "submissionId", "submission_id") or "")
    challenge_id = str(submission_field(submission, "challengeId", "challenge_id") or "")
    robot, environment, policy, policy_family = migration_policy_ids(challenge_id, submission_id)
    mapped = (robot, environment) != ("std/unknown", "std/unknown")
    metrics = submission.get("metrics") if isinstance(submission.get("metrics"), dict) else {}
    logs = submission.get("logs") if isinstance(submission.get("logs"), list) else []
    mapping: dict[str, Any] = {
        "submission_id": submission_id,
        "user_id": submission_field(submission, "userId", "user_id"),
        "challenge_id": challenge_id,
        "mapping_status": "mapped" if mapped else "legacy_unmapped",
        "robot": robot,
        "environment": environment,
        "policy": policy,
        "policy_family": policy_family,
        "legacy": {
            "score": submission.get("score"),
            "status": submission.get("status"),
            "created_at": submission_field(submission, "createdAt", "created_at"),
            "code_hash": submission_field(submission, "codeHash", "code_hash"),
            "metrics": metrics,
            "logs": [str(item) for item in logs],
        },
    }
    if not mapped:
        mapping["reason"] = "challenge resource mapping missing"
    return mapping


def migration_policy_ids(challenge_id: str, submission_id: str) -> tuple[str, str, str, str]:
    robot, environment = CHALLENGE_RESOURCE_CACHE.get(challenge_id, ("std/unknown", "std/unknown"))
    policy = f"migration/{canonical_slug(challenge_id)}-{canonical_slug(submission_id)[:12]}"
    policy_family = f"migration/{canonical_slug(challenge_id)}"
    return robot, environment, policy, policy_family


def recovered_artifact_bytes(api: ApiClient, artifact_ref: str) -> bytes:
    if artifact_ref.startswith(("http://", "https://", "/api/", "/artifacts/")):
        return api.request_bytes(artifact_ref)
    path = Path(artifact_ref)
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise RoboRankError(
            f"Recovered artifact {path} does not exist.",
            code="rerun_artifact_missing",
            exit_code=6,
            details={"artifact": artifact_ref},
        ) from exc


def validate_recovered_artifact_reference(artifact_ref: str, data: bytes) -> None:
    path = urllib.parse.urlparse(artifact_ref).path if artifact_ref.startswith(("http://", "https://")) else artifact_ref
    if not Path(path).name.lower().endswith(".rrd"):
        raise RoboRankError(
            "Recovered artifact must be a .rrd file.",
            code="rerun_artifact_invalid",
            exit_code=6,
            details={"artifact": artifact_ref},
        )
    if len(data) > MAX_RRD_BYTES:
        raise RoboRankError(
            "Recovered artifact exceeds the 40 MB v0 upload limit.",
            code="rerun_artifact_too_large",
            exit_code=6,
            details={"artifact": artifact_ref, "size_bytes": len(data), "max_bytes": MAX_RRD_BYTES},
        )


def recover_legacy_submission_artifact(
    ctx: Context,
    api: ApiClient,
    submission: dict[str, Any],
    artifact_ref: str,
    out_root: Path,
) -> dict[str, Any]:
    submission_id = str(submission.get("id"))
    challenge_id = submission_challenge_id(submission)
    if not submission_id or not challenge_id:
        raise RoboRankError(
            "Legacy submission is missing submission ID or challenge ID.",
            code="legacy_submission_incomplete",
            exit_code=8,
            details={"submission_id": submission_id},
        )
    artifact = recovered_artifact_bytes(api, artifact_ref)
    validate_recovered_artifact_reference(artifact_ref, artifact)

    bundle_dir = out_root / submission_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "recording.rrd").write_bytes(artifact)

    metrics = submission.get("metrics") if isinstance(submission.get("metrics"), dict) else {}
    logs = submission.get("logs") if isinstance(submission.get("logs"), list) else []
    write_json(bundle_dir / "metrics.json", metrics)
    (bundle_dir / "run.log").write_text("\n".join(str(item) for item in logs) + ("\n" if logs else ""))

    robot, environment, policy, policy_family = migration_policy_ids(challenge_id, submission_id)
    if (robot, environment) == ("std/unknown", "std/unknown"):
        ctx.warn("challenge_resource_mapping_missing", "Challenge resource mapping is missing; recovered evidence bundle uses std/unknown.")
    evidence = {
        "schema_version": "roborank.rerun_evidence.v0",
        "title": f"Recovered {challenge_id} submission artifact",
        "summary": "Evidence imported from a matched legacy Rerun artifact.",
        "robot": robot,
        "environment": environment,
        "policy": policy,
        "policy_family": policy_family,
        "recording_path": "recording.rrd",
        "metrics_path": "metrics.json",
        "run_mode": "simulation",
        "result_status": normalize_cli_result_status(submission),
        "license": DEFAULT_LICENSE,
        "visibility": "public",
        "source_kind": "migration",
        "legacy_submission_id": submission_id,
        "migration_provenance": "original_artifact_imported",
        "migration_status": "uploaded",
        "legacy_score": submission.get("score"),
        "legacy_metrics_json": metrics,
        "regenerated_score": submission.get("score"),
        "regenerated_metrics_json": metrics,
        "legacy_code_hash": submission.get("codeHash") or submission.get("code_hash"),
    }
    write_json(bundle_dir / "evidence.json", evidence)
    result = {
        "submission_id": submission_id,
        "challenge_id": challenge_id,
        "bundle_dir": str(bundle_dir),
        "provenance": "original_artifact_imported",
        "recording_path": str(bundle_dir / "recording.rrd"),
        "recording_sha256": sha256_file(bundle_dir / "recording.rrd"),
        "recording_size_bytes": len(artifact),
        "metrics_path": str(bundle_dir / "metrics.json"),
        "run_log_path": str(bundle_dir / "run.log"),
        "evidence_envelope_path": str(bundle_dir / "evidence.json"),
        "legacy_score": submission.get("score"),
        "legacy_status": submission.get("status"),
        "legacy_code_hash": submission.get("codeHash") or submission.get("code_hash"),
        "robot": robot,
        "environment": environment,
        "policy": policy,
        "policy_family": policy_family,
    }
    write_json(bundle_dir / "result.json", result)
    return result


def rerun_legacy_submission(ctx: Context, api: ApiClient, submission: dict[str, Any], out_root: Path) -> dict[str, Any]:
    del api
    submission_id = str(submission.get("id"))
    challenge_id = submission_challenge_id(submission)
    code = submission.get("code")
    if not challenge_id or not isinstance(code, str):
        raise RoboRankError(
            "Legacy submission is missing challenge ID or preserved code.",
            code="legacy_submission_incomplete",
            exit_code=8,
            details={"submission_id": submission_id},
        )
    bundle_dir = out_root / submission_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    policy_path = bundle_dir / "policy.py"
    policy_path.write_text(code)
    run_payload = run_local_challenge(challenge_id, policy_path, None, bundle_dir)
    write_json(bundle_dir / "result.json", run_payload)
    legacy_metrics = submission.get("metrics") if isinstance(submission.get("metrics"), dict) else {}
    metrics = run_payload.get("metrics") if isinstance(run_payload.get("metrics"), dict) else {}
    regenerated_score = run_payload.get("score")
    if regenerated_score is None and isinstance(metrics.get("score"), (int, float)):
        regenerated_score = metrics.get("score")
    write_json(bundle_dir / "metrics.json", metrics)
    logs = run_payload.get("logs") if isinstance(run_payload.get("logs"), list) else []
    (bundle_dir / "run.log").write_text("\n".join(str(item) for item in logs) + ("\n" if logs else ""))

    recording_status = "artifact_missing"
    try:
        copy_local_rerun_artifact(run_payload, bundle_dir)
        recording_status = "regenerated_from_code"
    except RoboRankError as error:
        ctx.warn("artifact_copy_failed", f"Could not copy regenerated artifact for {submission_id}: {error}")

    robot, environment, policy, policy_family = migration_policy_ids(challenge_id, submission_id)
    evidence = {
        "schema_version": "roborank.rerun_evidence.v0",
        "title": f"Migrated {challenge_id} submission",
        "summary": "Evidence regenerated from preserved legacy submission code.",
        "robot": robot,
        "environment": environment,
        "policy": policy,
        "policy_family": policy_family,
        "recording_path": "recording.rrd",
        "metrics_path": "metrics.json",
        "run_mode": "simulation",
        "result_status": normalize_cli_result_status(run_payload),
        "license": DEFAULT_LICENSE,
        "visibility": "public",
        "source_kind": "migration",
        "legacy_submission_id": submission_id,
        "migration_provenance": recording_status,
        "migration_status": "uploaded" if recording_status == "regenerated_from_code" else recording_status,
        "legacy_score": submission.get("score"),
        "legacy_metrics_json": legacy_metrics,
        "regenerated_score": regenerated_score,
        "regenerated_metrics_json": metrics,
        "legacy_code_hash": submission.get("codeHash") or submission.get("code_hash"),
    }
    write_json(bundle_dir / "evidence.json", evidence)
    return {
        "submission_id": submission_id,
        "challenge_id": challenge_id,
        "bundle_dir": str(bundle_dir),
        "provenance": recording_status,
        "recording_path": str(bundle_dir / "recording.rrd") if (bundle_dir / "recording.rrd").exists() else None,
        "evidence_envelope_path": str(bundle_dir / "evidence.json"),
        "legacy_score": submission.get("score"),
        "regenerated_score": regenerated_score,
    }


def command_migration_verify(ctx: Context, args: argparse.Namespace) -> int:
    if not args.from_path:
        raise RoboRankError("migration verify requires --from <bundle-dir>.", code="usage_error", exit_code=2)
    bundle = Path(args.from_path)
    evidence_path = bundle / "evidence.json"
    result_path = bundle / "result.json"
    metrics_path = bundle / "metrics.json"
    recording_path = bundle / "recording.rrd"
    missing = [str(path) for path in (evidence_path, result_path, metrics_path) if not path.exists()]
    evidence = load_json(evidence_path) if evidence_path.exists() else {}
    exposes_code = False
    if result_path.exists():
        result_text = result_path.read_text()
        exposes_code = "policy_source" in result_text or '"code"' in result_text
    report = {
        "schema_version": "roborank.migration_verify.v0",
        "bundle": str(bundle),
        "missing": missing,
        "has_recording": recording_path.exists(),
        "recording_size_bytes": recording_path.stat().st_size if recording_path.exists() else None,
        "evidence": {
            "legacy_submission_id": evidence.get("legacy_submission_id"),
            "provenance": evidence.get("migration_provenance"),
            "robot": evidence.get("robot"),
            "environment": evidence.get("environment"),
        },
        "privacy": {
            "result_json_exposes_code": exposes_code,
        },
        "valid": not missing and not exposes_code,
    }
    if not report["valid"]:
        raise RoboRankError(
            "Migration bundle verification failed.",
            code="migration_verify_failed",
            exit_code=8,
            details=report,
        )
    return emit_success(ctx, "migration.verify", report)


def report_payload(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = load_json(Path(path))
    result = payload.get("result")
    if isinstance(result, dict) and payload.get("ok") is True:
        return result
    return payload


def report_payloads(paths: Iterable[str]) -> list[dict[str, Any]]:
    return [report_payload(path) for path in paths]


def looks_like_private_source_text(value: str) -> bool:
    lowered = value.lower()
    return "class robotpolicy" in lowered or ("def act(" in lowered and "set_wheel_velocity" in lowered)


def contains_private_source_code(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in {"source_code", "sourcecode", "policy_source", "policysource"}:
                return True
            if normalized == "code" and isinstance(child, str) and looks_like_private_source_text(child):
                return True
            if contains_private_source_code(child):
                return True
    elif isinstance(value, list):
        return any(contains_private_source_code(item) for item in value)
    elif isinstance(value, str):
        return looks_like_private_source_text(value)
    return False


def migration_state_payload(path: str | None) -> dict[str, Any]:
    payload = report_payload(path)
    if not payload:
        return {"processed": {}, "failed": {}, "completed": False}
    if payload.get("schema_version") == MIGRATION_RERUN_STATE_VERSION:
        return payload
    processed = {str(item.get("submission_id")): item for item in payload.get("results", []) if isinstance(item, dict) and item.get("submission_id")}
    failures = payload.get("failures") if isinstance(payload.get("failures"), dict) else {}
    return {
        "schema_version": payload.get("schema_version"),
        "processed": processed,
        "failed": failures,
        "completed": payload.get("status") == "complete",
        "next_offset": payload.get("next_offset"),
    }


def migration_processed_items(state: dict[str, Any]) -> list[dict[str, Any]]:
    processed = state.get("processed")
    if isinstance(processed, dict):
        return [item for item in processed.values() if isinstance(item, dict)]
    return []


def migration_failed_items(state: dict[str, Any]) -> dict[str, Any]:
    failed = state.get("failed")
    return failed if isinstance(failed, dict) else {}


def migration_failure_key(item: Any) -> str:
    if isinstance(item, dict):
        value = item.get("code") or item.get("message") or item.get("error") or item.get("stage")
        if value:
            return normalized_gate_name(value)
    return normalized_gate_name(str(item) or "unknown")


def upload_report_run_id(report: dict[str, Any]) -> str | None:
    for key in ("run_id", "runId", "evidence_run_id", "evidenceRunId"):
        value = report.get(key)
        if isinstance(value, str) and value:
            return value
    server = report.get("server") if isinstance(report.get("server"), dict) else {}
    for key in ("runId", "run_id", "evidenceRunId", "evidence_run_id"):
        value = server.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def report_failure_payload(report: dict[str, Any], default_stage: str) -> dict[str, Any] | None:
    if report.get("valid") is False:
        return {
            "stage": default_stage,
            "submission_id": (report.get("evidence") or {}).get("legacy_submission_id") if isinstance(report.get("evidence"), dict) else None,
            "message": "report marked invalid",
        }
    error = report.get("error")
    if isinstance(error, dict):
        return {
            "stage": default_stage,
            "code": error.get("code"),
            "message": error.get("message"),
            "details": error.get("details"),
        }
    if report.get("ok") is False:
        return {"stage": default_stage, "message": "report marked not ok"}
    return None


def count_processed_provenance(items: Iterable[dict[str, Any]], provenance: str) -> int:
    return sum(1 for item in items if item.get("provenance") == provenance)


def migration_verification_summary(reports: list[dict[str, Any]]) -> tuple[int, int, list[dict[str, Any]]]:
    valid = 0
    invalid = 0
    failures: list[dict[str, Any]] = []
    for report in reports:
        if report.get("valid") is True:
            valid += 1
        else:
            invalid += 1
            evidence = report.get("evidence") if isinstance(report.get("evidence"), dict) else {}
            failures.append(
                {
                    "submission_id": evidence.get("legacy_submission_id"),
                    "stage": "bundle_verify",
                    "error": "migration bundle verification failed",
                    "action": "inspect bundle and rerun migration verify",
                }
            )
    return valid, invalid, failures


def migration_challenge_inventory(mappings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_challenge: dict[str, dict[str, Any]] = {}
    for mapping in mappings:
        challenge_id = str(mapping.get("challenge_id") or "unknown")
        entry = by_challenge.setdefault(
            challenge_id,
            {
                "challenge_id": challenge_id,
                "status": "mapped",
                "robot": mapping.get("robot"),
                "environment": mapping.get("environment"),
                "count": 0,
                "notes": "",
            },
        )
        entry["count"] += 1
        if mapping.get("mapping_status") != "mapped":
            entry["status"] = "legacy_unmapped"
            entry["notes"] = str(mapping.get("reason") or "challenge resource mapping missing")
    return sorted(by_challenge.values(), key=lambda item: item["challenge_id"])


def migration_mapping_sample(mappings: list[dict[str, Any]], processed_by_submission: dict[str, dict[str, Any]], limit: int = 25) -> list[dict[str, Any]]:
    sample = []
    for mapping in mappings[:limit]:
        submission_id = str(mapping.get("submission_id") or "")
        processed = processed_by_submission.get(submission_id, {})
        sample.append(
            {
                "submission_id": submission_id,
                "user_id": mapping.get("user_id"),
                "challenge_id": mapping.get("challenge_id"),
                "policy": mapping.get("policy"),
                "policy_family": mapping.get("policy_family"),
                "evidence_run_id": processed.get("evidence_run_id") or processed.get("run_id"),
                "provenance": processed.get("provenance"),
                "migration_status": mapping.get("mapping_status"),
            }
        )
    return sample


def command_migration_dry_run_report(ctx: Context, args: argparse.Namespace) -> dict[str, Any]:
    del ctx
    inventory = report_payload(args.inventory)
    mapping = report_payload(args.mapping)
    state = migration_state_payload(args.rerun_state)
    verify_reports = report_payloads(args.verify_report)
    upload_reports = report_payloads(args.upload_report)
    restore_report = report_payload(args.restore_report) if args.restore_report else {}
    smoke_report = report_payload(args.smoke_report) if args.smoke_report else {}

    submissions = inventory.get("submissions") if isinstance(inventory.get("submissions"), list) else []
    mappings = mapping.get("mappings") if isinstance(mapping.get("mappings"), list) else []
    processed_items = migration_processed_items(state)
    processed_by_submission = {str(item.get("submission_id")): item for item in processed_items if item.get("submission_id")}
    failures = migration_failed_items(state)
    valid_verify_count, invalid_verify_count, verification_failures = migration_verification_summary(verify_reports)
    source_code_exposed = contains_private_source_code(
        {
            "inventory": inventory,
            "mapping": mapping,
            "rerun_state": state,
            "verify_reports": verify_reports,
            "upload_reports": upload_reports,
        }
    )
    if upload_reports:
        license_confirmed = all(
            report.get("license") in ALLOWED_LICENSES
            or (
                isinstance(report.get("server"), dict)
                and (report["server"].get("licenseArtifact") in ALLOWED_LICENSES or report["server"].get("license_artifact") in ALLOWED_LICENSES)
            )
            for report in upload_reports
        )
    else:
        license_confirmed = False
    smoke_checks = smoke_report.get("checks") if isinstance(smoke_report.get("checks"), list) else []
    legacy_readable = bool(smoke_checks) and not any(
        isinstance(check, dict) and check.get("name") == "legacy_leaderboard" and check.get("status") == "fail"
        for check in smoke_checks
    )
    privacy_checks = [
        {
            "name": "no_preserved_source_code_in_reports",
            "status": "fail" if source_code_exposed else "pass",
            "evidence": "report inputs scanned for code, policy_source, and common RobotPolicy snippets",
        },
        {
            "name": "public_evidence_uses_hashes_or_links",
            "status": "pass" if not source_code_exposed else "fail",
            "evidence": "migration reports include legacy code hashes and generated resource IDs only",
        },
        {
            "name": "license_confirmation_recorded",
            "status": "pass" if license_confirmed else "warning",
            "evidence": "upload reports include allowed artifact license" if upload_reports else "no upload reports supplied",
        },
        {
            "name": "legacy_submissions_remain_readable",
            "status": "pass" if legacy_readable else "warning",
            "evidence": "launch smoke legacy leaderboard check passed" if smoke_checks else "no launch smoke report supplied",
        },
    ]

    mapping_counts = mapping.get("counts") if isinstance(mapping.get("counts"), dict) else {}
    legacy_unmapped = int(mapping_counts.get("legacy_unmapped") or 0)
    recovered_count = count_processed_provenance(processed_items, "original_artifact_imported")
    regenerated_count = count_processed_provenance(processed_items, "regenerated_from_code")
    artifact_missing_count = count_processed_provenance(processed_items, "artifact_missing")
    validation_failures = [
        *verification_failures,
        *[
            {
                "submission_id": submission_id,
                "stage": "rerun",
                "error": item.get("message") if isinstance(item, dict) else str(item),
                "action": "inspect preserved submission and retry or mark regeneration_failed",
            }
            for submission_id, item in failures.items()
        ],
    ]
    return {
        "schema_version": "roborank.migration_dry_run_report.v0",
        "environment": args.environment,
        "metadata": {
            "d1_export_path": args.d1_export,
            "artifact_inventory_path": args.artifact_inventory,
            "worker_version": args.worker_version,
            "frontend_version": args.frontend_version,
            "backend_version": args.backend_version,
            "roborank_envs_version": args.roborank_envs_version,
            "started_at": args.started_at,
            "completed_at": args.completed_at or now_iso(),
            "operator": args.operator,
        },
        "summary": {
            "legacy_submissions_inspected": len(submissions) or int(mapping_counts.get("total") or 0),
            "submissions_with_recovered_rrd_artifacts": recovered_count,
            "submissions_regenerated_from_preserved_code": regenerated_count,
            "schema_valid_evidence_runs": valid_verify_count,
            "evidence_runs_excluded_from_schema_leaderboards": artifact_missing_count + legacy_unmapped,
            "legacy_unmapped_submissions": legacy_unmapped,
            "artifact_missing_submissions": artifact_missing_count,
            "regeneration_failed_submissions": len(failures),
            "invalid_verified_bundles": invalid_verify_count,
        },
        "challenge_inventory": migration_challenge_inventory(mappings),
        "submission_mapping_sample": migration_mapping_sample(mappings, processed_by_submission),
        "artifact_recovery": [
            {
                "source": args.artifact_inventory or "migration rerun state",
                "files_inspected": len(processed_items) + len(failures),
                "matched": recovered_count + regenerated_count,
                "rejected": artifact_missing_count + len(failures),
                "notes": "Counts derived from migration rerun/recover state.",
            }
        ],
        "validation_failures": validation_failures,
        "privacy_checks": privacy_checks,
        "rollback_plan": {
            "d1_restore_point": args.rollback_d1,
            "object_storage_restore_point": args.rollback_objects,
            "worker_rollback_version": args.rollback_worker,
            "frontend_rollback_version": args.rollback_frontend,
            "migration_pause_resume_state_path": args.rerun_state,
        },
        "source_reports": {
            "inventory": args.inventory,
            "mapping": args.mapping,
            "rerun_state": args.rerun_state,
            "verify_reports": args.verify_report,
            "upload_reports": args.upload_report,
            "restore_report": args.restore_report,
            "smoke_report": args.smoke_report,
        },
        "restore_report_summary": restore_report.get("counts") if isinstance(restore_report.get("counts"), dict) else None,
        "smoke_report_summary": smoke_report.get("counts") if isinstance(smoke_report.get("counts"), dict) else None,
    }


def command_migration_progress_report(ctx: Context, args: argparse.Namespace) -> dict[str, Any]:
    del ctx
    mapping = report_payload(args.mapping)
    state = migration_state_payload(args.rerun_state)
    verify_reports = report_payloads(args.verify_report)
    upload_reports = report_payloads(args.upload_report)
    mappings = mapping.get("mappings") if isinstance(mapping.get("mappings"), list) else []
    mapping_counts = mapping.get("counts") if isinstance(mapping.get("counts"), dict) else {}
    mapped = int(mapping_counts.get("mapped") or sum(1 for item in mappings if isinstance(item, dict) and item.get("mapping_status") == "mapped"))
    legacy_unmapped = int(
        mapping_counts.get("legacy_unmapped")
        or sum(1 for item in mappings if isinstance(item, dict) and item.get("mapping_status") == "legacy_unmapped")
    )
    total = int(mapping_counts.get("total") or len(mappings) or mapped + legacy_unmapped)
    processed_items = migration_processed_items(state)
    rerun_failures = migration_failed_items(state)
    upload_successes = [report for report in upload_reports if upload_report_run_id(report)]
    upload_failures = [failure for failure in (report_failure_payload(report, "upload") for report in upload_reports) if failure]
    verify_successes = [report for report in verify_reports if report.get("valid") is True]
    verify_failures = [failure for failure in (report_failure_payload(report, "verify") for report in verify_reports) if failure]
    processed_submission_ids = {str(item.get("submission_id")) for item in processed_items if item.get("submission_id")}
    uploaded_run_ids = {str(upload_report_run_id(report)) for report in upload_successes if upload_report_run_id(report)}
    verified_submission_ids = {
        str(report.get("evidence", {}).get("legacy_submission_id"))
        for report in verify_successes
        if isinstance(report.get("evidence"), dict) and report["evidence"].get("legacy_submission_id")
    }
    all_failures = [
        *[
            {
                "stage": "rerun",
                "submission_id": submission_id,
                "code": item.get("code") if isinstance(item, dict) else None,
                "message": item.get("message") if isinstance(item, dict) else str(item),
            }
            for submission_id, item in rerun_failures.items()
        ],
        *upload_failures,
        *verify_failures,
    ]
    failure_codes: dict[str, int] = {}
    for failure in all_failures:
        key = migration_failure_key(failure)
        failure_codes[key] = failure_codes.get(key, 0) + 1
    repeated_failures = sorted(key for key, count in failure_codes.items() if count >= args.repeated_failure_threshold)
    completed_count = len(verified_submission_ids) if verify_reports else len(uploaded_run_ids) if upload_reports else len(processed_submission_ids)
    remaining = max(mapped - completed_count - len(rerun_failures), 0)
    failure_count = len(all_failures)
    attempted = completed_count + failure_count
    failure_rate = (failure_count / attempted) if attempted else 0.0
    pause_reasons = []
    if contains_private_source_code({"mapping": mapping, "rerun_state": state, "verify_reports": verify_reports, "upload_reports": upload_reports}):
        pause_reasons.append("migration reports expose preserved source code")
    if failure_count > args.max_failures:
        pause_reasons.append(f"failure count {failure_count} exceeds max {args.max_failures}")
    if attempted and failure_rate > args.max_failure_rate:
        pause_reasons.append(f"failure rate {failure_rate:.2%} exceeds max {args.max_failure_rate:.2%}")
    if repeated_failures:
        pause_reasons.append(f"repeated failure keys reached threshold {args.repeated_failure_threshold}: {', '.join(repeated_failures)}")
    decision = "pause" if pause_reasons else "complete" if remaining == 0 and bool(state.get("completed")) else "continue"
    next_offset = state.get("next_offset")
    if next_offset is None:
        next_offset = min(completed_count + failure_count + legacy_unmapped, total)
    return {
        "schema_version": "roborank.migration_progress.v0",
        "environment": args.environment,
        "generated_at": now_iso(),
        "decision": decision,
        "pause_recommended": decision == "pause",
        "pause_reasons": pause_reasons,
        "counts": {
            "legacy_submissions_total": total,
            "mapped_submissions": mapped,
            "legacy_unmapped_submissions": legacy_unmapped,
            "rerun_processed": len(processed_items),
            "rerun_failed": len(rerun_failures),
            "evidence_uploaded": len(upload_successes),
            "upload_failed": len(upload_failures),
            "verified_valid": len(verify_successes),
            "verified_failed": len(verify_failures),
            "migration_completed": completed_count,
            "migration_remaining": remaining,
            "failure_count": failure_count,
        },
        "health": {
            "attempted": attempted,
            "failure_rate": failure_rate,
            "max_failures": args.max_failures,
            "max_failure_rate": args.max_failure_rate,
            "failure_codes": failure_codes,
            "repeated_failure_threshold": args.repeated_failure_threshold,
            "repeated_failures": repeated_failures,
        },
        "batch": {
            "batch_size": args.batch_size,
            "next_offset": next_offset,
            "next_limit": args.batch_size,
            "state_completed": bool(state.get("completed")),
        },
        "sample": {
            "processed_submission_ids": sorted(processed_submission_ids)[:25],
            "verified_submission_ids": sorted(verified_submission_ids)[:25],
            "uploaded_run_ids": sorted(uploaded_run_ids)[:25],
            "failures": all_failures[:25],
        },
        "source_reports": {
            "mapping": args.mapping,
            "rerun_state": args.rerun_state,
            "verify_reports": args.verify_report,
            "upload_reports": args.upload_report,
        },
    }


def api_path_segment(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def check_list_payload(payload: Any, key: str) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get(key), list):
        return payload[key]
    return []


def evidence_run_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    nested = payload.get("run")
    if isinstance(nested, dict):
        return nested
    return payload


def launch_smoke_counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        status = str(check.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def launch_smoke_skip(checks: list[dict[str, Any]], name: str, message: str) -> None:
    checks.append({"name": name, "status": "skipped", "message": message})


def launch_smoke_warning(ctx: Context, checks: list[dict[str, Any]], name: str, message: str, data: dict[str, Any] | None = None) -> None:
    ctx.warn("launch_smoke_weak_evidence", f"{name}: {message}")
    check: dict[str, Any] = {"name": name, "status": "warning", "message": message}
    if data:
        check["data"] = data
    checks.append(check)


def launch_smoke_check(checks: list[dict[str, Any]], name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any] | None:
    started = time.time()
    try:
        data = fn()
    except RoboRankError as exc:
        checks.append(
            {
                "name": name,
                "status": "fail",
                "code": exc.code,
                "message": str(exc),
                "details": exc.details,
                "duration_ms": round((time.time() - started) * 1000),
            }
        )
        return None
    except Exception as exc:  # noqa: BLE001 - smoke checks should continue and report every failed probe.
        checks.append(
            {
                "name": name,
                "status": "fail",
                "code": "internal_error",
                "message": str(exc),
                "duration_ms": round((time.time() - started) * 1000),
            }
        )
        return None
    checks.append({"name": name, "status": "pass", "duration_ms": round((time.time() - started) * 1000), "data": data})
    return data


def explorer_facet_filter_params(args: argparse.Namespace) -> dict[str, str]:
    if args.resource_kind and args.resource_id and args.resource_kind in {"robot", "environment", "policy", "policy_family"}:
        return {args.resource_kind: args.resource_id}
    if args.environment:
        return {"environment": args.environment}
    return {}


def evidence_artifact_url(run_payload: dict[str, Any] | None) -> str | None:
    if not run_payload:
        return None
    artifacts = run_payload.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    for artifact in artifacts:
        if isinstance(artifact, dict) and isinstance(artifact.get("url"), str):
            return artifact["url"]
    return None


def command_launch(ctx: Context, args: argparse.Namespace) -> int:
    if args.launch_command == "smoke":
        return command_launch_smoke(ctx, args)
    if args.launch_command == "evidence-examples":
        return command_launch_evidence_examples(ctx, args)
    if args.launch_command == "privacy-scan":
        return command_launch_privacy_scan(ctx, args)
    if args.launch_command == "schema-guard":
        return command_launch_schema_guard(ctx, args)
    if args.launch_command == "trust-actions":
        return command_launch_trust_actions(ctx, args)
    if args.launch_command == "resource-guard":
        return command_launch_resource_guard(ctx, args)
    if args.launch_command == "browser-qa":
        return command_launch_browser_qa(ctx, args)
    if args.launch_command == "known-issues":
        return command_launch_known_issues(ctx, args)
    if args.launch_command == "triage":
        return command_launch_triage(ctx, args)
    if args.launch_command == "signoff":
        return command_launch_signoff(ctx, args)
    if args.launch_command == "cli-release":
        return command_launch_cli_release(ctx, args)
    if args.launch_command == "cutover":
        return command_launch_cutover(ctx, args)
    if args.launch_command == "preflight":
        return command_launch_preflight(ctx, args)
    if args.launch_command == "watch":
        return command_launch_watch(ctx, args)
    if args.launch_command == "restore-check":
        return command_launch_restore_check(ctx, args)
    if args.launch_command == "backup-manifest":
        return command_launch_backup_manifest(ctx, args)
    if args.launch_command == "go-no-go":
        return command_launch_go_no_go(ctx, args)
    raise RoboRankError("Unknown launch command.", code="usage_error", exit_code=2)


def command_launch_smoke(ctx: Context, args: argparse.Namespace) -> int:
    if args.include_mutating and not ctx.options.yes:
        raise RoboRankError("launch smoke --include-mutating requires --yes.", code="confirmation_required", exit_code=2)

    started_at = datetime.now(timezone.utc).isoformat()
    checks: list[dict[str, Any]] = []
    api = client(ctx)
    anonymous_api = ApiClient(ctx.api_url)
    run_detail: dict[str, Any] | None = None

    if ctx.options.token:
        def check_auth() -> dict[str, Any]:
            payload = api.request_json("GET", "/api/me")
            user = payload.get("user") if isinstance(payload, dict) else None
            scopes = user.get("tokenScopes") if isinstance(user, dict) and isinstance(user.get("tokenScopes"), list) else []
            return {
                "authenticated": isinstance(user, dict),
                "is_admin": bool(user.get("isAdmin")) if isinstance(user, dict) else False,
                "token_scope_count": len(scopes),
            }

        launch_smoke_check(checks, "auth", check_auth)
    else:
        launch_smoke_skip(checks, "auth", "No token configured; authenticated smoke check was not run.")

    def check_challenge_catalog() -> dict[str, Any]:
        payload = api.request_json("GET", "/api/challenges")
        challenges = check_list_payload(payload, "challenges")
        if not challenges:
            raise RoboRankError("Challenge catalog returned no challenges.", code="challenge_catalog_empty", exit_code=9)
        return {"count": len(challenges)}

    launch_smoke_check(checks, "problem_list", check_challenge_catalog)

    def check_legacy_leaderboard() -> dict[str, Any]:
        payload = api.request_json("GET", "/api/leaderboard", params={"limit": 1, "offset": 0})
        entries = check_list_payload(payload, "entries")
        return {"entries": len(entries), "total": payload.get("total") if isinstance(payload, dict) else None}

    launch_smoke_check(checks, "legacy_leaderboard", check_legacy_leaderboard)

    if args.legacy_submission_id and ctx.options.token:
        def check_legacy_submission_detail() -> dict[str, Any]:
            payload = api.request_json("GET", f"/api/submissions/{api_path_segment(args.legacy_submission_id)}")
            submission = payload.get("submission") if isinstance(payload, dict) else None
            if not isinstance(submission, dict):
                raise RoboRankError(
                    "Legacy submission detail returned an invalid response.",
                    code="legacy_submission_detail_shape_invalid",
                    exit_code=9,
                )
            actual_id = submission.get("id") or submission.get("submissionId") or submission.get("submission_id")
            if actual_id and str(actual_id) != args.legacy_submission_id:
                raise RoboRankError(
                    "Legacy submission detail returned an unexpected submission ID.",
                    code="legacy_submission_detail_mismatch",
                    exit_code=9,
                    details={"expected": args.legacy_submission_id, "actual": actual_id},
                )
            return {
                "submission_id": args.legacy_submission_id,
                "status": submission.get("status"),
                "challenge_id": submission.get("challengeId") or submission.get("challenge_id"),
                "has_evidence": bool(submission.get("evidenceRunId") or submission.get("evidence_run_id")),
            }

        launch_smoke_check(checks, "legacy_submission_detail", check_legacy_submission_detail)
    elif args.legacy_submission_id:
        launch_smoke_skip(checks, "legacy_submission_detail", "No token configured; legacy submission detail requires submitter/admin auth.")
    else:
        launch_smoke_skip(checks, "legacy_submission_detail", "Pass --legacy-submission-id to verify an existing submission remains readable.")

    def check_anonymous_explorer_limit() -> dict[str, Any]:
        requested = max(args.anonymous_limit + 1, 3)
        payload = anonymous_api.request_json("GET", "/api/explorer/runs", params={"limit": requested, "offset": 0})
        runs = check_list_payload(payload, "runs")
        if len(runs) > args.anonymous_limit:
            raise RoboRankError(
                "Anonymous explorer returned more runs than the configured preview limit.",
                code="anonymous_limit_bypassed",
                exit_code=9,
                details={"requested": requested, "returned": len(runs), "limit": args.anonymous_limit},
            )
        return {"requested": requested, "returned": len(runs), "limit": args.anonymous_limit}

    anonymous_result = launch_smoke_check(checks, "explorer_anonymous_limit", check_anonymous_explorer_limit)
    if anonymous_result and anonymous_result["returned"] < args.anonymous_limit:
        launch_smoke_warning(
            ctx,
            checks,
            "explorer_anonymous_limit_dataset",
            "Dataset has fewer public runs than the anonymous limit, so the cap was not fully proven.",
            anonymous_result,
        )

    def check_explorer_facets() -> dict[str, Any]:
        params = explorer_facet_filter_params(args)
        payload = anonymous_api.request_json("GET", "/api/explorer/facets", params=params)
        facets = check_list_payload(payload, "facets")
        invalid = [
            index
            for index, facet in enumerate(facets)
            if not isinstance(facet, dict)
            or not isinstance(facet.get("kind"), str)
            or not isinstance(facet.get("canonicalId") or facet.get("canonical_id"), str)
        ]
        if invalid:
            raise RoboRankError(
                "Explorer facets returned invalid facet rows.",
                code="explorer_facets_shape_invalid",
                exit_code=9,
                details={"invalid_indexes": invalid[:10], "params": params},
            )
        kinds = sorted({str(facet.get("kind")) for facet in facets if isinstance(facet, dict) and facet.get("kind")})
        return {"params": params, "facet_count": len(facets), "kinds": kinds}

    facets_result = launch_smoke_check(checks, "explorer_facets", check_explorer_facets)
    if facets_result and facets_result["facet_count"] == 0:
        launch_smoke_warning(
            ctx,
            checks,
            "explorer_facets_dataset",
            "Explorer facets returned no facet rows, so contextual facet data was not fully proven.",
            facets_result,
        )

    if ctx.options.token:
        def check_signed_in_explorer() -> dict[str, Any]:
            payload = api.request_json("GET", "/api/explorer/runs", params={"limit": args.signed_in_limit, "offset": 0})
            runs = check_list_payload(payload, "runs")
            return {
                "requested": args.signed_in_limit,
                "returned": len(runs),
                "has_more": payload.get("hasMore") if isinstance(payload, dict) else None,
            }

        launch_smoke_check(checks, "explorer_signed_in_pagination", check_signed_in_explorer)
    else:
        launch_smoke_skip(checks, "explorer_signed_in_pagination", "No token configured; signed-in explorer pagination was not run.")

    if args.run_id:
        def check_run_detail() -> dict[str, Any]:
            payload = api.request_json("GET", f"/api/evidence-runs/{api_path_segment(args.run_id)}")
            run_payload = evidence_run_payload(payload)
            if not run_payload:
                raise RoboRankError("Evidence run detail returned an invalid response.", code="run_detail_shape_invalid", exit_code=9)
            actual_id = run_payload.get("id")
            if actual_id and actual_id != args.run_id:
                raise RoboRankError("Evidence run detail returned an unexpected run ID.", code="run_detail_mismatch", exit_code=9)
            nonlocal run_detail
            run_detail = run_payload
            artifacts = check_list_payload(run_detail, "artifacts")
            return {"run_id": actual_id or args.run_id, "artifact_count": len(artifacts)}

        launch_smoke_check(checks, "run_detail", check_run_detail)
    else:
        launch_smoke_skip(checks, "run_detail", "Pass --run-id to verify a specific evidence run page.")

    artifact_url = args.artifact_url or evidence_artifact_url(run_detail)
    if artifact_url:
        def check_artifact_download() -> dict[str, Any]:
            data = api.request_bytes(artifact_url)
            if not data:
                raise RoboRankError("Artifact download returned zero bytes.", code="artifact_empty", exit_code=9)
            return {"url": artifact_url, "size_bytes": len(data)}

        launch_smoke_check(checks, "artifact_download", check_artifact_download)
    else:
        launch_smoke_skip(checks, "artifact_download", "Pass --artifact-url or --run-id with artifacts to verify artifact bytes.")

    if args.resource_kind and args.resource_id:
        def check_resource_page() -> dict[str, Any]:
            namespace, slug = canonical_id_parts(args.resource_id)
            payload = api.request_json(
                "GET",
                f"/api/resources/{api_path_segment(args.resource_kind)}/{api_path_segment(namespace)}/{api_path_segment(slug)}",
            )
            resource = payload.get("resource") if isinstance(payload, dict) else None
            if not isinstance(resource, dict):
                raise RoboRankError("Resource page returned an invalid response.", code="resource_page_shape_invalid", exit_code=9)
            return {"kind": args.resource_kind, "resource_id": args.resource_id, "found": True}

        launch_smoke_check(checks, "resource_page", check_resource_page)
    else:
        launch_smoke_skip(checks, "resource_page", "Pass --resource-kind and --resource-id to verify a resource page.")

    if args.environment:
        def check_environment_leaderboard() -> dict[str, Any]:
            namespace, slug = canonical_id_parts(args.environment)
            payload = api.request_json(
                "GET",
                f"/api/leaderboards/{api_path_segment(namespace)}/{api_path_segment(slug)}",
            )
            if not isinstance(payload, dict) or not isinstance(payload.get("environment"), dict):
                raise RoboRankError(
                    "Environment leaderboard returned an invalid response.",
                    code="environment_leaderboard_shape_invalid",
                    exit_code=9,
                )
            entries = check_list_payload(payload, "entries")
            return {"environment": args.environment, "entries": len(entries)}

        launch_smoke_check(checks, "environment_leaderboard", check_environment_leaderboard)
    else:
        launch_smoke_skip(checks, "environment_leaderboard", "Pass --environment to verify an environment leaderboard.")

    if len(args.compare_run) >= 2:
        def check_compare() -> dict[str, Any]:
            validate_compare_run_count(args.compare_run)
            payload = api.request_json("GET", "/api/compare", params={"run": args.compare_run})
            runs = check_list_payload(payload, "runs")
            if len(runs) < 2:
                raise RoboRankError("Compare returned fewer than two runs.", code="compare_incomplete", exit_code=9)
            return {"requested": len(args.compare_run), "returned": len(runs)}

        launch_smoke_check(checks, "compare", check_compare)
    else:
        launch_smoke_skip(checks, "compare", "Pass --compare-run at least twice to verify evidence compare.")

    if args.include_mutating:
        if args.policy_source:
            launch_smoke_check(checks, "problem_submit_evidence_creation", lambda: launch_problem_submit_smoke(api, args))
        else:
            launch_smoke_skip(checks, "problem_submit_evidence_creation", "Pass --policy-source with --include-mutating to submit a challenge run.")

        if args.direct_rrd and args.robot and args.environment and args.policy:
            launch_smoke_check(checks, "direct_evidence_creation", lambda: launch_direct_evidence_smoke(ctx, args))
        else:
            launch_smoke_skip(
                checks,
                "direct_evidence_creation",
                "Pass --direct-rrd, --robot, --environment, and --policy with --include-mutating to upload direct evidence.",
            )
    else:
        launch_smoke_skip(checks, "problem_submit_evidence_creation", "Use --include-mutating --yes to submit a challenge run.")
        launch_smoke_skip(checks, "direct_evidence_creation", "Use --include-mutating --yes with direct evidence inputs to upload evidence.")

    report = {
        "schema_version": "roborank.launch_smoke.v0",
        "target": args.target,
        "api_url": ctx.api_url,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "strict": args.strict,
        "include_mutating": args.include_mutating,
        "checks": checks,
        "counts": launch_smoke_counts(checks),
    }
    if args.out:
        write_json(Path(args.out), report)
    failed = [check for check in checks if check.get("status") == "fail"]
    incomplete = [check for check in checks if check.get("status") in {"skipped", "warning"}]
    if failed or (args.strict and incomplete):
        raise RoboRankError(
            "Launch smoke checks failed.",
            code="launch_smoke_failed",
            exit_code=9,
            details=report,
        )
    return emit_success(ctx, "launch.smoke", report)


def launch_problem_submit_smoke(api: ApiClient, args: argparse.Namespace) -> dict[str, Any]:
    policy_path = Path(args.policy_source)
    try:
        policy_source = policy_path.read_text()
    except FileNotFoundError as exc:
        raise RoboRankError(f"{policy_path} does not exist.", code="policy_source_missing", exit_code=2) from exc
    payload = api.request_json(
        "POST",
        "/api/runs",
        body={
            "challengeId": args.challenge_id,
            "challenge_id": args.challenge_id,
            "language": "python",
            "code": policy_source,
            "evidenceLicenseAccepted": True,
        },
    )
    if not isinstance(payload, dict):
        raise RoboRankError("Challenge submission returned an invalid response.", code="problem_submit_shape_invalid", exit_code=9)
    evidence_run_id = payload.get("evidence_run_id") or payload.get("evidenceRunId")
    if not evidence_run_id:
        raise RoboRankError("Challenge submission did not create an evidence run.", code="evidence_side_effect_missing", exit_code=9)
    if not any(key in payload for key in ("score", "status", "metrics")):
        raise RoboRankError("Challenge submission did not return normal result fields.", code="problem_submit_shape_invalid", exit_code=9)
    return {
        "challenge_id": args.challenge_id,
        "evidence_run_id": evidence_run_id,
        "evidence_url": payload.get("evidence_url") or payload.get("evidenceUrl"),
        "score": payload.get("score"),
        "status": payload.get("status"),
    }


def launch_direct_evidence_smoke(ctx: Context, args: argparse.Namespace) -> dict[str, Any]:
    evidence_args = argparse.Namespace(
        evidence_command="upload",
        title="Launch smoke direct evidence",
        summary="Direct evidence upload created by roborank launch smoke.",
        notes=None,
        superseded_by_run_id=None,
        robot=args.robot,
        environment=args.environment,
        policy=args.policy,
        policy_family=args.policy_family,
        run_mode="smoke",
        result_status="unknown",
        license=args.license,
        visibility=args.visibility,
        from_path=None,
        rrd=args.direct_rrd,
        metrics=args.metrics,
        source_link=args.source_link,
        client_upload_id=args.client_upload_id or f"smoke_{uuid.uuid4().hex}",
        allow_new_policy=args.allow_new_policy,
        allow_new_policy_family=args.allow_new_policy_family,
        allow_new_robot=False,
        allow_new_environment=False,
        policy_source=None,
    )
    preflight = preflight_evidence(ctx, evidence_args, require_confirmation=True)
    files: dict[str, tuple[Path, str]] = {
        "recording": (Path(preflight["rrd_path"]), "application/octet-stream"),
    }
    if preflight["metrics_path"]:
        files["metrics"] = (Path(preflight["metrics_path"]), "application/json")
    payload = client(ctx).multipart(
        "/api/evidence-runs",
        fields={"metadata": json.dumps(preflight["metadata"])},
        files=files,
    )
    if not isinstance(payload, dict):
        raise RoboRankError("Direct evidence upload returned an invalid response.", code="evidence_upload_shape_invalid", exit_code=9)
    run_id = payload.get("runId") or payload.get("run_id")
    if not run_id:
        raise RoboRankError("Direct evidence upload did not return a run ID.", code="evidence_upload_shape_invalid", exit_code=9)
    return {
        "run_id": run_id,
        "run_url": payload.get("runUrl") or payload.get("run_url"),
        "client_upload_id": preflight["metadata"].get("client_upload_id"),
        "recording": preflight["recording"],
    }


def normalized_evidence_scenario(value: str) -> str:
    normalized = normalized_gate_name(value)
    scenario = CURATED_EVIDENCE_SCENARIO_ALIASES.get(normalized)
    if not scenario:
        raise RoboRankError(
            "Unsupported curated evidence scenario.",
            code="invalid_evidence_scenario",
            exit_code=2,
            details={"scenario": value, "allowed": sorted(CURATED_EVIDENCE_SCENARIOS)},
        )
    return scenario


def parse_evidence_example(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise RoboRankError("--example must use scenario=run_id.", code="usage_error", exit_code=2)
    scenario_raw, run_id = value.split("=", 1)
    scenario = normalized_evidence_scenario(scenario_raw)
    if not run_id:
        raise RoboRankError("--example run_id cannot be empty.", code="usage_error", exit_code=2)
    return scenario, run_id


def run_metric_schema_id(run: dict[str, Any]) -> str | None:
    for key in ("metricsSchemaId", "metrics_schema_id", "metricsSchemaID"):
        value = run.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def run_warnings(run: dict[str, Any]) -> list[str]:
    warnings = run.get("warnings")
    if not isinstance(warnings, list):
        return []
    result = []
    for warning in warnings:
        if isinstance(warning, str):
            result.append(warning)
        elif isinstance(warning, dict):
            result.append(str(warning.get("message") or warning.get("code") or ""))
    return result


def run_tag_canonical_id(run: dict[str, Any], kind: str) -> str | None:
    tags = run.get("tags")
    if isinstance(tags, dict):
        value = tags.get(kind)
        if isinstance(value, dict):
            for key in ("canonicalId", "canonical_id", "canonical"):
                if isinstance(value.get(key), str):
                    return value[key]
        if isinstance(value, str):
            return value
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict) and tag.get("kind") == kind:
                for key in ("canonicalId", "canonical_id", "canonical"):
                    if isinstance(tag.get(key), str):
                        return tag[key]
    return None


def run_artifacts(run: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = run.get("artifacts")
    return [artifact for artifact in artifacts if isinstance(artifact, dict)] if isinstance(artifacts, list) else []


def artifact_viewer_state(artifact: dict[str, Any]) -> str:
    return normalized_gate_name(artifact.get("viewerCompatibilityState") or artifact.get("viewer_compatibility_state") or "viewer_unknown")


def run_metric_success_false(run: dict[str, Any]) -> bool:
    metrics = run.get("metrics")
    if isinstance(metrics, list):
        for metric in metrics:
            if isinstance(metric, dict) and normalized_gate_name(metric.get("name")) == "success":
                value = metric.get("valueNumber")
                if value is None:
                    value = metric.get("value_number")
                text = normalized_gate_name(metric.get("valueText") or metric.get("value_text"))
                return value == 0 or text in {"false", "failed", "failure"}
    if isinstance(metrics, dict):
        return metrics.get("success") is False
    return False


def validate_curated_evidence_example(api: ApiClient, scenario: str, run_id: str) -> dict[str, Any]:
    payload = api.request_json("GET", f"/api/evidence-runs/{api_path_segment(run_id)}")
    run = evidence_run_payload(payload)
    if not run:
        raise RoboRankError("Evidence example returned an invalid response.", code="evidence_example_shape_invalid", exit_code=9)
    actual_id = run.get("id")
    if actual_id and actual_id != run_id:
        raise RoboRankError(
            "Evidence example returned an unexpected run ID.",
            code="evidence_example_mismatch",
            exit_code=9,
            details={"expected": run_id, "actual": actual_id},
        )
    if contains_private_source_code(run):
        raise RoboRankError("Evidence example exposes private source code.", code="evidence_example_exposes_source", exit_code=9)
    artifacts = run_artifacts(run)
    warnings = run_warnings(run)
    schema_id = run_metric_schema_id(run)
    if scenario in {"schema_backed", "missing_optional_metrics"}:
        if not schema_id:
            raise RoboRankError(
                "Curated schema-backed example is missing metrics schema evidence.",
                code="evidence_example_schema_missing",
                exit_code=9,
                details={"scenario": scenario, "run_id": run_id},
            )
        if any("metrics_schema_validation_failed" in warning or "invalid metric" in warning.lower() for warning in warnings):
            raise RoboRankError(
                "Curated schema-backed example has metrics validation warnings.",
                code="evidence_example_schema_invalid",
                exit_code=9,
                details={"scenario": scenario, "run_id": run_id, "warnings": warnings},
            )
    if scenario == "schema_backed" and not artifacts:
        raise RoboRankError("Schema-backed example must retain an artifact.", code="evidence_example_artifact_missing", exit_code=9)
    if scenario == "std_unknown" and run_tag_canonical_id(run, "environment") != "std/unknown":
        raise RoboRankError(
            "std/unknown curated example is not tagged with std/unknown.",
            code="evidence_example_std_unknown_missing",
            exit_code=9,
            details={"run_id": run_id, "environment": run_tag_canonical_id(run, "environment")},
        )
    if scenario == "failed_result":
        status = normalized_gate_name(run.get("resultStatus") or run.get("result_status"))
        if status not in {"failure", "failed", "timeout", "aborted", "error"} and not run_metric_success_false(run):
            raise RoboRankError(
                "Failed-result curated example does not show a failed result.",
                code="evidence_example_failed_result_missing",
                exit_code=9,
                details={"run_id": run_id, "result_status": status},
            )
    if scenario == "viewer_incompatible":
        if not artifacts:
            raise RoboRankError("Viewer-incompatible example must retain an artifact.", code="evidence_example_artifact_missing", exit_code=9)
        viewer_states = [artifact_viewer_state(artifact) for artifact in artifacts]
        if all(state in {"viewer_ok", "viewer_compatible"} for state in viewer_states):
            raise RoboRankError(
                "Viewer-incompatible example does not show retained incompatible/fallback artifact state.",
                code="evidence_example_viewer_state_missing",
                exit_code=9,
                details={"run_id": run_id, "viewer_states": viewer_states},
            )
    return {
        "scenario": scenario,
        "description": CURATED_EVIDENCE_SCENARIOS[scenario],
        "run_id": actual_id or run_id,
        "metrics_schema_id": schema_id,
        "environment": run_tag_canonical_id(run, "environment"),
        "result_status": run.get("resultStatus") or run.get("result_status"),
        "artifact_count": len(artifacts),
        "viewer_states": [artifact_viewer_state(artifact) for artifact in artifacts],
    }


def command_launch_evidence_examples(ctx: Context, args: argparse.Namespace) -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    api = client(ctx)
    checks: list[dict[str, Any]] = []
    examples: dict[str, str] = {}
    for value in args.example:
        scenario, run_id = parse_evidence_example(value)
        examples[scenario] = run_id
    for scenario in sorted(CURATED_EVIDENCE_SCENARIOS):
        run_id = examples.get(scenario)
        if not run_id:
            checks.append(
                {
                    "name": scenario,
                    "status": "fail",
                    "code": "evidence_example_missing",
                    "message": f"Missing curated evidence example for {CURATED_EVIDENCE_SCENARIOS[scenario]}.",
                }
            )
            continue
        launch_smoke_check(checks, scenario, lambda scenario=scenario, run_id=run_id: validate_curated_evidence_example(api, scenario, run_id))
    report = {
        "schema_version": "roborank.launch_evidence_examples.v0",
        "target": args.target,
        "api_url": ctx.api_url,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "counts": launch_smoke_counts(checks),
    }
    if args.out:
        write_json(Path(args.out), report)
    failed = [check for check in checks if check.get("status") == "fail"]
    if failed:
        raise RoboRankError("Curated evidence examples failed verification.", code="launch_evidence_examples_failed", exit_code=9, details=report)
    return emit_success(ctx, "launch.evidence-examples", report)


def public_payload_privacy_probe(payload: Any, surface: str, evidence: dict[str, Any]) -> dict[str, Any]:
    if contains_private_source_code(payload):
        raise RoboRankError(
            "Public evidence surface exposes private policy source code.",
            code="privacy_source_exposure",
            exit_code=9,
            details={"surface": surface, **evidence},
        )
    return evidence


def expect_authenticated_endpoint_denied(api: ApiClient, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = api.request_json("GET", path, params=params)
    except RoboRankError as exc:
        if exc.exit_code == 3 or exc.details.get("status") in {401, 403}:
            return {"path": path, "status": exc.details.get("status"), "code": exc.code}
        raise
    raise RoboRankError(
        "Anonymous request unexpectedly accessed an authenticated evidence endpoint.",
        code="anonymous_access_allowed",
        exit_code=9,
        details={"path": path, "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else []},
    )


def command_launch_privacy_scan(ctx: Context, args: argparse.Namespace) -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    anonymous_api = ApiClient(ctx.api_url)
    authenticated_api = client(ctx) if ctx.options.token else None
    checks: list[dict[str, Any]] = []

    def check_public_explorer() -> dict[str, Any]:
        payload = anonymous_api.request_json("GET", "/api/explorer/runs", params={"limit": args.explorer_limit, "offset": 0})
        runs = check_list_payload(payload, "runs")
        return public_payload_privacy_probe(
            payload,
            "explorer_public",
            {"path": "/api/explorer/runs", "requested": args.explorer_limit, "returned": len(runs)},
        )

    launch_smoke_check(checks, "explorer_public", check_public_explorer)

    if args.run_id:
        def check_public_run_detail() -> dict[str, Any]:
            scanned: list[str] = []
            for run_id in args.run_id:
                path = f"/api/evidence-runs/{api_path_segment(run_id)}"
                payload = anonymous_api.request_json("GET", path)
                public_payload_privacy_probe(payload, "run_detail_public", {"path": path, "run_id": run_id})
                scanned.append(run_id)
            return {"paths": len(scanned), "run_ids": scanned}

        launch_smoke_check(checks, "run_detail_public", check_public_run_detail)
    else:
        launch_smoke_skip(checks, "run_detail_public", "Pass --run-id to scan public evidence run detail payloads.")

    if args.resource_kind and args.resource_id:
        def check_public_resource_page() -> dict[str, Any]:
            namespace, slug = canonical_id_parts(args.resource_id)
            path = f"/api/resources/{api_path_segment(args.resource_kind)}/{api_path_segment(namespace)}/{api_path_segment(slug)}"
            payload = anonymous_api.request_json("GET", path)
            return public_payload_privacy_probe(
                payload,
                "resource_page_public",
                {"path": path, "kind": args.resource_kind, "resource_id": args.resource_id},
            )

        launch_smoke_check(checks, "resource_page_public", check_public_resource_page)
    else:
        launch_smoke_skip(checks, "resource_page_public", "Pass --resource-kind and --resource-id to scan a public resource page payload.")

    compare_runs = list(args.compare_run)
    if len(compare_runs) < 2 and len(args.run_id) >= 2:
        compare_runs = args.run_id[:2]
    if len(compare_runs) >= 2:
        compare_params = {"run": compare_runs}

        def check_compare_anonymous_denied() -> dict[str, Any]:
            validate_compare_run_count(compare_runs)
            return expect_authenticated_endpoint_denied(anonymous_api, "/api/compare", params=compare_params)

        launch_smoke_check(checks, "compare_anonymous_denied", check_compare_anonymous_denied)

        if authenticated_api:
            def check_authenticated_compare() -> dict[str, Any]:
                validate_compare_run_count(compare_runs)
                payload = authenticated_api.request_json("GET", "/api/compare", params=compare_params)
                runs = check_list_payload(payload, "runs")
                return public_payload_privacy_probe(
                    payload,
                    "compare_authenticated",
                    {"path": "/api/compare", "requested": len(compare_runs), "returned": len(runs), "run_ids": compare_runs},
                )

            launch_smoke_check(checks, "compare_authenticated", check_authenticated_compare)
        else:
            launch_smoke_skip(checks, "compare_authenticated", "Pass --token to scan authenticated compare payloads.")
    else:
        launch_smoke_skip(
            checks,
            "compare_anonymous_denied",
            "Pass --compare-run at least twice, or pass at least two --run-id values, to verify compare auth gating.",
        )
        launch_smoke_skip(
            checks,
            "compare_authenticated",
            "Pass --compare-run at least twice, or pass at least two --run-id values, to scan compare payloads.",
        )

    report = {
        "schema_version": "roborank.privacy_scan.v0",
        "target": args.target,
        "api_url": ctx.api_url,
        "access": {
            "public_surfaces": "anonymous",
            "compare": "authenticated" if authenticated_api else "missing_token",
        },
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "strict": args.strict,
        "checks": checks,
        "counts": launch_smoke_counts(checks),
        "source_exposure": any(check.get("code") == "privacy_source_exposure" for check in checks),
    }
    if args.out:
        write_json(Path(args.out), report)
    failed = [check for check in checks if check.get("status") == "fail"]
    incomplete = [check for check in checks if check.get("status") in {"skipped", "warning"}]
    if failed or (args.strict and incomplete):
        raise RoboRankError("Public evidence privacy scan failed.", code="privacy_scan_failed", exit_code=9, details=report)
    return emit_success(ctx, "launch.privacy-scan", report)


def schema_probe_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "title": "Launch schema guard validation probe",
        "summary": "Read-only validation probe generated by roborank launch schema-guard.",
        "tags": {
            "robot": args.robot,
            "environment": args.environment,
            "policy": args.policy,
            "policy_family": args.policy_family,
        },
        "run": {"mode": "simulation", "result_status": "unknown"},
        "license": {"artifact": args.license, "metadata": args.license, "confirmed": True},
        "visibility": "private",
        "allow_new_policy": args.allow_new_policy,
        "allow_new_policy_family": args.allow_new_policy_family,
        "client_upload_id": f"schema_guard_{uuid.uuid4().hex}",
    }


def invalid_metrics_for_schema(schema: dict[str, Any]) -> dict[str, Any]:
    required = [field for field in schema.get("required", []) if isinstance(field, str)] if isinstance(schema.get("required"), list) else []
    if required:
        return {}
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    for field, spec in properties.items():
        if not isinstance(field, str) or not isinstance(spec, dict):
            continue
        if "const" in spec:
            value = spec["const"]
            return {field: "__roborank_invalid_const__" if value != "__roborank_invalid_const__" else "__roborank_invalid_const_alt__"}
        enum_values = spec.get("enum")
        if isinstance(enum_values, list) and enum_values:
            return {field: "__roborank_invalid_enum__"}
        schema_type = spec.get("type")
        if isinstance(schema_type, list):
            schema_type = next((item for item in schema_type if item != "null"), schema_type[0] if schema_type else None)
        if schema_type == "number" or schema_type == "integer":
            return {field: "__roborank_not_a_number__"}
        if schema_type == "boolean":
            return {field: "__roborank_not_a_boolean__"}
        if schema_type == "string":
            return {field: 12345}
        if schema_type == "object":
            return {field: "__roborank_not_an_object__"}
        if schema_type == "array":
            return {field: "__roborank_not_an_array__"}
    raise RoboRankError(
        "Active metrics schema has no required fields or typed properties that schema-guard can use for a negative probe.",
        code="metrics_schema_too_loose",
        exit_code=9,
        details={"schema": schema},
    )


def expect_validation_rejection(api: ApiClient, body: dict[str, Any], probe: str, expected_message: str) -> dict[str, Any]:
    try:
        payload = api.request_json("POST", "/api/evidence-runs/validate", body=body)
    except RoboRankError as exc:
        if exc.details.get("status") == 400:
            message = str(exc)
            if expected_message.lower() not in message.lower():
                raise RoboRankError(
                    "Launch validation probe was rejected for an unexpected reason.",
                    code=f"{probe}_unexpected_rejection",
                    exit_code=9,
                    details={"probe": probe, "message": message, "expected_message": expected_message},
                ) from exc
            return {"status": 400, "message": message}
        raise
    if isinstance(payload, dict) and payload.get("valid") is False:
        message = str(payload.get("detail") or payload.get("message") or "")
        if expected_message.lower() not in message.lower():
            raise RoboRankError(
                "Launch validation probe was rejected for an unexpected reason.",
                code=f"{probe}_unexpected_rejection",
                exit_code=9,
                details={"probe": probe, "response": payload, "expected_message": expected_message},
            )
        return {"status": 200, "valid": False, "response": payload}
    raise RoboRankError(
        "Launch validation probe was accepted by the API.",
        code=f"{probe}_accepted",
        exit_code=9,
        details={"probe": probe, "response": payload},
    )


def command_launch_schema_guard(ctx: Context, args: argparse.Namespace) -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    api = client(ctx)
    checks: list[dict[str, Any]] = []
    schema_payload: dict[str, Any] | None = None
    schema: dict[str, Any] | None = None

    def load_active_schema() -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal schema_payload, schema
        if schema_payload is None:
            schema_payload = fetch_metrics_schema(ctx, args.environment)
            schema = schema_from_response(schema_payload)
        if schema is None:
            raise RoboRankError(
                "Target environment has no active metrics schema.",
                code="metrics_schema_missing",
                exit_code=9,
                details={"environment": args.environment},
            )
        return schema_payload, schema

    def check_metrics_schema_active() -> dict[str, Any]:
        payload, active_schema = load_active_schema()
        schema_info = payload.get("schema") if isinstance(payload.get("schema"), dict) else {}
        return {
            "environment": args.environment,
            "schema_id": schema_info.get("id") or schema_info.get("schemaId"),
            "schema_hash": schema_info.get("schemaHash") or schema_info.get("schema_hash"),
            "required": active_schema.get("required", []),
        }

    launch_smoke_check(checks, "metrics_schema_active", check_metrics_schema_active)

    def check_missing_metrics_rejected() -> dict[str, Any]:
        load_active_schema()
        metadata = schema_probe_metadata(args)
        result = expect_validation_rejection(api, {"metadata": metadata, "metrics": None}, "missing_metrics", "metrics.json is required")
        return {"environment": args.environment, **result}

    launch_smoke_check(checks, "missing_metrics_rejected", check_missing_metrics_rejected)

    def check_invalid_metrics_rejected() -> dict[str, Any]:
        _, active_schema = load_active_schema()
        invalid_metrics = load_json(Path(args.invalid_metrics)) if args.invalid_metrics else invalid_metrics_for_schema(active_schema)
        metadata = schema_probe_metadata(args)
        result = expect_validation_rejection(api, {"metadata": metadata, "metrics": invalid_metrics}, "invalid_metrics", "metrics.json did not validate")
        return {"environment": args.environment, "invalid_metrics_keys": sorted(invalid_metrics), **result}

    launch_smoke_check(checks, "invalid_metrics_rejected", check_invalid_metrics_rejected)

    report = {
        "schema_version": "roborank.schema_guard.v0",
        "target": args.target,
        "api_url": ctx.api_url,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "environment": args.environment,
        "checks": checks,
        "counts": launch_smoke_counts(checks),
    }
    if args.out:
        write_json(Path(args.out), report)
    failed = [check for check in checks if check.get("status") == "fail"]
    if failed:
        raise RoboRankError("Schema-backed metrics guard failed.", code="schema_guard_failed", exit_code=9, details=report)
    return emit_success(ctx, "launch.schema-guard", report)


def run_trust_labels_from_payload(payload: dict[str, Any]) -> list[str]:
    payload = evidence_run_payload(payload) or payload
    labels = payload.get("trustLabels") if isinstance(payload.get("trustLabels"), list) else payload.get("trust_labels")
    return [str(label) for label in labels] if isinstance(labels, list) else []


def resource_context_trust_state(payload: dict[str, Any], run_id: str) -> tuple[str | None, list[str]]:
    runs = payload.get("runs") if isinstance(payload.get("runs"), list) else []
    states: list[str] = []
    target_state: str | None = None
    for item in runs:
        if not isinstance(item, dict):
            continue
        state = item.get("resourceTrustState", item.get("resource_trust_state"))
        state_text = str(state) if state in {"endorsed", "community", "disputed"} else "unknown"
        states.append(state_text)
        if item.get("id") == run_id:
            target_state = state_text
    return target_state, states


def assert_resource_context_order(states: list[str]) -> None:
    rank = {"endorsed": 0, "community": 1, "disputed": 2}
    numeric = [rank.get(state, 99) for state in states]
    if numeric != sorted(numeric):
        raise RoboRankError(
            "Resource-context explorer results are not ordered by trust state.",
            code="resource_context_order_invalid",
            exit_code=9,
            details={"states": states},
        )


def command_launch_trust_actions(ctx: Context, args: argparse.Namespace) -> int:
    if not ctx.options.yes:
        raise RoboRankError("launch trust-actions mutates trust labels and requires --yes.", code="confirmation_required", exit_code=2)
    started_at = datetime.now(timezone.utc).isoformat()
    api = client(ctx)
    checks: list[dict[str, Any]] = []
    namespace, slug = canonical_id_parts(args.resource_id)
    resource_path = f"/api/resources/{api_path_segment(args.resource_kind)}/{api_path_segment(namespace)}/{api_path_segment(slug)}"
    trust_path = f"{resource_path}/runs/{api_path_segment(args.run_id)}"
    resource_context_params = {
        "limit": 50,
        "offset": 0,
        args.resource_kind: args.resource_id,
        "resource_context_kind": args.resource_kind,
        "resource_context": args.resource_id,
    }

    def check_resource_page() -> dict[str, Any]:
        payload = api.request_json("GET", resource_path)
        resource = payload.get("resource") if isinstance(payload, dict) else None
        if not isinstance(resource, dict):
            raise RoboRankError("Resource page returned an invalid response.", code="resource_page_shape_invalid", exit_code=9)
        return {"kind": args.resource_kind, "resource_id": args.resource_id, "path": resource_path}

    launch_smoke_check(checks, "resource_page", check_resource_page)

    def check_run_detail() -> dict[str, Any]:
        payload = api.request_json("GET", f"/api/evidence-runs/{api_path_segment(args.run_id)}")
        run_payload = evidence_run_payload(payload)
        if not run_payload:
            raise RoboRankError("Evidence run detail returned an invalid response.", code="run_detail_shape_invalid", exit_code=9)
        actual_id = run_payload.get("id")
        if actual_id and actual_id != args.run_id:
            raise RoboRankError("Evidence run detail returned an unexpected run ID.", code="run_detail_mismatch", exit_code=9)
        return {"run_id": actual_id or args.run_id, "trust_labels": run_trust_labels_from_payload(run_payload)}

    launch_smoke_check(checks, "run_detail", check_run_detail)

    def check_endorse() -> dict[str, Any]:
        payload = api.request_json("POST", f"{trust_path}/endorse", body={"note": args.note})
        if isinstance(payload, dict) and payload.get("ok") is False:
            raise RoboRankError("Resource endorsement returned ok=false.", code="endorse_failed", exit_code=9, details=payload)
        return {"resource_id": args.resource_id, "run_id": args.run_id}

    launch_smoke_check(checks, "endorse", check_endorse)

    def check_endorsed_resource_context() -> dict[str, Any]:
        payload = api.request_json("GET", "/api/explorer/runs", params=resource_context_params)
        state, states = resource_context_trust_state(payload, args.run_id)
        assert_resource_context_order(states)
        if state != "endorsed":
            raise RoboRankError(
                "Resource-context explorer did not mark the endorsed run as endorsed.",
                code="resource_context_trust_state_mismatch",
                exit_code=9,
                details={"expected": "endorsed", "actual": state, "states": states},
            )
        return {"run_id": args.run_id, "resource_trust_state": state, "states": states}

    launch_smoke_check(checks, "endorsed_resource_context", check_endorsed_resource_context)

    def check_dispute() -> dict[str, Any]:
        payload = api.request_json("POST", f"{trust_path}/dispute", body={"reason": args.reason})
        if isinstance(payload, dict) and payload.get("ok") is False:
            raise RoboRankError("Resource dispute returned ok=false.", code="dispute_failed", exit_code=9, details=payload)
        return {"resource_id": args.resource_id, "run_id": args.run_id}

    launch_smoke_check(checks, "dispute", check_dispute)

    def check_disputed_resource_context() -> dict[str, Any]:
        payload = api.request_json("GET", "/api/explorer/runs", params=resource_context_params)
        state, states = resource_context_trust_state(payload, args.run_id)
        assert_resource_context_order(states)
        if state != "disputed":
            raise RoboRankError(
                "Resource-context explorer did not mark the disputed run as disputed.",
                code="resource_context_trust_state_mismatch",
                exit_code=9,
                details={"expected": "disputed", "actual": state, "states": states},
            )
        return {"run_id": args.run_id, "resource_trust_state": state, "states": states}

    launch_smoke_check(checks, "disputed_resource_context", check_disputed_resource_context)

    def check_trust_labels() -> dict[str, Any]:
        payload = api.request_json("GET", f"/api/evidence-runs/{api_path_segment(args.run_id)}")
        run_payload = evidence_run_payload(payload)
        if not run_payload:
            raise RoboRankError("Evidence run detail returned an invalid response.", code="run_detail_shape_invalid", exit_code=9)
        labels = set(run_trust_labels_from_payload(run_payload))
        missing = sorted({"disputed"} - labels)
        if missing:
            raise RoboRankError(
                "Evidence run detail does not show expected trust labels after dispute.",
                code="trust_labels_missing",
                exit_code=9,
                details={"missing": missing, "labels": sorted(labels)},
            )
        return {"run_id": args.run_id, "trust_labels": sorted(labels)}

    launch_smoke_check(checks, "trust_labels", check_trust_labels)

    report = {
        "schema_version": "roborank.trust_actions.v0",
        "target": args.target,
        "api_url": ctx.api_url,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "resource": {"kind": args.resource_kind, "id": args.resource_id},
        "run_id": args.run_id,
        "checks": checks,
        "counts": launch_smoke_counts(checks),
    }
    if args.out:
        write_json(Path(args.out), report)
    failed = [check for check in checks if check.get("status") == "fail"]
    if failed:
        raise RoboRankError("Resource trust action verification failed.", code="trust_actions_failed", exit_code=9, details=report)
    return emit_success(ctx, "launch.trust-actions", report)


def resource_guard_metadata(
    args: argparse.Namespace,
    *,
    robot: str | None = None,
    environment: str | None = None,
) -> dict[str, Any]:
    return {
        "title": "Launch resource guard validation probe",
        "summary": "Read-only validation probe generated by roborank launch resource-guard.",
        "tags": {
            "robot": robot or args.robot,
            "environment": environment or args.environment,
            "policy": args.policy,
            "policy_family": args.policy_family,
        },
        "run": {"mode": "simulation", "result_status": "unknown"},
        "license": {"artifact": args.license, "metadata": args.license, "confirmed": True},
        "visibility": "private",
        "allow_new_policy": args.allow_new_policy,
        "allow_new_policy_family": args.allow_new_policy_family,
        "client_upload_id": f"resource_guard_{uuid.uuid4().hex}",
    }


def command_launch_resource_guard(ctx: Context, args: argparse.Namespace) -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    api = client(ctx)
    checks: list[dict[str, Any]] = []

    def check_known_resources_validate() -> dict[str, Any]:
        payload = api.request_json("POST", "/api/evidence-runs/validate", body={"metadata": resource_guard_metadata(args), "metrics": None})
        if not isinstance(payload, dict) or payload.get("valid") is not True:
            raise RoboRankError(
                "Known robot/environment validation probe did not return valid=true.",
                code="known_resources_validation_failed",
                exit_code=9,
                details={"response": payload},
            )
        resolved = payload.get("resolved") if isinstance(payload.get("resolved"), dict) else {}
        return {"robot": args.robot, "environment": args.environment, "resolved": resolved}

    launch_smoke_check(checks, "known_resources_validate", check_known_resources_validate)

    def check_missing_robot_rejected() -> dict[str, Any]:
        metadata = resource_guard_metadata(args, robot=args.missing_robot)
        result = expect_validation_rejection(
            api,
            {"metadata": metadata, "metrics": None},
            "missing_robot",
            "Robot and environment resources must exist",
        )
        return {"missing_robot": args.missing_robot, **result}

    launch_smoke_check(checks, "missing_robot_rejected", check_missing_robot_rejected)

    def check_missing_environment_rejected() -> dict[str, Any]:
        metadata = resource_guard_metadata(args, environment=args.missing_environment)
        result = expect_validation_rejection(
            api,
            {"metadata": metadata, "metrics": None},
            "missing_environment",
            "Robot and environment resources must exist",
        )
        return {"missing_environment": args.missing_environment, **result}

    launch_smoke_check(checks, "missing_environment_rejected", check_missing_environment_rejected)

    report = {
        "schema_version": "roborank.resource_guard.v0",
        "target": args.target,
        "api_url": ctx.api_url,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "known_resources": {"robot": args.robot, "environment": args.environment, "policy": args.policy},
        "negative_resources": {"robot": args.missing_robot, "environment": args.missing_environment},
        "checks": checks,
        "counts": launch_smoke_counts(checks),
    }
    if args.out:
        write_json(Path(args.out), report)
    failed = [check for check in checks if check.get("status") == "fail"]
    if failed:
        raise RoboRankError("Resource guard verification failed.", code="resource_guard_failed", exit_code=9, details=report)
    return emit_success(ctx, "launch.resource-guard", report)


def normalized_browser_surface(value: str) -> str:
    normalized = normalized_gate_name(value)
    aliases = {
        "problem": "problem_page",
        "problem_page": "problem_page",
        "explorer": "explorer_desktop",
        "explorer_desktop": "explorer_desktop",
        "explorer_mobile": "explorer_mobile",
        "run": "run_detail",
        "run_detail": "run_detail",
        "compare": "compare",
        "resource": "resource_page",
        "resource_page": "resource_page",
        "leaderboard": "environment_leaderboard",
        "environment_leaderboard": "environment_leaderboard",
    }
    surface = aliases.get(normalized)
    if not surface:
        raise RoboRankError(
            "Unsupported browser QA surface.",
            code="invalid_browser_surface",
            exit_code=2,
            details={"surface": value, "allowed": sorted(GO_NO_GO_REQUIRED_BROWSER_SURFACES)},
        )
    return surface


def parse_browser_surface(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise RoboRankError("--surface must use surface=screenshot_path.", code="usage_error", exit_code=2)
    surface_raw, path_raw = value.split("=", 1)
    if not path_raw:
        raise RoboRankError("--surface screenshot path cannot be empty.", code="usage_error", exit_code=2)
    return normalized_browser_surface(surface_raw), Path(path_raw)


def validate_browser_surface_file(surface: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RoboRankError(
            "Browser QA screenshot is missing.",
            code="browser_qa_screenshot_missing",
            exit_code=9,
            details={"surface": surface, "path": str(path)},
        )
    if not path.is_file():
        raise RoboRankError(
            "Browser QA screenshot path is not a file.",
            code="browser_qa_screenshot_invalid",
            exit_code=9,
            details={"surface": surface, "path": str(path)},
        )
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise RoboRankError(
            "Browser QA screenshot is empty.",
            code="browser_qa_screenshot_empty",
            exit_code=9,
            details={"surface": surface, "path": str(path)},
        )
    return {
        "surface": surface,
        "status": "pass",
        "screenshot": str(path),
        "size_bytes": size_bytes,
        "sha256": sha256_file(path),
    }


def command_launch_browser_qa(ctx: Context, args: argparse.Namespace) -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    checks: list[dict[str, Any]] = []
    surface_paths: dict[str, Path] = {}
    for value in args.surface:
        surface, path = parse_browser_surface(value)
        surface_paths[surface] = path
    surfaces: dict[str, Any] = {}
    for surface in sorted(GO_NO_GO_REQUIRED_BROWSER_SURFACES):
        path = surface_paths.get(surface)
        if path is None:
            checks.append(
                {
                    "name": surface,
                    "status": "fail",
                    "code": "browser_qa_surface_missing",
                    "message": f"Missing browser QA screenshot for {surface}.",
                }
            )
            surfaces[surface] = {"status": "fail", "message": "missing screenshot"}
            continue
        result = launch_smoke_check(checks, surface, lambda surface=surface, path=path: validate_browser_surface_file(surface, path))
        if result:
            surfaces[surface] = result
        else:
            surfaces[surface] = {"status": "fail", "screenshot": str(path)}
    report = {
        "schema_version": "roborank.browser_qa.v0",
        "target": args.target,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "surfaces": surfaces,
        "checks": checks,
        "counts": launch_smoke_counts(checks),
    }
    if args.out:
        write_json(Path(args.out), report)
    failed = [check for check in checks if check.get("status") == "fail"]
    if failed:
        raise RoboRankError("Browser QA verification failed.", code="browser_qa_failed", exit_code=9, details=report)
    return emit_success(ctx, "launch.browser-qa", report)


def parse_known_issue(value: str, index: int) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RoboRankError("--issue must be a JSON object.", code="invalid_known_issue", exit_code=2) from exc
    if not isinstance(parsed, dict):
        raise RoboRankError("--issue must be a JSON object.", code="invalid_known_issue", exit_code=2)
    parsed.setdefault("id", f"issue_{index}")
    parsed.setdefault("status", "open")
    parsed.setdefault("severity", "p2")
    return parsed


def known_issue_blockers(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blockers = []
    for issue in issues:
        severity = normalized_gate_name(issue.get("severity"))
        launch_blocker = bool(issue.get("launch_blocker") or issue.get("launchBlocker"))
        if issue_is_open(issue) and (launch_blocker or severity in {"blocker", "critical", "p0"}):
            blockers.append(issue)
    return blockers


def command_launch_known_issues(ctx: Context, args: argparse.Namespace) -> int:
    issues: list[dict[str, Any]] = []
    if args.from_path:
        source = report_payload(args.from_path)
        issues.extend(known_issue_items(source))
    for index, value in enumerate(args.issue, start=len(issues) + 1):
        issues.append(parse_known_issue(value, index))
    blockers = known_issue_blockers(issues)
    report = {
        "schema_version": "roborank.known_issues.v0",
        "target": args.target,
        "generated_at": now_iso(),
        "issues": issues,
        "counts": {
            "total": len(issues),
            "open": sum(1 for issue in issues if issue_is_open(issue)),
            "open_blockers": len(blockers),
        },
        "blockers": blockers,
    }
    if args.out:
        write_json(Path(args.out), report)
    if blockers:
        raise RoboRankError("Known issues include open launch blockers.", code="known_issues_blocked", exit_code=9, details=report)
    return emit_success(ctx, "launch.known-issues", report)


def issue_identifier(prefix: str, value: Any) -> str:
    name = normalized_gate_name(value) or "issue"
    return f"{prefix}-{name}".replace("_", "-")


def issue_from_failed_check(prefix: str, check: dict[str, Any], *, source: str, severity: str = "critical") -> dict[str, Any]:
    name = normalized_gate_name(check.get("name")) or "check"
    return {
        "id": issue_identifier(prefix, name),
        "title": f"{source} failed: {name.replace('_', ' ')}",
        "severity": severity,
        "status": "open",
        "launch_blocker": severity in {"critical", "blocker", "p0"},
        "source": source,
        "evidence": {
            "check": check.get("name"),
            "message": check.get("message"),
            "code": check.get("code"),
            "details": check.get("details") or check.get("data"),
        },
        "action": "Investigate the failing check, fix the underlying issue, rerun the source report, and update this issue.",
    }


def failed_report_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    checks = report.get("checks")
    if not isinstance(checks, list):
        return []
    return [check for check in checks if isinstance(check, dict) and normalized_gate_name(check.get("status")) == "fail"]


def merge_issue(issues_by_id: dict[str, dict[str, Any]], issue: dict[str, Any]) -> None:
    issue_id = str(issue.get("id") or issue_identifier("issue", len(issues_by_id) + 1))
    issue["id"] = issue_id
    if issue_id not in issues_by_id:
        issues_by_id[issue_id] = issue
        return
    existing = issues_by_id[issue_id]
    existing_sources = existing.setdefault("sources", [])
    if isinstance(existing_sources, list) and issue.get("source") and issue.get("source") not in existing_sources:
        existing_sources.append(issue.get("source"))
    existing.setdefault("evidence_items", []).append(issue.get("evidence", {}))
    if normalized_gate_name(issue.get("severity")) in {"blocker", "critical", "p0"}:
        existing["severity"] = issue.get("severity")
        existing["launch_blocker"] = bool(issue.get("launch_blocker", True))


def issues_from_watch_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    if not report:
        return []
    issues = [issue_from_failed_check("watch", check, source="launch_watch", severity="critical") for check in failed_report_checks(report)]
    for event in report.get("recent_events", []) if isinstance(report.get("recent_events"), list) else []:
        if not isinstance(event, dict):
            continue
        severity = normalized_gate_name(event.get("severity"))
        if severity in {"error", "critical", "fatal"}:
            issues.append(
                {
                    "id": issue_identifier("monitoring-event", event.get("type") or event.get("message") or severity),
                    "title": str(event.get("message") or "Recent severe monitoring event"),
                    "severity": "critical" if severity in {"critical", "fatal"} else "p1",
                    "status": "open",
                    "launch_blocker": severity in {"critical", "fatal"},
                    "source": "monitoring_event",
                    "evidence": event,
                    "action": "Inspect the monitoring event, identify impacted users or runs, and update the incident notes.",
                }
            )
    return issues


def issues_from_migration_progress(report: dict[str, Any]) -> list[dict[str, Any]]:
    if not report or report.get("decision") != "pause":
        return []
    return [
        {
            "id": "migration-progress-pause",
            "title": "Migration batches are paused",
            "severity": "critical",
            "status": "open",
            "launch_blocker": True,
            "source": "migration_progress",
            "evidence": {
                "pause_reasons": report.get("pause_reasons", []),
                "counts": report.get("counts", {}),
                "health": report.get("health", {}),
            },
            "action": "Keep migration batches paused until failure causes are fixed and migration progress returns continue or complete.",
        }
    ]


def issues_from_go_no_go(report: dict[str, Any]) -> list[dict[str, Any]]:
    if not report or report.get("decision") in {None, "go"}:
        return []
    issues = []
    for check in failed_report_checks(report):
        issues.append(issue_from_failed_check("go-no-go", check, source="launch_go_no_go", severity="critical"))
    return issues


def issue_counts(issues: list[dict[str, Any]], blockers: list[dict[str, Any]]) -> dict[str, Any]:
    by_severity: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for issue in issues:
        severity = normalized_gate_name(issue.get("severity") or "p2")
        source = normalized_gate_name(issue.get("source") or "manual")
        by_severity[severity] = by_severity.get(severity, 0) + 1
        by_source[source] = by_source.get(source, 0) + 1
    return {
        "total": len(issues),
        "open": sum(1 for issue in issues if issue_is_open(issue)),
        "open_blockers": len(blockers),
        "by_severity": by_severity,
        "by_source": by_source,
    }


def triage_columns(issues: list[dict[str, Any]], blockers: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "launch_blockers": [str(issue.get("id")) for issue in blockers],
        "monitoring": [str(issue.get("id")) for issue in issues if normalized_gate_name(issue.get("source")) in {"launch_watch", "monitoring_event"}],
        "migration": [str(issue.get("id")) for issue in issues if normalized_gate_name(issue.get("source")) == "migration_progress"],
        "known": [str(issue.get("id")) for issue in issues if normalized_gate_name(issue.get("source")) in {"known_issues", "manual", ""}],
        "resolved_or_deferred": [str(issue.get("id")) for issue in issues if not issue_is_open(issue)],
    }


def command_launch_triage(ctx: Context, args: argparse.Namespace) -> int:
    issues_by_id: dict[str, dict[str, Any]] = {}
    for path in args.known_issues or []:
        for issue in known_issue_items(report_payload(path)):
            issue.setdefault("source", "known_issues")
            merge_issue(issues_by_id, issue)
    for index, value in enumerate(args.issue, start=len(issues_by_id) + 1):
        issue = parse_known_issue(value, index)
        issue.setdefault("source", "manual")
        merge_issue(issues_by_id, issue)
    watch_report = report_payload(args.watch_report) if args.watch_report else {}
    migration_progress = report_payload(args.migration_progress_report) if args.migration_progress_report else {}
    smoke_report = report_payload(args.smoke_report) if args.smoke_report else {}
    go_no_go_report = report_payload(args.go_no_go_report) if args.go_no_go_report else {}
    for issue in issues_from_watch_report(watch_report):
        merge_issue(issues_by_id, issue)
    for issue in issues_from_migration_progress(migration_progress):
        merge_issue(issues_by_id, issue)
    for issue in [issue_from_failed_check("smoke", check, source="launch_smoke", severity="critical") for check in failed_report_checks(smoke_report)]:
        merge_issue(issues_by_id, issue)
    for issue in issues_from_go_no_go(go_no_go_report):
        merge_issue(issues_by_id, issue)
    issues = sorted(issues_by_id.values(), key=lambda item: (not bool(item.get("launch_blocker")), normalized_gate_name(item.get("severity")), str(item.get("id"))))
    blockers = known_issue_blockers(issues)
    report = {
        "schema_version": "roborank.post_launch_triage.v0",
        "target": args.target,
        "generated_at": now_iso(),
        "decision": "investigate" if blockers else "monitor",
        "issues": issues,
        "columns": triage_columns(issues, blockers),
        "counts": issue_counts(issues, blockers),
        "source_reports": {
            "known_issues": args.known_issues,
            "watch_report": args.watch_report,
            "migration_progress_report": args.migration_progress_report,
            "smoke_report": args.smoke_report,
            "go_no_go_report": args.go_no_go_report,
        },
    }
    if args.out:
        write_json(Path(args.out), report)
    if blockers and not args.allow_open_blockers:
        raise RoboRankError("Post-launch triage contains open blockers.", code="post_launch_triage_blocked", exit_code=9, details=report)
    return emit_success(ctx, "launch.triage", report)


def command_launch_signoff(ctx: Context, args: argparse.Namespace) -> int:
    signoffs: dict[str, Any] = {}
    if args.from_path:
        source = report_payload(args.from_path)
        signoffs.update(normalized_signoff_roles(source))
    for role in GO_NO_GO_REQUIRED_SIGNOFF_ROLES:
        value = getattr(args, role, None)
        if value is not None:
            signoffs[role] = value
    missing = sorted(role for role in GO_NO_GO_REQUIRED_SIGNOFF_ROLES if not signoff_value_approved(signoffs.get(role)))
    report = {
        "schema_version": "roborank.launch_signoff.v0",
        "target": args.target,
        "generated_at": now_iso(),
        "signoffs": signoffs,
        "missing": missing,
    }
    if args.out:
        write_json(Path(args.out), report)
    if missing:
        raise RoboRankError("Required launch signoffs are missing.", code="launch_signoff_missing", exit_code=9, details=report)
    return emit_success(ctx, "launch.signoff", report)


def int_stat(stats: dict[str, Any], key: str) -> int:
    value = stats.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def watch_thresholds_from_args(args: argparse.Namespace) -> dict[str, int]:
    return {
        "uploadFailures7d": args.max_upload_failures,
        "metricsValidationFailures7d": args.max_metrics_validation_failures,
        "storageFailures7d": args.max_storage_failures,
        "viewerFailures7d": args.max_viewer_failures,
        "explorerSlowEvents7d": args.max_explorer_slow_events,
        "migrationFailures7d": args.max_migration_failures,
        "rateLimitEvents7d": args.max_rate_limit_events,
    }


def command_launch_watch(ctx: Context, args: argparse.Namespace) -> int:
    payload = client(ctx).request_json("GET", "/api/admin/dashboard", params={"limit": 1, "offset": 0})
    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    monitoring = payload.get("monitoring") if isinstance(payload.get("monitoring"), dict) else {}
    recent_events = monitoring.get("recentEvents") if isinstance(monitoring.get("recentEvents"), list) else []
    checks: list[dict[str, Any]] = []
    thresholds = watch_thresholds_from_args(args)
    for key, threshold in thresholds.items():
        actual = int_stat(stats, key)
        checks.append(
            {
                "name": normalized_gate_name(LAUNCH_WATCH_STAT_LABELS[key]),
                "status": "pass" if actual <= threshold else "fail",
                "message": f"{LAUNCH_WATCH_STAT_LABELS[key]} 7d = {actual}, threshold = {threshold}",
                "data": {"stat": key, "actual": actual, "threshold": threshold},
            }
        )
    severe_events = []
    fail_severities = {normalized_gate_name(value) for value in args.fail_on_severity}
    for event in recent_events:
        if isinstance(event, dict) and normalized_gate_name(event.get("severity")) in fail_severities:
            severe_events.append(event)
    checks.append(
        {
            "name": "recent_monitoring_event_severity",
            "status": "pass" if not severe_events else "fail",
            "message": "No recent monitoring events matched fail severities." if not severe_events else "Recent monitoring events matched fail severities.",
            "data": {"fail_on_severity": sorted(fail_severities), "events": severe_events},
        }
    )
    if args.min_evidence_runs is not None:
        total_evidence = int_stat(stats, "totalEvidenceRuns")
        checks.append(
            {
                "name": "evidence_run_count",
                "status": "pass" if total_evidence >= args.min_evidence_runs else "fail",
                "message": f"total evidence runs = {total_evidence}, minimum = {args.min_evidence_runs}",
                "data": {"actual": total_evidence, "minimum": args.min_evidence_runs},
            }
        )
    report = {
        "schema_version": "roborank.launch_watch.v0",
        "target": args.target,
        "api_url": ctx.api_url,
        "generated_at": now_iso(),
        "stats": stats,
        "recent_events": recent_events,
        "checks": checks,
        "counts": launch_smoke_counts(checks),
    }
    if args.out:
        write_json(Path(args.out), report)
    failed = [check for check in checks if check.get("status") == "fail"]
    if failed:
        raise RoboRankError("Launch watch detected post-launch monitoring failures.", code="launch_watch_failed", exit_code=9, details=report)
    return emit_success(ctx, "launch.watch", report)


def cli_init_version(cli_root: Path) -> str | None:
    init_path = cli_root / "roborank" / "__init__.py"
    if not init_path.exists():
        init_path = cli_root / "src" / "roborank" / "__init__.py"
    if not init_path.exists():
        return None
    for line in init_path.read_text().splitlines():
        if line.strip().startswith("__version__") and "=" in line:
            return line.split("=", 1)[1].strip().strip("\"'")
    return None


def cli_release_command_check(name: str, command: list[str], *, cwd: Path, timeout: int) -> dict[str, Any]:
    started = time.time()
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "name": name,
            "status": "fail",
            "code": "cli_release_command_missing",
            "message": f"{command[0]} is not installed or not on PATH.",
            "duration_ms": round((time.time() - started) * 1000),
            "command": command,
            "details": {"error": str(exc)},
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "status": "fail",
            "code": "cli_release_command_timeout",
            "message": f"{name} timed out after {timeout} seconds.",
            "duration_ms": round((time.time() - started) * 1000),
            "command": command,
            "stdout_tail": tail_text(exc.stdout or ""),
            "stderr_tail": tail_text(exc.stderr or ""),
        }
    check: dict[str, Any] = {
        "name": name,
        "status": "pass" if process.returncode == 0 else "fail",
        "duration_ms": round((time.time() - started) * 1000),
        "command": command,
        "returncode": process.returncode,
    }
    if process.returncode != 0:
        check.update(
            {
                "code": "cli_release_command_failed",
                "message": f"{name} exited with status {process.returncode}.",
                "stdout_tail": tail_text(process.stdout),
                "stderr_tail": tail_text(process.stderr),
            }
        )
    else:
        check["stdout_tail"] = tail_text(process.stdout, 20000)
    return check


def cli_release_artifact_payload(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def command_launch_cli_release(ctx: Context, args: argparse.Namespace) -> int:
    cli_root = Path(args.cli_root).resolve()
    dist_dir = Path(args.dist_dir).resolve()
    checks: list[dict[str, Any]] = []
    pyproject_path = cli_root / "pyproject.toml"
    if not pyproject_path.exists():
        checks.append({"name": "package_metadata", "status": "fail", "code": "pyproject_missing", "message": "pyproject.toml is missing."})
        metadata: dict[str, Any] = {}
    else:
        metadata = tomllib.loads(pyproject_path.read_text()).get("project", {})
        checks.append(
            {
                "name": "package_metadata",
                "status": "pass" if metadata.get("name") and metadata.get("version") else "fail",
                "message": "CLI package metadata found." if metadata.get("name") and metadata.get("version") else "CLI package metadata missing name or version.",
                "data": {"name": metadata.get("name"), "version": metadata.get("version")},
            }
        )
    package_name = str(metadata.get("name") or "roborank")
    package_version = str(metadata.get("version") or "")
    init_version = cli_init_version(cli_root)
    checks.append(
        {
            "name": "version_consistency",
            "status": "pass" if package_version and init_version == package_version else "fail",
            "message": "pyproject version matches roborank.__version__." if package_version and init_version == package_version else "pyproject and roborank.__version__ differ.",
            "data": {"pyproject_version": package_version, "package_version": init_version},
        }
    )
    uv = shutil.which("uv")
    if not uv:
        checks.append({"name": "build_artifacts", "status": "fail", "code": "uv_missing", "message": "uv is required to build the CLI release."})
        artifacts: list[dict[str, Any]] = []
    else:
        dist_dir.mkdir(parents=True, exist_ok=True)
        if args.clean:
            for path in dist_dir.glob(f"{package_name.replace('-', '_')}-*"):
                if path.is_file():
                    path.unlink()
            for path in dist_dir.glob(f"{package_name}-*"):
                if path.is_file():
                    path.unlink()
        build_check = cli_release_command_check("build_artifacts", [uv, "build", str(cli_root), "--out-dir", str(dist_dir)], cwd=repo_root(), timeout=args.timeout)
        checks.append(build_check)
        built_paths = sorted(path for path in dist_dir.iterdir() if path.is_file() and path.name.startswith(f"{package_name.replace('-', '_')}-{package_version}"))
        built_paths.extend(
            path
            for path in sorted(dist_dir.iterdir())
            if path.is_file() and path.name.startswith(f"{package_name}-{package_version}") and path not in built_paths
        )
        wheel_paths = [path for path in built_paths if path.suffix == ".whl"]
        sdist_paths = [path for path in built_paths if path.name.endswith(".tar.gz")]
        artifacts = [cli_release_artifact_payload(path) for path in built_paths]
        if build_check["status"] == "pass" and (not wheel_paths or not sdist_paths):
            build_check.update(
                {
                    "status": "fail",
                    "code": "cli_release_artifacts_missing",
                    "message": "CLI build did not produce both wheel and source distribution.",
                    "data": {"wheel_count": len(wheel_paths), "sdist_count": len(sdist_paths), "dist_dir": str(dist_dir)},
                }
            )
        if wheel_paths:
            with tempfile.TemporaryDirectory(prefix="roborank-cli-release-") as directory:
                smoke_check = cli_release_command_check(
                    "entrypoint_smoke",
                    [uv, "run", "--with", str(wheel_paths[-1]), "roborank", "prime", "--agent", "--json"],
                    cwd=Path(directory),
                    timeout=args.timeout,
                )
            if smoke_check["status"] == "pass":
                try:
                    payload = json.loads(smoke_check.get("stdout_tail") or "{}")
                    if not (payload.get("ok") is True and payload.get("command") == "prime"):
                        smoke_check.update(
                            {
                                "status": "fail",
                                "code": "cli_release_smoke_invalid",
                                "message": "Installed CLI did not return the expected prime envelope.",
                                "data": {"command": payload.get("command"), "ok": payload.get("ok")},
                            }
                        )
                except json.JSONDecodeError as exc:
                    smoke_check.update(
                        {
                            "status": "fail",
                            "code": "cli_release_smoke_invalid_json",
                            "message": f"Installed CLI returned invalid JSON: {exc}",
                        }
                    )
            checks.append(smoke_check)
        else:
            checks.append({"name": "entrypoint_smoke", "status": "fail", "code": "wheel_missing", "message": "Cannot smoke-test CLI entry point without a built wheel."})
    report = {
        "schema_version": "roborank.cli_release.v0",
        "target": args.target,
        "generated_at": now_iso(),
        "package": {"name": package_name, "version": package_version},
        "cli_root": str(cli_root),
        "dist_dir": str(dist_dir),
        "artifacts": artifacts,
        "checks": checks,
        "counts": launch_smoke_counts(checks),
        "publish_ready": all(check.get("status") == "pass" for check in checks),
    }
    if args.out:
        write_json(Path(args.out), report)
    if not report["publish_ready"]:
        raise RoboRankError("CLI release verification failed.", code="cli_release_failed", exit_code=9, details=report)
    return emit_success(ctx, "launch.cli-release", report)


def tail_text(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def run_launch_preflight_check(name: str, root: Path, timeout: int) -> dict[str, Any]:
    command = LAUNCH_PREFLIGHT_COMMANDS[name]
    started = time.time()
    input_text = None
    if name == "d1_schema_import":
        schema_path = root / "cloudflare" / "schema.sql"
        if not schema_path.exists():
            return {
                "name": name,
                "status": "fail",
                "code": "schema_sql_missing",
                "message": "cloudflare/schema.sql does not exist.",
                "duration_ms": round((time.time() - started) * 1000),
                "command": command,
            }
        input_text = schema_path.read_text()
    try:
        process = subprocess.run(
            command,
            cwd=root,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "name": name,
            "status": "fail",
            "code": "preflight_command_missing",
            "message": f"{command[0]} is not installed or not on PATH.",
            "duration_ms": round((time.time() - started) * 1000),
            "command": command,
            "details": {"error": str(exc)},
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "status": "fail",
            "code": "preflight_command_timeout",
            "message": f"{name} timed out after {timeout} seconds.",
            "duration_ms": round((time.time() - started) * 1000),
            "command": command,
            "stdout_tail": tail_text(exc.stdout or ""),
            "stderr_tail": tail_text(exc.stderr or ""),
        }
    check: dict[str, Any] = {
        "name": name,
        "status": "pass" if process.returncode == 0 else "fail",
        "duration_ms": round((time.time() - started) * 1000),
        "command": command,
        "returncode": process.returncode,
    }
    if process.returncode != 0:
        check["code"] = "preflight_command_failed"
        check["message"] = f"{name} exited with status {process.returncode}."
        check["stdout_tail"] = tail_text(process.stdout)
        check["stderr_tail"] = tail_text(process.stderr)
    return check


def command_launch_preflight(ctx: Context, args: argparse.Namespace) -> int:
    requested = args.check or list(LAUNCH_PREFLIGHT_COMMANDS)
    skipped = set(args.skip or [])
    unknown = sorted((set(requested) | skipped) - set(LAUNCH_PREFLIGHT_COMMANDS))
    if unknown:
        raise RoboRankError(
            "Unknown launch preflight check.",
            code="unknown_preflight_check",
            exit_code=2,
            details={"unknown": unknown, "available": sorted(LAUNCH_PREFLIGHT_COMMANDS)},
        )
    checks: list[dict[str, Any]] = []
    root = Path(args.repo_root).resolve()
    for name in requested:
        if name in skipped:
            checks.append({"name": name, "status": "skipped", "message": "Skipped by --skip.", "command": LAUNCH_PREFLIGHT_COMMANDS[name]})
            continue
        checks.append(run_launch_preflight_check(name, root, args.timeout))
    report = {
        "schema_version": "roborank.launch_preflight.v0",
        "target": args.target,
        "repo_root": str(root),
        "generated_at": now_iso(),
        "checks": checks,
        "counts": launch_smoke_counts(checks),
    }
    if args.out:
        write_json(Path(args.out), report)
    if any(check.get("status") == "fail" for check in checks):
        raise RoboRankError("Launch preflight checks failed.", code="launch_preflight_failed", exit_code=9, details=report)
    return emit_success(ctx, "launch.preflight", report)


def command_launch_restore_check(ctx: Context, args: argparse.Namespace) -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    checks: list[dict[str, Any]] = []
    if args.d1_sql:
        launch_smoke_check(checks, "d1_sql_import", lambda: restore_check_d1_sql(Path(args.d1_sql), args.required_table))
    else:
        launch_smoke_skip(checks, "d1_sql_import", "Pass --d1-sql to verify a restored D1 SQL export.")

    if args.object_manifest:
        launch_smoke_check(
            checks,
            "object_integrity",
            lambda: restore_check_objects(Path(args.object_manifest), Path(args.object_root) if args.object_root else None, args.required_object),
        )
    else:
        launch_smoke_skip(checks, "object_integrity", "Pass --object-manifest and --object-root to verify restored object bytes.")

    if args.api_smoke_report:
        launch_smoke_check(checks, "api_smoke_report", lambda: restore_check_smoke_report(Path(args.api_smoke_report)))
    else:
        launch_smoke_skip(checks, "api_smoke_report", "Pass --api-smoke-report from launch smoke to bind restore checks to live staging probes.")

    report = {
        "schema_version": "roborank.launch_restore_check.v0",
        "target": args.target,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "strict": args.strict,
        "checks": checks,
        "counts": launch_smoke_counts(checks),
    }
    failed = [check for check in checks if check.get("status") == "fail"]
    incomplete = [check for check in checks if check.get("status") in {"skipped", "warning"}]
    if failed or (args.strict and incomplete):
        raise RoboRankError(
            "Launch restore checks failed.",
            code="launch_restore_check_failed",
            exit_code=9,
            details=report,
        )
    return emit_success(ctx, "launch.restore-check", report)


def file_artifact_summary(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def command_launch_backup_manifest(ctx: Context, args: argparse.Namespace) -> int:
    d1_sql = Path(args.d1_sql)
    object_manifest = Path(args.object_manifest)
    object_root = Path(args.object_root) if args.object_root else None
    checks: list[dict[str, Any]] = []
    launch_smoke_check(checks, "d1_sql_import", lambda: restore_check_d1_sql(d1_sql, args.required_table))
    launch_smoke_check(checks, "object_integrity", lambda: restore_check_objects(object_manifest, object_root, args.required_object))
    try:
        object_count = len(object_manifest_entries(object_manifest))
    except RoboRankError:
        object_count = 0
    report = {
        "schema_version": "roborank.backup_manifest.v0",
        "target": args.target,
        "label": args.label,
        "generated_at": now_iso(),
        "backups": {
            "d1_sql": file_artifact_summary(d1_sql),
            "object_manifest": {
                **file_artifact_summary(object_manifest),
                "object_count": object_count,
                "object_root": str(object_root) if object_root else None,
            },
        },
        "checks": checks,
        "counts": launch_smoke_counts(checks),
    }
    if args.out:
        write_json(Path(args.out), report)
    if any(check.get("status") == "fail" for check in checks):
        raise RoboRankError("Backup manifest verification failed.", code="backup_manifest_failed", exit_code=9, details=report)
    return emit_success(ctx, "launch.backup-manifest", report)


def restore_check_d1_sql(path: Path, extra_required_tables: Iterable[str]) -> dict[str, Any]:
    if not path.exists():
        raise RoboRankError(f"{path} does not exist.", code="file_missing", exit_code=2)
    sql = path.read_text()
    try:
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(sql)
            rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise RoboRankError(
            "D1 SQL export could not be imported into a clean SQLite database.",
            code="d1_restore_import_failed",
            exit_code=9,
            details={"path": str(path), "error": str(exc)},
        ) from exc
    tables = sorted(str(row[0]) for row in rows)
    required = sorted(D1_RESTORE_REQUIRED_TABLES | set(extra_required_tables or []))
    missing = [table for table in required if table not in tables]
    if missing:
        raise RoboRankError(
            "D1 SQL export is missing required Rerun-first tables.",
            code="d1_restore_tables_missing",
            exit_code=9,
            details={"path": str(path), "missing": missing, "tables": tables},
        )
    return {"path": str(path), "tables": tables, "required_tables": required}


def object_manifest_entries(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    raw = payload.get("objects")
    if raw is None:
        raw = payload.get("artifacts")
    if not isinstance(raw, list):
        raise RoboRankError("Object manifest must contain an objects or artifacts list.", code="invalid_object_manifest", exit_code=2)
    return [item for item in raw if isinstance(item, dict)]


def manifest_object_key(item: dict[str, Any]) -> str | None:
    for key in ("key", "storage_key", "storageKey", "path", "name"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value.lstrip("/")
    storage_uri = item.get("storage_uri") or item.get("storageUri")
    if isinstance(storage_uri, str) and storage_uri.startswith("r2://"):
        return storage_uri.split("/", 3)[-1].lstrip("/")
    return None


def manifest_object_path(item: dict[str, Any], object_root: Path | None, object_key: str) -> Path | None:
    for key in ("local_path", "localPath", "file", "filepath"):
        value = item.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            return path if path.is_absolute() else (object_root / path if object_root else path)
    if object_root:
        return object_root / object_key
    return None


def manifest_hash_value(item: dict[str, Any]) -> str | None:
    for key in ("sha256", "hash", "checksum"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value.removeprefix("sha256:").lower()
    return None


def manifest_size_value(item: dict[str, Any]) -> int | None:
    for key in ("size_bytes", "sizeBytes", "size"):
        value = item.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def restore_check_objects(manifest_path: Path, object_root: Path | None, required_objects: Iterable[str]) -> dict[str, Any]:
    entries = object_manifest_entries(manifest_path)
    objects_by_key: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    verified = 0
    for item in entries:
        key = manifest_object_key(item)
        if not key:
            failures.append({"reason": "missing_key", "object": item})
            continue
        objects_by_key[key] = item
        object_path = manifest_object_path(item, object_root, key)
        if object_path is None:
            failures.append({"key": key, "reason": "missing_object_root"})
            continue
        if not object_path.exists():
            failures.append({"key": key, "path": str(object_path), "reason": "missing_file"})
            continue
        expected_size = manifest_size_value(item)
        actual_size = object_path.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            failures.append({"key": key, "path": str(object_path), "reason": "size_mismatch", "expected": expected_size, "actual": actual_size})
            continue
        expected_hash = manifest_hash_value(item)
        if expected_hash:
            actual_hash = sha256_file(object_path)
            if actual_hash.lower() != expected_hash:
                failures.append(
                    {
                        "key": key,
                        "path": str(object_path),
                        "reason": "sha256_mismatch",
                        "expected": expected_hash,
                        "actual": actual_hash,
                    }
                )
                continue
        verified += 1
    missing_required = [key for key in required_objects if key not in objects_by_key]
    if missing_required:
        failures.append({"reason": "required_object_missing", "keys": missing_required})
    if failures:
        raise RoboRankError(
            "Restored object integrity check failed.",
            code="restore_object_integrity_failed",
            exit_code=9,
            details={"manifest": str(manifest_path), "object_root": str(object_root) if object_root else None, "failures": failures},
        )
    return {
        "manifest": str(manifest_path),
        "object_root": str(object_root) if object_root else None,
        "manifest_count": len(entries),
        "verified_count": verified,
        "required_objects": list(required_objects),
    }


def restore_check_smoke_report(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema_version") != "roborank.launch_smoke.v0":
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        if isinstance(result, dict) and result.get("schema_version") == "roborank.launch_smoke.v0":
            payload = result
        else:
            raise RoboRankError("Smoke report is not a roborank.launch_smoke.v0 payload.", code="invalid_smoke_report", exit_code=2)
    checks = payload.get("checks")
    if not isinstance(checks, list):
        raise RoboRankError("Smoke report is missing checks.", code="invalid_smoke_report", exit_code=2)
    failed = [check for check in checks if isinstance(check, dict) and check.get("status") == "fail"]
    if failed:
        raise RoboRankError(
            "Smoke report contains failed checks.",
            code="smoke_report_failed",
            exit_code=9,
            details={"path": str(path), "failed": failed},
        )
    required = {
        "auth",
        "problem_list",
        "artifact_download",
        "explorer_anonymous_limit",
        "explorer_facets",
        "run_detail",
        "resource_page",
        "environment_leaderboard",
        "legacy_leaderboard",
        "legacy_submission_detail",
    }
    present = {str(check.get("name")) for check in checks if isinstance(check, dict) and check.get("status") == "pass"}
    missing = sorted(required - present)
    if missing:
        raise RoboRankError(
            "Smoke report does not prove the required restore drill probes.",
            code="smoke_report_required_checks_missing",
            exit_code=9,
            details={"path": str(path), "missing": missing, "present": sorted(present)},
        )
    return {"path": str(path), "passed_checks": sorted(present), "check_count": len(checks)}


def normalized_gate_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    normalized = "".join(char if char.isalnum() else "_" for char in text)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def report_check_statuses(report: dict[str, Any]) -> dict[str, str]:
    checks = report.get("checks")
    if not isinstance(checks, list):
        return {}
    statuses: dict[str, str] = {}
    for check in checks:
        if isinstance(check, dict):
            name = normalized_gate_name(check.get("name"))
            status = normalized_gate_name(check.get("status"))
            if name:
                statuses[name] = status
    return statuses


def gate_result(name: str, status: str, message: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "status": status, "message": message}
    if evidence:
        result["evidence"] = evidence
    return result


def required_pass_statuses(statuses: dict[str, str], required: set[str]) -> list[str]:
    return sorted(name for name in required if statuses.get(name) != "pass")


def validate_compare_run_count(run_ids: list[str]) -> None:
    if len(run_ids) > 4:
        raise RoboRankError(
            "Compare supports at most four run IDs.",
            code="compare_too_many_runs",
            exit_code=9,
            details={"count": len(run_ids), "max": 4},
        )


def evaluate_launch_smoke_report(report: dict[str, Any]) -> dict[str, Any]:
    statuses = report_check_statuses(report)
    missing = required_pass_statuses(statuses, GO_NO_GO_REQUIRED_SMOKE_CHECKS)
    failed = sorted(name for name, status in statuses.items() if status == "fail")
    if report.get("schema_version") != "roborank.launch_smoke.v0":
        return gate_result("smoke_report", "fail", "Smoke report has an unsupported schema version.", {"schema_version": report.get("schema_version")})
    if failed or missing:
        return gate_result("smoke_report", "fail", "Smoke report does not prove all required production smoke checks.", {"failed": failed, "missing": missing})
    return gate_result("smoke_report", "pass", "Required smoke checks passed.", {"checks": sorted(GO_NO_GO_REQUIRED_SMOKE_CHECKS)})


def evaluate_preflight_report(report: dict[str, Any]) -> dict[str, Any]:
    statuses = report_check_statuses(report)
    required = set(LAUNCH_PREFLIGHT_COMMANDS)
    missing = required_pass_statuses(statuses, required)
    failed = sorted(name for name, status in statuses.items() if status == "fail")
    if report.get("schema_version") != "roborank.launch_preflight.v0":
        return gate_result("preflight_report", "fail", "Preflight report has an unsupported schema version.", {"schema_version": report.get("schema_version")})
    if failed or missing:
        return gate_result("preflight_report", "fail", "Preflight report does not prove all required local release checks.", {"failed": failed, "missing": missing})
    return gate_result("preflight_report", "pass", "Required local release checks passed.", {"checks": sorted(required)})


def evaluate_restore_report(report: dict[str, Any]) -> dict[str, Any]:
    statuses = report_check_statuses(report)
    required = {"d1_sql_import", "object_integrity", "api_smoke_report"}
    missing = required_pass_statuses(statuses, required)
    failed = sorted(name for name, status in statuses.items() if status == "fail")
    if report.get("schema_version") != "roborank.launch_restore_check.v0":
        return gate_result("restore_report", "fail", "Restore report has an unsupported schema version.", {"schema_version": report.get("schema_version")})
    if failed or missing:
        return gate_result("restore_report", "fail", "Restore report does not prove D1, object, and API restore checks.", {"failed": failed, "missing": missing})
    return gate_result("restore_report", "pass", "Restore drill checks passed.", {"checks": sorted(required)})


def evaluate_backup_manifest_report(report: dict[str, Any]) -> dict[str, Any]:
    statuses = report_check_statuses(report)
    required = {"d1_sql_import", "object_integrity"}
    missing = required_pass_statuses(statuses, required)
    failed = sorted(name for name, status in statuses.items() if status == "fail")
    if report.get("schema_version") != "roborank.backup_manifest.v0":
        return gate_result("backup_manifest", "fail", "Backup manifest report has an unsupported schema version.", {"schema_version": report.get("schema_version")})
    if failed or missing:
        return gate_result("backup_manifest", "fail", "Backup manifest does not prove D1 and object backup integrity.", {"failed": failed, "missing": missing})
    return gate_result("backup_manifest", "pass", "Backup manifest proves D1 import and object integrity.", {"backups": report.get("backups", {})})


def evaluate_evidence_examples_report(report: dict[str, Any]) -> dict[str, Any]:
    statuses = report_check_statuses(report)
    required = set(CURATED_EVIDENCE_SCENARIOS)
    missing = required_pass_statuses(statuses, required)
    failed = sorted(name for name, status in statuses.items() if status == "fail")
    if report.get("schema_version") != "roborank.launch_evidence_examples.v0":
        return gate_result(
            "evidence_examples",
            "fail",
            "Curated evidence examples report has an unsupported schema version.",
            {"schema_version": report.get("schema_version")},
        )
    if failed or missing:
        return gate_result(
            "evidence_examples",
            "fail",
            "Curated evidence examples report does not prove all required scenarios.",
            {"failed": failed, "missing": missing},
        )
    return gate_result("evidence_examples", "pass", "Required curated evidence scenarios passed.", {"scenarios": sorted(required)})


def evaluate_privacy_scan_report(report: dict[str, Any]) -> dict[str, Any]:
    statuses = report_check_statuses(report)
    missing = required_pass_statuses(statuses, GO_NO_GO_REQUIRED_PRIVACY_CHECKS)
    failed = sorted(name for name, status in statuses.items() if status == "fail")
    if report.get("schema_version") != "roborank.privacy_scan.v0":
        return gate_result(
            "privacy_scan",
            "fail",
            "Privacy scan report has an unsupported schema version.",
            {"schema_version": report.get("schema_version")},
        )
    if report.get("source_exposure") is True or failed or missing:
        return gate_result(
            "privacy_scan",
            "fail",
            "Privacy scan does not prove all required public evidence surfaces are source-free.",
            {"failed": failed, "missing": missing, "source_exposure": bool(report.get("source_exposure"))},
        )
    return gate_result(
        "privacy_scan",
        "pass",
        "Required public evidence privacy checks passed.",
        {"checks": sorted(GO_NO_GO_REQUIRED_PRIVACY_CHECKS)},
    )


def evaluate_schema_guard_report(report: dict[str, Any]) -> dict[str, Any]:
    statuses = report_check_statuses(report)
    missing = required_pass_statuses(statuses, GO_NO_GO_REQUIRED_SCHEMA_GUARD_CHECKS)
    failed = sorted(name for name, status in statuses.items() if status == "fail")
    if report.get("schema_version") != "roborank.schema_guard.v0":
        return gate_result(
            "schema_guard",
            "fail",
            "Schema guard report has an unsupported schema version.",
            {"schema_version": report.get("schema_version")},
        )
    if failed or missing:
        return gate_result(
            "schema_guard",
            "fail",
            "Schema guard does not prove schema-backed metrics rejection behavior.",
            {"failed": failed, "missing": missing},
        )
    return gate_result(
        "schema_guard",
        "pass",
        "Required schema-backed metrics guard checks passed.",
        {"checks": sorted(GO_NO_GO_REQUIRED_SCHEMA_GUARD_CHECKS), "environment": report.get("environment")},
    )


def evaluate_trust_actions_report(report: dict[str, Any]) -> dict[str, Any]:
    statuses = report_check_statuses(report)
    missing = required_pass_statuses(statuses, GO_NO_GO_REQUIRED_TRUST_ACTION_CHECKS)
    failed = sorted(name for name, status in statuses.items() if status == "fail")
    if report.get("schema_version") != "roborank.trust_actions.v0":
        return gate_result(
            "trust_actions",
            "fail",
            "Trust actions report has an unsupported schema version.",
            {"schema_version": report.get("schema_version")},
        )
    if failed or missing:
        return gate_result(
            "trust_actions",
            "fail",
            "Trust actions report does not prove resource endorsement and dispute behavior.",
            {"failed": failed, "missing": missing},
        )
    return gate_result(
        "trust_actions",
        "pass",
        "Required resource endorsement and dispute checks passed.",
        {"checks": sorted(GO_NO_GO_REQUIRED_TRUST_ACTION_CHECKS), "resource": report.get("resource"), "run_id": report.get("run_id")},
    )


def evaluate_resource_guard_report(report: dict[str, Any]) -> dict[str, Any]:
    statuses = report_check_statuses(report)
    missing = required_pass_statuses(statuses, GO_NO_GO_REQUIRED_RESOURCE_GUARD_CHECKS)
    failed = sorted(name for name, status in statuses.items() if status == "fail")
    if report.get("schema_version") != "roborank.resource_guard.v0":
        return gate_result(
            "resource_guard",
            "fail",
            "Resource guard report has an unsupported schema version.",
            {"schema_version": report.get("schema_version")},
        )
    if failed or missing:
        return gate_result(
            "resource_guard",
            "fail",
            "Resource guard does not prove robot/environment typo rejection behavior.",
            {"failed": failed, "missing": missing},
        )
    return gate_result(
        "resource_guard",
        "pass",
        "Required resource guard checks passed.",
        {"checks": sorted(GO_NO_GO_REQUIRED_RESOURCE_GUARD_CHECKS), "known_resources": report.get("known_resources")},
    )


def evaluate_migration_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != "roborank.migration_dry_run_report.v0":
        return gate_result("migration_report", "fail", "Migration report has an unsupported schema version.", {"schema_version": report.get("schema_version")})
    privacy_checks = report.get("privacy_checks") if isinstance(report.get("privacy_checks"), list) else []
    privacy_failed = [check for check in privacy_checks if isinstance(check, dict) and check.get("status") == "fail"]
    validation_failures = report.get("validation_failures") if isinstance(report.get("validation_failures"), list) else []
    rollback = report.get("rollback_plan") if isinstance(report.get("rollback_plan"), dict) else {}
    required_rollback = {
        "d1_restore_point",
        "object_storage_restore_point",
        "worker_rollback_version",
        "frontend_rollback_version",
        "migration_pause_resume_state_path",
    }
    missing_rollback = sorted(key for key in required_rollback if not rollback.get(key))
    if privacy_failed or validation_failures or missing_rollback:
        return gate_result(
            "migration_report",
            "fail",
            "Migration report has unresolved privacy, validation, or rollback gaps.",
            {"privacy_failed": privacy_failed, "validation_failures": validation_failures, "missing_rollback": missing_rollback},
        )
    return gate_result("migration_report", "pass", "Migration report has privacy checks, no validation failures, and rollback refs.", {})


def browser_surface_statuses(report: dict[str, Any]) -> dict[str, str]:
    surfaces = report.get("surfaces")
    statuses: dict[str, str] = {}
    if isinstance(surfaces, dict):
        for name, value in surfaces.items():
            if isinstance(value, dict):
                statuses[normalized_gate_name(name)] = normalized_gate_name(value.get("status"))
            else:
                statuses[normalized_gate_name(name)] = normalized_gate_name(value)
    elif isinstance(surfaces, list):
        for surface in surfaces:
            if isinstance(surface, dict):
                name = normalized_gate_name(surface.get("surface") or surface.get("name"))
                if name:
                    statuses[name] = normalized_gate_name(surface.get("status"))
    return statuses


def evaluate_browser_report(report: dict[str, Any]) -> dict[str, Any]:
    statuses = browser_surface_statuses(report)
    missing = required_pass_statuses(statuses, GO_NO_GO_REQUIRED_BROWSER_SURFACES)
    failed = sorted(name for name, status in statuses.items() if status == "fail")
    if report.get("schema_version") not in {None, "roborank.browser_qa.v0"}:
        return gate_result("browser_report", "fail", "Browser report has an unsupported schema version.", {"schema_version": report.get("schema_version")})
    if failed or missing:
        return gate_result("browser_report", "fail", "Browser report does not prove required desktop/mobile surfaces.", {"failed": failed, "missing": missing})
    return gate_result("browser_report", "pass", "Required browser surfaces passed.", {"surfaces": sorted(GO_NO_GO_REQUIRED_BROWSER_SURFACES)})


def known_issue_items(report: Any) -> list[dict[str, Any]]:
    if isinstance(report, list):
        return [item for item in report if isinstance(item, dict)]
    if isinstance(report, dict):
        issues = report.get("issues")
        if isinstance(issues, list):
            return [item for item in issues if isinstance(item, dict)]
    return []


def issue_is_open(issue: dict[str, Any]) -> bool:
    status = normalized_gate_name(issue.get("status") or issue.get("state") or "open")
    return status not in {"closed", "resolved", "done", "deferred", "accepted"}


def evaluate_known_issues(report: dict[str, Any]) -> dict[str, Any]:
    issues = known_issue_items(report)
    blockers = []
    for issue in issues:
        severity = normalized_gate_name(issue.get("severity"))
        launch_blocker = bool(issue.get("launch_blocker") or issue.get("launchBlocker"))
        if issue_is_open(issue) and (launch_blocker or severity in {"blocker", "critical", "p0"}):
            blockers.append(issue)
    if blockers:
        return gate_result("known_issues", "fail", "Known issues include open launch blockers.", {"blockers": blockers})
    return gate_result("known_issues", "pass", "No open launch-blocking known issues.", {"issue_count": len(issues)})


def normalized_signoff_roles(report: dict[str, Any]) -> dict[str, Any]:
    raw = report.get("signoffs") if isinstance(report.get("signoffs"), dict) else report
    if not isinstance(raw, dict):
        return {}
    return {normalized_gate_name(key): value for key, value in raw.items()}


def signoff_value_approved(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return normalized_gate_name(value) in {"approved", "approve", "signed", "signed_off", "go"}
    if isinstance(value, dict):
        if value.get("approved") is True:
            return True
        for key in ("status", "decision", "signoff"):
            if normalized_gate_name(value.get(key)) in {"approved", "approve", "signed", "signed_off", "go"}:
                return True
    return False


def evaluate_signoff_report(report: dict[str, Any]) -> dict[str, Any]:
    roles = normalized_signoff_roles(report)
    missing = sorted(role for role in GO_NO_GO_REQUIRED_SIGNOFF_ROLES if not signoff_value_approved(roles.get(role)))
    if missing:
        return gate_result("signoff", "fail", "Required launch signoffs are missing.", {"missing": missing})
    return gate_result("signoff", "pass", "Required launch signoffs are present.", {"roles": sorted(GO_NO_GO_REQUIRED_SIGNOFF_ROLES)})


def evaluate_go_no_go_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != "roborank.launch_go_no_go.v0":
        return gate_result("go_no_go", "fail", "Go/no-go report has an unsupported schema version.", {"schema_version": report.get("schema_version")})
    failed = [check for check in report.get("checks", []) if isinstance(check, dict) and check.get("status") != "pass"]
    if report.get("decision") != "go" or failed:
        return gate_result("go_no_go", "fail", "Go/no-go report does not approve launch.", {"decision": report.get("decision"), "failed": failed})
    return gate_result("go_no_go", "pass", "Go/no-go report approved launch.", {})


def evaluate_watch_report(report: dict[str, Any]) -> dict[str, Any]:
    statuses = report_check_statuses(report)
    failed = sorted(name for name, status in statuses.items() if status == "fail")
    if report.get("schema_version") != "roborank.launch_watch.v0":
        return gate_result("watch_report", "fail", "Watch report has an unsupported schema version.", {"schema_version": report.get("schema_version")})
    if failed:
        return gate_result("watch_report", "fail", "Watch report has post-launch failures.", {"failed": failed})
    return gate_result("watch_report", "pass", "Post-launch watch checks passed.", {})


def evaluate_migration_progress_cutover(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != "roborank.migration_progress.v0":
        return gate_result("migration_progress", "fail", "Migration progress report has an unsupported schema version.", {"schema_version": report.get("schema_version")})
    if report.get("decision") == "pause" or report.get("pause_recommended") is True:
        return gate_result("migration_progress", "fail", "Migration progress requires pausing batches.", {"pause_reasons": report.get("pause_reasons", [])})
    return gate_result("migration_progress", "pass", "Migration progress is safe to continue or complete.", {"decision": report.get("decision")})


def evaluate_triage_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != "roborank.post_launch_triage.v0":
        return gate_result("triage_report", "fail", "Post-launch triage report has an unsupported schema version.", {"schema_version": report.get("schema_version")})
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    blockers = int(counts.get("open_blockers") or 0)
    if blockers or report.get("decision") == "investigate":
        return gate_result("triage_report", "fail", "Post-launch triage has open blockers.", {"open_blockers": blockers, "decision": report.get("decision")})
    return gate_result("triage_report", "pass", "Post-launch triage has no open blockers.", {"open_blockers": blockers})


def evaluate_cli_release_report(report: dict[str, Any]) -> dict[str, Any]:
    statuses = report_check_statuses(report)
    failed = sorted(name for name, status in statuses.items() if status == "fail")
    if report.get("schema_version") != "roborank.cli_release.v0":
        return gate_result("cli_release", "fail", "CLI release report has an unsupported schema version.", {"schema_version": report.get("schema_version")})
    if report.get("publish_ready") is not True or failed:
        return gate_result("cli_release", "fail", "CLI release artifacts are not publish-ready.", {"failed": failed, "publish_ready": report.get("publish_ready")})
    return gate_result("cli_release", "pass", "CLI release artifacts are publish-ready.", {"package": report.get("package"), "artifacts": report.get("artifacts", [])})


def evaluate_cutover_metadata(args: argparse.Namespace) -> dict[str, Any]:
    required = {
        "worker_version": args.worker_version,
        "frontend_version": args.frontend_version,
        "backend_version": args.backend_version,
        "d1_backup": args.d1_backup,
        "object_backup": args.object_backup,
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        return gate_result("cutover_metadata", "fail", "Production cutover metadata is incomplete.", {"missing": missing})
    return gate_result("cutover_metadata", "pass", "Production deployment and backup references are present.", required)


def command_launch_cutover(ctx: Context, args: argparse.Namespace) -> int:
    go_no_go_report = report_payload(args.go_no_go_report)
    production_smoke_report = report_payload(args.production_smoke_report)
    production_privacy_report = report_payload(args.production_privacy_report)
    production_schema_guard_report = report_payload(args.production_schema_guard_report)
    production_resource_guard_report = report_payload(args.production_resource_guard_report)
    backup_report = report_payload(args.backup_report)
    restore_report = report_payload(args.restore_report)
    migration_progress_report = report_payload(args.migration_progress_report)
    watch_report = report_payload(args.watch_report)
    triage_report = report_payload(args.triage_report)
    cli_release_report = report_payload(args.cli_release_report)
    checks = [
        evaluate_cutover_metadata(args),
        evaluate_go_no_go_report(go_no_go_report),
        evaluate_backup_manifest_report(backup_report),
        evaluate_restore_report(restore_report),
        evaluate_launch_smoke_report(production_smoke_report),
        evaluate_privacy_scan_report(production_privacy_report),
        evaluate_schema_guard_report(production_schema_guard_report),
        evaluate_resource_guard_report(production_resource_guard_report),
        evaluate_migration_progress_cutover(migration_progress_report),
        evaluate_watch_report(watch_report),
        evaluate_triage_report(triage_report),
        evaluate_cli_release_report(cli_release_report),
    ]
    decision = "released" if all(check.get("status") == "pass" for check in checks) else "hold"
    report = {
        "schema_version": "roborank.production_release.v0",
        "target": args.target,
        "decision": decision,
        "generated_at": now_iso(),
        "deployment": {
            "worker_version": args.worker_version,
            "frontend_version": args.frontend_version,
            "backend_version": args.backend_version,
            "roborank_envs_version": args.roborank_envs_version,
        },
        "backups": {
            "d1_backup": args.d1_backup,
            "object_backup": args.object_backup,
        },
        "checks": checks,
        "source_reports": {
            "go_no_go_report": args.go_no_go_report,
            "production_smoke_report": args.production_smoke_report,
            "production_privacy_report": args.production_privacy_report,
            "production_schema_guard_report": args.production_schema_guard_report,
            "production_resource_guard_report": args.production_resource_guard_report,
            "backup_report": args.backup_report,
            "restore_report": args.restore_report,
            "migration_progress_report": args.migration_progress_report,
            "watch_report": args.watch_report,
            "triage_report": args.triage_report,
            "cli_release_report": args.cli_release_report,
        },
    }
    if args.out:
        write_json(Path(args.out), report)
    if decision != "released":
        raise RoboRankError("Production cutover release gate is on hold.", code="production_release_hold", exit_code=9, details=report)
    return emit_success(ctx, "launch.cutover", report)


def command_launch_go_no_go(ctx: Context, args: argparse.Namespace) -> int:
    preflight_report = report_payload(args.preflight_report)
    smoke_report = report_payload(args.smoke_report)
    restore_report = report_payload(args.restore_report)
    evidence_examples_report = report_payload(args.evidence_examples_report)
    privacy_report = report_payload(args.privacy_report)
    schema_guard_report = report_payload(args.schema_guard_report)
    trust_actions_report = report_payload(args.trust_actions_report)
    resource_guard_report = report_payload(args.resource_guard_report)
    migration_report = report_payload(args.migration_report)
    browser_report = report_payload(args.browser_report)
    known_issues = report_payload(args.known_issues)
    signoff = report_payload(args.signoff)
    checks = [
        evaluate_preflight_report(preflight_report),
        evaluate_launch_smoke_report(smoke_report),
        evaluate_restore_report(restore_report),
        evaluate_evidence_examples_report(evidence_examples_report),
        evaluate_privacy_scan_report(privacy_report),
        evaluate_schema_guard_report(schema_guard_report),
        evaluate_trust_actions_report(trust_actions_report),
        evaluate_resource_guard_report(resource_guard_report),
        evaluate_migration_report(migration_report),
        evaluate_browser_report(browser_report),
        evaluate_known_issues(known_issues),
        evaluate_signoff_report(signoff),
    ]
    decision = "go" if all(check["status"] == "pass" for check in checks) else "no-go"
    report = {
        "schema_version": "roborank.launch_go_no_go.v0",
        "target": args.target,
        "decision": decision,
        "completed_at": now_iso(),
        "checks": checks,
        "source_reports": {
            "preflight_report": args.preflight_report,
            "smoke_report": args.smoke_report,
            "restore_report": args.restore_report,
            "evidence_examples_report": args.evidence_examples_report,
            "privacy_report": args.privacy_report,
            "schema_guard_report": args.schema_guard_report,
            "trust_actions_report": args.trust_actions_report,
            "resource_guard_report": args.resource_guard_report,
            "migration_report": args.migration_report,
            "browser_report": args.browser_report,
            "known_issues": args.known_issues,
            "signoff": args.signoff,
        },
    }
    if args.out:
        write_json(Path(args.out), report)
    if decision != "go":
        raise RoboRankError("Launch go/no-go decision is no-go.", code="launch_no_go", exit_code=9, details=report)
    return emit_success(ctx, "launch.go-no-go", report)


def first_rerun_artifact_url(payload: dict[str, Any]) -> str | None:
    replay = payload.get("replay") if isinstance(payload.get("replay"), dict) else payload.get("trace")
    if not isinstance(replay, dict):
        return None
    artifacts = replay.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    for artifact in artifacts:
        if isinstance(artifact, dict) and artifact.get("type") == "rerun_rrd" and isinstance(artifact.get("url"), str):
            return str(artifact["url"])
    return None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_local_challenge(challenge_id: str, policy_path: Path, max_steps: int | None, artifact_dir: Path) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        from roborank_envs.simulation import rerun_export

        previous_artifact_dir = rerun_export.RUN_ARTIFACT_DIR
        rerun_export.RUN_ARTIFACT_DIR = artifact_dir
        try:
            result = run_policy_file(challenge_id=challenge_id, policy_path=policy_path, max_steps=max_steps)
        finally:
            rerun_export.RUN_ARTIFACT_DIR = previous_artifact_dir
    except EnvironmentRunError as exc:
        raise RoboRankError(
            "Local challenge evaluation failed.",
            code="local_eval_failed",
            exit_code=8,
            details={"challenge_id": challenge_id, "detail": str(exc)},
        ) from exc
    except Exception as exc:  # noqa: BLE001 - local CLI boundary returns structured failures.
        raise RoboRankError(
            "Local challenge evaluation failed.",
            code="local_eval_failed",
            exit_code=8,
            details={"challenge_id": challenge_id, "type": type(exc).__name__, "detail": str(getattr(exc, "detail", None) or exc)},
        ) from exc
    payload = result.model_dump(mode="json")
    payload["_local_artifact_dir"] = str(artifact_dir)
    return payload


def copy_local_rerun_artifact(result_payload: dict[str, Any], out_dir: Path) -> Path:
    artifact_url = first_rerun_artifact_url(result_payload)
    if not artifact_url:
        replay = result_payload.get("replay") if isinstance(result_payload.get("replay"), dict) else {}
        metadata = replay.get("metadata") if isinstance(replay.get("metadata"), dict) else {}
        raise RoboRankError(
            "Local challenge run did not produce a Rerun artifact.",
            code="rerun_artifact_missing",
            exit_code=6,
            details={"rerun_export_error": metadata.get("rerun_export_error")},
            suggested_commands=["Install the visualization extra before creating upload-ready local eval bundles."],
        )
    artifact_path = urllib.parse.urlparse(artifact_url).path
    filename = Path(artifact_path).name
    if not filename.endswith(".rrd"):
        raise RoboRankError(
            "Local challenge run produced an unsupported artifact path.",
            code="rerun_artifact_invalid",
            exit_code=6,
            details={"artifact_url": artifact_url},
        )
    artifact_dir = Path(str(result_payload.get("_local_artifact_dir") or (Path.cwd() / "runs")))
    source = artifact_dir / filename
    if not source.exists():
        replay = result_payload.get("replay") if isinstance(result_payload.get("replay"), dict) else {}
        metadata = replay.get("metadata") if isinstance(replay.get("metadata"), dict) else {}
        raise RoboRankError(
            "Local Rerun artifact metadata exists, but the artifact bytes are missing.",
            code="rerun_artifact_missing",
            exit_code=6,
            details={"artifact_url": artifact_url, "expected_path": str(source), "rerun_export_error": metadata.get("rerun_export_error")},
        )
    target = out_dir / "recording.rrd"
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
        if source.parent.resolve() == out_dir.resolve():
            source.unlink()
    return target


def write_local_eval_bundle(ctx: Context, args: argparse.Namespace, policy_path: Path, policy_source: str) -> dict[str, Any]:
    challenge_id = args.challenge_id
    out_dir = Path(args.out or (Path("runs") / f"{canonical_slug(challenge_id)}-{uuid.uuid4().hex[:8]}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    result_payload = run_local_challenge(challenge_id, policy_path, args.max_steps, out_dir)
    recording_path = copy_local_rerun_artifact(result_payload, out_dir)
    metrics = result_payload.get("metrics") if isinstance(result_payload.get("metrics"), dict) else {}
    logs = result_payload.get("logs") if isinstance(result_payload.get("logs"), list) else []
    result_status = normalize_cli_result_status(result_payload)
    robot, environment = CHALLENGE_RESOURCE_CACHE.get(challenge_id, ("std/unknown", "std/unknown"))
    if (robot, environment) == ("std/unknown", "std/unknown"):
        ctx.warn("challenge_resource_mapping_missing", "Challenge resource mapping is missing; evidence bundle uses std/unknown.")

    policy_hash = hashlib.sha256(policy_source.encode("utf-8")).hexdigest()
    policy = args.policy or f"local/{canonical_slug(challenge_id)}-{policy_hash[:8]}"
    policy_family = args.policy_family or f"local/{canonical_slug(challenge_id)}"
    if not args.policy:
        ctx.warn("local_policy_id", "Local eval used a generated local policy ID; pass --policy for upload-ready ownership.")

    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "run.log").write_text("\n".join(str(item) for item in logs) + ("\n" if logs else ""))
    visibility = validate_evidence_visibility(args.visibility)
    evidence = {
        "schema_version": "roborank.rerun_evidence.v0",
        "title": f"Local {challenge_id} run",
        "summary": "Evidence generated by local RoboRank challenge evaluation.",
        "robot": robot,
        "environment": environment,
        "policy": policy,
        "policy_family": policy_family,
        "recording_path": recording_path.name,
        "metrics_path": "metrics.json",
        "run_mode": "simulation",
        "result_status": result_status,
        "license": args.license,
        "visibility": visibility,
    }
    write_json(out_dir / "evidence.json", evidence)
    score = metrics.get("score") if isinstance(metrics, dict) else None
    summary = {
        "challenge_id": challenge_id,
        "robot": robot,
        "environment": environment,
        "policy": policy,
        "policy_family": policy_family,
        "result_status": result_status,
        "score": score,
        "bundle_dir": str(out_dir),
        "metrics_path": "metrics.json",
        "recording_path": recording_path.name,
        "evidence_envelope_path": "evidence.json",
        "run_log_path": "run.log",
    }
    write_json(out_dir / "result.json", summary)
    return summary


def normalize_cli_result_status(payload: dict[str, Any]) -> str:
    status = str(payload.get("status") or "")
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    metric_status = str(metrics.get("status") or "")
    success = metrics.get("success")
    if status in {"success", "passed"} or success is True:
        return "success"
    if status in {"timeout"} or metric_status == "timeout":
        return "timeout"
    if status in {"aborted"} or metric_status == "aborted":
        return "aborted"
    if status or metric_status or success is False:
        return "failure"
    return "unknown"


def canonical_slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "unknown"


def command_doctor(ctx: Context, args: argparse.Namespace) -> int:
    del args
    checks = {
        "api_url": ctx.api_url,
        "token_configured": bool(ctx.options.token),
        "max_rrd_bytes": MAX_RRD_BYTES,
    }
    return emit_success(ctx, "doctor", checks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roborank")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prime = subparsers.add_parser("prime")
    prime.add_argument("--agent", action="store_true")
    prime.add_argument("--task", choices=["upload", "metrics", "eval"], default=None)
    prime.add_argument("--environment")
    prime.add_argument("--challenge")
    prime.set_defaults(handler=command_prime)

    auth = subparsers.add_parser("auth")
    auth_sub = auth.add_subparsers(dest="auth_command", required=True)
    auth_sub.add_parser("status").set_defaults(handler=command_auth)
    auth_login = auth_sub.add_parser("login")
    auth_login.add_argument("--no-browser", action="store_true")
    auth_login.set_defaults(handler=command_auth)
    auth_sub.add_parser("logout").set_defaults(handler=command_auth)
    auth_token = auth_sub.add_parser("token")
    auth_token_sub = auth_token.add_subparsers(dest="token_command", required=True)
    token_create = auth_token_sub.add_parser("create")
    token_create.add_argument("--scope", action="append", default=[])
    token_create.add_argument("--name", default="CLI token")
    token_create.set_defaults(auth_command="token_create", handler=command_auth)

    resources = subparsers.add_parser("resources")
    resources_sub = resources.add_subparsers(dest="resources_command", required=True)
    for name in ("search", "list"):
        cmd = resources_sub.add_parser(name)
        cmd.add_argument("--query")
        cmd.add_argument("--kind", choices=sorted(RESOURCE_KINDS))
        cmd.add_argument("--namespace")
        cmd.add_argument("--owner")
        cmd.add_argument("--limit", type=int, default=25)
        cmd.set_defaults(handler=command_resources)
    for name in ("show", "resolve"):
        cmd = resources_sub.add_parser(name)
        cmd.add_argument("kind", choices=sorted(RESOURCE_KINDS))
        cmd.add_argument("resource_id")
        cmd.set_defaults(handler=command_resources)
    for name in ("create", "update"):
        cmd = resources_sub.add_parser(name)
        cmd.add_argument("kind", choices=sorted(RESOURCE_KINDS))
        cmd.add_argument("resource_id")
        if name == "update":
            cmd.add_argument("--new-id")
        cmd.add_argument("--title", required=name == "create")
        cmd.add_argument("--summary")
        cmd.add_argument("--markdown")
        cmd.set_defaults(handler=command_resources)

    metrics = subparsers.add_parser("metrics")
    metrics_sub = metrics.add_subparsers(dest="metrics_command", required=True)
    schema = metrics_sub.add_parser("schema")
    schema.add_argument("--environment", required=True)
    schema.add_argument("--out")
    schema.set_defaults(handler=command_metrics)
    init = metrics_sub.add_parser("init")
    init.add_argument("--environment", required=True)
    init.add_argument("--out", required=True)
    init.add_argument("--instructions")
    init.set_defaults(handler=command_metrics)
    setup = metrics_sub.add_parser("setup")
    setup.add_argument("--environment", required=True)
    setup.add_argument("--from", dest="from_path")
    setup.add_argument("--set", dest="set_values", action="append", default=[])
    setup.add_argument("--out", required=True)
    setup.set_defaults(handler=command_metrics)
    for name in ("validate", "explain"):
        cmd = metrics_sub.add_parser(name)
        cmd.add_argument("--environment", required=True)
        cmd.add_argument("--schema-hash")
        cmd.add_argument("metrics_path", nargs="?")
        cmd.set_defaults(handler=command_metrics)

    evidence = subparsers.add_parser("evidence")
    evidence_sub = evidence.add_subparsers(dest="evidence_command", required=True)
    evidence_init = evidence_sub.add_parser("init")
    add_evidence_metadata_args(evidence_init)
    evidence_init.add_argument("--out", required=True)
    evidence_init.set_defaults(handler=command_evidence)
    for name in ("validate", "upload"):
        cmd = evidence_sub.add_parser(name)
        add_evidence_upload_args(cmd)
        cmd.set_defaults(handler=command_evidence)
    evidence_show = evidence_sub.add_parser("show")
    evidence_show.add_argument("run_id", nargs="?")
    evidence_show.add_argument("--client-upload-id")
    evidence_show.set_defaults(handler=command_evidence)

    eval_parser = subparsers.add_parser("eval")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_sub.add_parser("list").set_defaults(handler=command_eval)
    eval_show = eval_sub.add_parser("show")
    eval_show.add_argument("challenge_id")
    eval_show.set_defaults(handler=command_eval)
    eval_run = eval_sub.add_parser("run")
    eval_run.add_argument("challenge_id")
    eval_run.add_argument("--policy-source", required=True)
    eval_run.add_argument("--out")
    eval_run.add_argument("--policy")
    eval_run.add_argument("--policy-family", dest="policy_family")
    eval_run.add_argument("--license", default=DEFAULT_LICENSE)
    eval_run.add_argument("--visibility", default="public", choices=sorted(EVIDENCE_VISIBILITIES))
    eval_run.add_argument("--max-steps", type=int)
    eval_run.add_argument("--require-result")
    eval_run.set_defaults(handler=command_eval)
    eval_submit = eval_sub.add_parser("submit")
    eval_submit.add_argument("challenge_id")
    eval_submit.add_argument("--policy-source", required=True)
    eval_submit.add_argument("--require-result")
    eval_submit.add_argument("--evidence", action="store_true")
    eval_submit.set_defaults(handler=command_eval)

    migration = subparsers.add_parser("migration")
    migration_sub = migration.add_subparsers(dest="migration_command", required=True)
    inventory = migration_sub.add_parser("inventory")
    inventory.add_argument("--challenge")
    inventory.add_argument("--limit", type=int, default=100)
    inventory.add_argument("--offset", type=int, default=0)
    inventory.add_argument("--out")
    inventory.set_defaults(handler=command_migration)
    migration_map = migration_sub.add_parser("map")
    migration_map.add_argument("--from", dest="from_path")
    migration_map.add_argument("--challenge")
    migration_map.add_argument("--limit", type=int, default=100)
    migration_map.add_argument("--offset", type=int, default=0)
    migration_map.add_argument("--out")
    migration_map.set_defaults(handler=command_migration)
    for name in ("rerun", "upload"):
        cmd = migration_sub.add_parser(name)
        cmd.add_argument("--submission-id")
        cmd.add_argument("--challenge")
        cmd.add_argument("--all", action="store_true")
        cmd.add_argument("--from", dest="from_path")
        cmd.add_argument("--out")
        cmd.add_argument("--resume", action="store_true")
        cmd.add_argument("--state")
        cmd.add_argument("--limit", type=int, default=100)
        cmd.add_argument("--offset", type=int, default=0)
        cmd.set_defaults(handler=command_migration)
    recover = migration_sub.add_parser("recover")
    recover.add_argument("--submission-id", required=True)
    recover.add_argument("--artifact", required=True)
    recover.add_argument("--out")
    recover.set_defaults(handler=command_migration)
    verify = migration_sub.add_parser("verify")
    verify.add_argument("--from", dest="from_path", required=True)
    verify.set_defaults(handler=command_migration)
    migration_report = migration_sub.add_parser("report")
    migration_report.add_argument("--inventory", required=True)
    migration_report.add_argument("--mapping", required=True)
    migration_report.add_argument("--rerun-state")
    migration_report.add_argument("--verify-report", action="append", default=[])
    migration_report.add_argument("--upload-report", action="append", default=[])
    migration_report.add_argument("--restore-report")
    migration_report.add_argument("--smoke-report")
    migration_report.add_argument("--environment", default="staging")
    migration_report.add_argument("--d1-export")
    migration_report.add_argument("--artifact-inventory")
    migration_report.add_argument("--worker-version")
    migration_report.add_argument("--frontend-version")
    migration_report.add_argument("--backend-version")
    migration_report.add_argument("--roborank-envs-version")
    migration_report.add_argument("--started-at")
    migration_report.add_argument("--completed-at")
    migration_report.add_argument("--operator")
    migration_report.add_argument("--rollback-d1")
    migration_report.add_argument("--rollback-objects")
    migration_report.add_argument("--rollback-worker")
    migration_report.add_argument("--rollback-frontend")
    migration_report.add_argument("--strict", action="store_true")
    migration_report.add_argument("--out")
    migration_report.set_defaults(handler=command_migration)
    migration_progress = migration_sub.add_parser("progress")
    migration_progress.add_argument("--mapping", required=True)
    migration_progress.add_argument("--rerun-state")
    migration_progress.add_argument("--verify-report", action="append", default=[])
    migration_progress.add_argument("--upload-report", action="append", default=[])
    migration_progress.add_argument("--environment", default="production")
    migration_progress.add_argument("--batch-size", type=int, default=100)
    migration_progress.add_argument("--max-failures", type=int, default=0)
    migration_progress.add_argument("--max-failure-rate", type=float, default=0.0)
    migration_progress.add_argument("--repeated-failure-threshold", type=int, default=3)
    migration_progress.add_argument("--out")
    migration_progress.set_defaults(handler=command_migration)

    launch = subparsers.add_parser("launch")
    launch_sub = launch.add_subparsers(dest="launch_command", required=True)
    smoke = launch_sub.add_parser("smoke")
    smoke.add_argument("--target", default="current")
    smoke.add_argument("--strict", action="store_true")
    smoke.add_argument("--run-id")
    smoke.add_argument("--artifact-url")
    smoke.add_argument("--resource-kind", choices=sorted(RESOURCE_KINDS))
    smoke.add_argument("--resource-id")
    smoke.add_argument("--environment")
    smoke.add_argument("--legacy-submission-id")
    smoke.add_argument("--anonymous-limit", type=int, default=2)
    smoke.add_argument("--signed-in-limit", type=int, default=3)
    smoke.add_argument("--compare-run", action="append", default=[])
    smoke.add_argument("--include-mutating", action="store_true")
    smoke.add_argument("--challenge-id", default="diff_drive_reach_target")
    smoke.add_argument("--policy-source")
    smoke.add_argument("--direct-rrd")
    smoke.add_argument("--metrics")
    smoke.add_argument("--robot")
    smoke.add_argument("--policy")
    smoke.add_argument("--policy-family", dest="policy_family")
    smoke.add_argument("--license", default=DEFAULT_LICENSE)
    smoke.add_argument("--visibility", default="public", choices=sorted(EVIDENCE_VISIBILITIES))
    smoke.add_argument("--client-upload-id")
    smoke.add_argument("--source-link", action="append", default=[])
    smoke.add_argument("--allow-new-policy", action="store_true")
    smoke.add_argument("--allow-new-policy-family", action="store_true")
    smoke.add_argument("--out")
    smoke.set_defaults(handler=command_launch)
    evidence_examples = launch_sub.add_parser("evidence-examples")
    evidence_examples.add_argument("--target", default="staging")
    evidence_examples.add_argument("--example", action="append", default=[])
    evidence_examples.add_argument("--out")
    evidence_examples.set_defaults(handler=command_launch)
    privacy_scan = launch_sub.add_parser("privacy-scan")
    privacy_scan.add_argument("--target", default="staging")
    privacy_scan.add_argument("--strict", action="store_true")
    privacy_scan.add_argument("--explorer-limit", type=int, default=3)
    privacy_scan.add_argument("--run-id", action="append", default=[])
    privacy_scan.add_argument("--resource-kind", choices=sorted(RESOURCE_KINDS))
    privacy_scan.add_argument("--resource-id")
    privacy_scan.add_argument("--compare-run", action="append", default=[])
    privacy_scan.add_argument("--out")
    privacy_scan.set_defaults(handler=command_launch)
    schema_guard = launch_sub.add_parser("schema-guard")
    schema_guard.add_argument("--target", default="staging")
    schema_guard.add_argument("--environment", default="roborank/diff-drive-reach-target")
    schema_guard.add_argument("--robot", default="roborank/differential-drive-cube-v1")
    schema_guard.add_argument("--policy", required=True)
    schema_guard.add_argument("--policy-family", dest="policy_family")
    schema_guard.add_argument("--allow-new-policy", action="store_true")
    schema_guard.add_argument("--allow-new-policy-family", action="store_true")
    schema_guard.add_argument("--invalid-metrics")
    schema_guard.add_argument("--license", default=DEFAULT_LICENSE)
    schema_guard.add_argument("--out")
    schema_guard.set_defaults(handler=command_launch)
    trust_actions = launch_sub.add_parser("trust-actions")
    trust_actions.add_argument("--target", default="staging")
    trust_actions.add_argument("--resource-kind", choices=sorted(RESOURCE_KINDS), required=True)
    trust_actions.add_argument("--resource-id", required=True)
    trust_actions.add_argument("--run-id", required=True)
    trust_actions.add_argument("--note", default="Launch trust-action endorsement probe.")
    trust_actions.add_argument("--reason", default="Launch trust-action dispute probe.")
    trust_actions.add_argument("--out")
    trust_actions.set_defaults(handler=command_launch)
    resource_guard = launch_sub.add_parser("resource-guard")
    resource_guard.add_argument("--target", default="staging")
    resource_guard.add_argument("--robot", default="roborank/differential-drive-cube-v1")
    resource_guard.add_argument("--environment", default="std/unknown")
    resource_guard.add_argument("--policy", required=True)
    resource_guard.add_argument("--policy-family", dest="policy_family")
    resource_guard.add_argument("--allow-new-policy", action="store_true")
    resource_guard.add_argument("--allow-new-policy-family", action="store_true")
    resource_guard.add_argument("--missing-robot", default="qa/missing-robot-typo")
    resource_guard.add_argument("--missing-environment", default="qa/missing-environment-typo")
    resource_guard.add_argument("--license", default=DEFAULT_LICENSE)
    resource_guard.add_argument("--out")
    resource_guard.set_defaults(handler=command_launch)
    browser_qa = launch_sub.add_parser("browser-qa")
    browser_qa.add_argument("--target", default="staging")
    browser_qa.add_argument("--surface", action="append", default=[])
    browser_qa.add_argument("--out")
    browser_qa.set_defaults(handler=command_launch)
    known_issues = launch_sub.add_parser("known-issues")
    known_issues.add_argument("--target", default="staging")
    known_issues.add_argument("--from", dest="from_path")
    known_issues.add_argument("--issue", action="append", default=[])
    known_issues.add_argument("--out")
    known_issues.set_defaults(handler=command_launch)
    triage = launch_sub.add_parser("triage")
    triage.add_argument("--target", default="production")
    triage.add_argument("--known-issues", action="append", default=[])
    triage.add_argument("--watch-report")
    triage.add_argument("--migration-progress-report")
    triage.add_argument("--smoke-report")
    triage.add_argument("--go-no-go-report")
    triage.add_argument("--issue", action="append", default=[])
    triage.add_argument("--allow-open-blockers", action="store_true")
    triage.add_argument("--out")
    triage.set_defaults(handler=command_launch)
    signoff = launch_sub.add_parser("signoff")
    signoff.add_argument("--target", default="staging")
    signoff.add_argument("--from", dest="from_path")
    signoff.add_argument("--data-storage", dest="data_storage")
    signoff.add_argument("--backend")
    signoff.add_argument("--frontend")
    signoff.add_argument("--product")
    signoff.add_argument("--out")
    signoff.set_defaults(handler=command_launch)
    cli_release = launch_sub.add_parser("cli-release")
    cli_release.add_argument("--target", default="production")
    cli_release.add_argument("--cli-root", default=str(repo_root()))
    cli_release.add_argument("--dist-dir", default=str(repo_root() / "dist" / "cli-release"))
    cli_release.add_argument("--timeout", type=int, default=300)
    cli_release.add_argument("--clean", action="store_true")
    cli_release.add_argument("--out")
    cli_release.set_defaults(handler=command_launch)
    cutover = launch_sub.add_parser("cutover")
    cutover.add_argument("--target", default="production")
    cutover.add_argument("--go-no-go-report", required=True)
    cutover.add_argument("--production-smoke-report", required=True)
    cutover.add_argument("--production-privacy-report", required=True)
    cutover.add_argument("--production-schema-guard-report", required=True)
    cutover.add_argument("--production-resource-guard-report", required=True)
    cutover.add_argument("--backup-report", required=True)
    cutover.add_argument("--restore-report", required=True)
    cutover.add_argument("--migration-progress-report", required=True)
    cutover.add_argument("--watch-report", required=True)
    cutover.add_argument("--triage-report", required=True)
    cutover.add_argument("--cli-release-report", required=True)
    cutover.add_argument("--worker-version", required=True)
    cutover.add_argument("--frontend-version", required=True)
    cutover.add_argument("--backend-version", required=True)
    cutover.add_argument("--roborank-envs-version")
    cutover.add_argument("--d1-backup", required=True)
    cutover.add_argument("--object-backup", required=True)
    cutover.add_argument("--out")
    cutover.set_defaults(handler=command_launch)
    preflight = launch_sub.add_parser("preflight")
    preflight.add_argument("--target", default="staging")
    preflight.add_argument("--repo-root", default=str(repo_root()))
    preflight.add_argument("--check", choices=sorted(LAUNCH_PREFLIGHT_COMMANDS), action="append", default=[])
    preflight.add_argument("--skip", choices=sorted(LAUNCH_PREFLIGHT_COMMANDS), action="append", default=[])
    preflight.add_argument("--timeout", type=int, default=900)
    preflight.add_argument("--out")
    preflight.set_defaults(handler=command_launch)
    watch = launch_sub.add_parser("watch")
    watch.add_argument("--target", default="production")
    watch.add_argument("--max-upload-failures", type=int, default=0)
    watch.add_argument("--max-metrics-validation-failures", type=int, default=0)
    watch.add_argument("--max-storage-failures", type=int, default=0)
    watch.add_argument("--max-viewer-failures", type=int, default=0)
    watch.add_argument("--max-explorer-slow-events", type=int, default=0)
    watch.add_argument("--max-migration-failures", type=int, default=0)
    watch.add_argument("--max-rate-limit-events", type=int, default=0)
    watch.add_argument("--fail-on-severity", action="append", default=["error", "critical"])
    watch.add_argument("--min-evidence-runs", type=int)
    watch.add_argument("--out")
    watch.set_defaults(handler=command_launch)
    restore_check = launch_sub.add_parser("restore-check")
    restore_check.add_argument("--target", default="staging")
    restore_check.add_argument("--strict", action="store_true")
    restore_check.add_argument("--d1-sql")
    restore_check.add_argument("--object-manifest")
    restore_check.add_argument("--object-root")
    restore_check.add_argument("--api-smoke-report")
    restore_check.add_argument("--required-table", action="append", default=[])
    restore_check.add_argument("--required-object", action="append", default=[])
    restore_check.set_defaults(handler=command_launch)
    backup_manifest = launch_sub.add_parser("backup-manifest")
    backup_manifest.add_argument("--target", default="production")
    backup_manifest.add_argument("--label")
    backup_manifest.add_argument("--d1-sql", required=True)
    backup_manifest.add_argument("--object-manifest", required=True)
    backup_manifest.add_argument("--object-root", required=True)
    backup_manifest.add_argument("--required-table", action="append", default=[])
    backup_manifest.add_argument("--required-object", action="append", default=[])
    backup_manifest.add_argument("--out")
    backup_manifest.set_defaults(handler=command_launch)
    go_no_go = launch_sub.add_parser("go-no-go")
    go_no_go.add_argument("--target", default="staging")
    go_no_go.add_argument("--preflight-report", required=True)
    go_no_go.add_argument("--smoke-report", required=True)
    go_no_go.add_argument("--restore-report", required=True)
    go_no_go.add_argument("--evidence-examples-report", required=True)
    go_no_go.add_argument("--privacy-report", required=True)
    go_no_go.add_argument("--schema-guard-report", required=True)
    go_no_go.add_argument("--trust-actions-report", required=True)
    go_no_go.add_argument("--resource-guard-report", required=True)
    go_no_go.add_argument("--migration-report", required=True)
    go_no_go.add_argument("--browser-report", required=True)
    go_no_go.add_argument("--known-issues", required=True)
    go_no_go.add_argument("--signoff", required=True)
    go_no_go.add_argument("--out")
    go_no_go.set_defaults(handler=command_launch)

    doctor = subparsers.add_parser("doctor")
    doctor.set_defaults(handler=command_doctor)
    return parser


def add_evidence_metadata_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--title")
    parser.add_argument("--summary")
    parser.add_argument("--notes")
    parser.add_argument("--superseded-by-run-id", dest="superseded_by_run_id")
    parser.add_argument("--robot")
    parser.add_argument("--environment")
    parser.add_argument("--policy")
    parser.add_argument("--policy-family", dest="policy_family")
    parser.add_argument("--run-mode")
    parser.add_argument("--result-status")
    parser.add_argument("--license")
    parser.add_argument("--visibility", choices=sorted(EVIDENCE_VISIBILITIES))


def add_evidence_upload_args(parser: argparse.ArgumentParser) -> None:
    add_evidence_metadata_args(parser)
    parser.add_argument("--from", dest="from_path")
    parser.add_argument("--rrd")
    parser.add_argument("--metrics")
    parser.add_argument("--source-link", action="append", default=[])
    parser.add_argument("--client-upload-id")
    parser.add_argument("--allow-new-policy", action="store_true")
    parser.add_argument("--allow-new-policy-family", action="store_true")
    parser.add_argument("--allow-new-robot", action="store_true")
    parser.add_argument("--allow-new-environment", action="store_true")
    parser.add_argument("--policy-source")


def command_name(args: argparse.Namespace) -> str:
    parts = [getattr(args, "command", "unknown")]
    for attr in (
        "auth_command",
        "resources_command",
        "metrics_command",
        "evidence_command",
        "eval_command",
        "migration_command",
        "launch_command",
    ):
        value = getattr(args, attr, None)
        if value:
            parts.append(str(value))
            break
    return ".".join(parts)


def main(argv: list[str] | None = None, stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    command = "unknown"
    try:
        options, filtered = extract_global_options(argv)
        ctx = Context(options=options, stdout=stdout, stderr=stderr)
        parser = build_parser()
        args = parser.parse_args(filtered)
        command = command_name(args)
        handler: Callable[[Context, argparse.Namespace], int] = args.handler
        return handler(ctx, args)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    except RoboRankError as exc:
        ctx = Context(options=options if "options" in locals() else GlobalOptions(api_url=DEFAULT_API_URL), stdout=stdout, stderr=stderr)
        return emit_error(ctx, command, exc)
    except Exception as exc:  # noqa: BLE001 - CLI boundary must return structured failures.
        ctx = Context(options=options if "options" in locals() else GlobalOptions(api_url=DEFAULT_API_URL), stdout=stdout, stderr=stderr)
        return emit_error(ctx, command, RoboRankError(str(exc), code="internal_error", exit_code=1))


if __name__ == "__main__":
    raise SystemExit(main())
