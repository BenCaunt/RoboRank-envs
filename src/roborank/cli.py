from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

from roborank_envs.catalog import get_challenge, list_challenges
from roborank_envs.runner import EnvironmentRunError, run_policy_file


MAX_RRD_BYTES = 40 * 1024 * 1024
MAX_METRICS_BYTES = 1024 * 1024
DEFAULT_API_URL = "https://roborank.dev"
DEFAULT_LICENSE = "CC-BY-4.0"
API_USER_AGENT = "RoboRank CLI/0.1.0 (+https://roborank.dev)"
AUTH_FILE_RELATIVE_PATH = Path(".roborank") / "auth.json"
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
    token_source: str | None = None
    auth_path: str | None = None
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
    stdin: TextIO
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


def auth_path(base_dir: Path | None = None) -> Path:
    configured = os.environ.get("ROBORANK_AUTH")
    if configured:
        return Path(configured).expanduser()
    return (base_dir or Path.cwd()) / AUTH_FILE_RELATIVE_PATH


def load_auth_data(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RoboRankError(f"{path} is not valid JSON: {exc}", code="invalid_auth_file", exit_code=2) from exc
    if not isinstance(data, dict):
        raise RoboRankError(f"{path} must contain a JSON object.", code="invalid_auth_file", exit_code=2)
    return data


def load_auth_file(base_dir: Path | None = None) -> dict[str, Any]:
    return load_auth_data(auth_path(base_dir))


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
    raw_auth = load_auth_file()
    auth_config = profile_config(raw_auth, profile)

    output_format = values.get("--format") or os.environ.get("ROBORANK_FORMAT") or config_string(config, "format", "output_format") or "human"
    if "--json" in bools:
        output_format = "json"
    if output_format == "markdown":
        output_format = "human"
    if output_format not in {"human", "json", "yaml"}:
        raise RoboRankError("--format must be human, json, or yaml.", code="usage_error", exit_code=2)

    auth_api_url = config_string(auth_config, "api_url", "api-url")
    api_url = (
        values.get("--api-url")
        or os.environ.get("ROBORANK_API_URL")
        or config_string(config, "api_url", "api-url")
        or auth_api_url
        or DEFAULT_API_URL
    )
    flag_token = values.get("--token")
    env_token = os.environ.get("ROBORANK_TOKEN")
    config_token = config_string(config, "token")
    auth_token = config_string(auth_config, "token", "access_token")
    token = flag_token or env_token or config_token
    token_source = None
    if flag_token:
        token_source = "flag"
    elif env_token:
        token_source = "env"
    elif config_token:
        token_source = "config"
    elif auth_token and (not auth_api_url or auth_api_url.rstrip("/") == api_url.rstrip("/")):
        token = auth_token
        token_source = "auth_file"

    return (
        GlobalOptions(
            api_url=api_url,
            token=token,
            token_source=token_source,
            auth_path=str(auth_path()),
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


def write_auth_file(path: Path, *, profile: str, api_url: str, token: str) -> None:
    data = load_auth_data(path)
    if not isinstance(data, dict):
        data = {}
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        profiles = {}
    profiles[profile] = {
        "api_url": api_url.rstrip("/"),
        "token": token,
    }
    data["version"] = 1
    data["profiles"] = profiles
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def remove_auth_profile(path: Path, *, profile: str) -> bool:
    if not path.exists():
        return False
    data = load_auth_data(path)
    if not isinstance(data, dict):
        return False
    removed = False
    profiles = data.get("profiles")
    if isinstance(profiles, dict) and profile in profiles:
        profiles.pop(profile)
        removed = True
    if profile == "default" and "token" in data:
        data.pop("token", None)
        data.pop("api_url", None)
        removed = True
    if not removed:
        return False
    if isinstance(profiles, dict) and profiles:
        data["profiles"] = profiles
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    else:
        path.unlink()
    return True


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


def build_agent_primer(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "primer_version": "roborank-agent-primer-v0",
        "index_version": "roborank-procedures-index-v1",
        "purpose": (
            "Use prime as the starting index for RoboRank automation. It names the stable concepts, "
            "the boundaries of each concept, and the standard command procedures to discover the "
            "remaining details from the API instead of guessing."
        ),
        "model": {
            "primary_artifact": "recording.rrd",
            "required_tags": ["robot", "environment", "policy"],
            "optional_tags": ["policy_family"],
            "resource_id_grammar": "<namespace>/<slug>",
            "max_rrd_bytes": MAX_RRD_BYTES,
            "default_license": DEFAULT_LICENSE,
            "allowed_licenses": sorted(ALLOWED_LICENSES),
        },
        "terms": [
            {
                "term": "resource",
                "definition": (
                    "A canonical RoboRank object that evidence can reference: robot, environment, "
                    "policy, or policy_family."
                ),
                "boundary": (
                    "Resources carry identity, ownership, summary metadata, aliases, and README "
                    "markdown. They are not evidence artifacts, private policy source, or eval results."
                ),
            },
            {
                "term": "resource_id",
                "definition": "The stable external identifier for a resource, formatted as <namespace>/<slug>.",
                "boundary": "Do not fabricate IDs. Search, read, or resolve them through the API.",
            },
            {
                "term": "README markdown",
                "definition": "The resource page body stored as markdown and returned in the resource payload.",
                "boundary": "Use it for public documentation and provenance, not for secrets or private source.",
            },
            {
                "term": "eval",
                "definition": "A RoboRank challenge execution against a policy, either local or hosted.",
                "boundary": (
                    "Evals execute packaged RoboRank challenges. They are not general robotics jobs, "
                    "ad hoc simulations, or resource records."
                ),
            },
            {
                "term": "evidence",
                "definition": "A submitted run artifact, normally one Rerun .rrd plus optional metrics.json.",
                "boundary": (
                    "Evidence links resources to an observed run. Public evidence upload does not upload "
                    "policy source code by default."
                ),
            },
            {
                "term": "metrics",
                "definition": "Structured JSON values validated against an environment metrics schema when one exists.",
                "boundary": "Metrics summarize a run; they do not replace the Rerun recording or resource tags.",
            },
        ],
        "standard_procedures": [
            {
                "name": "Authenticate",
                "when": "Before reads that require identity or any resource/evidence write.",
                "steps": [
                    "Check the active API URL and token source.",
                    "Create or configure a token with the smallest required scopes.",
                ],
                "commands": [
                    "roborank auth status --json",
                    "roborank auth login",
                    "roborank auth token create --scope resources:read --scope resources:write --json",
                ],
            },
            {
                "name": "Discover resources",
                "when": "Before tagging evidence, creating related resources, or reading docs for dependencies.",
                "steps": [
                    "Search by kind, namespace, or text.",
                    "Read the canonical resource payload.",
                    "Resolve aliases before storing or reusing an ID.",
                ],
                "commands": [
                    'roborank resources search --kind robot --query "<name>" --json',
                    "roborank resources read robot <namespace>/<slug> --json",
                    "roborank resources resolve robot <namespace>/<slug> --json",
                ],
            },
            {
                "name": "Manage resources",
                "when": "When a robot, environment, policy, or policy family needs a canonical public record.",
                "steps": [
                    "Choose the correct kind and canonical resource ID.",
                    "Create the record with a title, summary, and README markdown.",
                    "Update metadata or rename with --new-id only when the old ID should become an alias.",
                    "Retrieve README markdown when another workflow needs resource-specific instructions.",
                ],
                "commands": [
                    'roborank resources create robot <namespace>/<slug> --title "<title>" --summary "<summary>" --markdown README.md --yes --non-interactive --json',
                    "roborank resources update robot <namespace>/<slug> --markdown README.md --yes --non-interactive --json",
                    "roborank resources readme robot <namespace>/<slug> --out README.md",
                ],
            },
            {
                "name": "Prepare evidence upload",
                "when": "Before pushing a Rerun recording into RoboRank.",
                "steps": [
                    "Resolve robot, environment, and exact policy IDs.",
                    "Fetch the environment metrics schema.",
                    "Create or validate metrics.json when a schema exists.",
                    "Upload one .rrd recording under the size limit with explicit license and visibility.",
                ],
                "commands": [
                    "roborank metrics schema --environment <namespace>/<slug> --json",
                    "roborank metrics setup --environment <namespace>/<slug> --from result.json --out metrics.json",
                    "roborank metrics validate --environment <namespace>/<slug> metrics.json --json",
                    (
                        "roborank evidence upload --rrd recording.rrd --metrics metrics.json "
                        "--robot <namespace>/<slug> --environment <namespace>/<slug> --policy <namespace>/<slug> "
                        "--license CC-BY-4.0 --yes --non-interactive --json"
                    ),
                ],
            },
            {
                "name": "Run evals",
                "when": "When validating a policy locally or creating an upload-ready run bundle.",
                "steps": [
                    "List or inspect packaged challenges.",
                    "Run locally with a policy source file and an output directory.",
                    "Use the generated recording and metrics with evidence commands when publishing.",
                ],
                "commands": [
                    "roborank eval list --json",
                    "roborank eval show <challenge_id> --json",
                    "roborank eval run <challenge_id> --policy-source robot_policy.py --out runs/local-001 --json",
                ],
            },
        ],
        "commands": {
            "auth_status": "roborank auth status --json",
            "resource_search": 'roborank resources search --kind robot --query "<name>" --json',
            "resource_read": "roborank resources read robot <namespace>/<slug> --json",
            "resource_resolve": "roborank resources resolve robot <namespace>/<slug> --json",
            "resource_create": 'roborank resources create robot <namespace>/<slug> --title "<title>" --markdown README.md --yes --non-interactive --json',
            "resource_update": "roborank resources update robot <namespace>/<slug> --markdown README.md --yes --non-interactive --json",
            "resource_readme": "roborank resources readme robot <namespace>/<slug> --out README.md",
            "metrics_schema": "roborank metrics schema --environment <namespace>/<slug> --json",
            "metrics_init": "roborank metrics init --environment <namespace>/<slug> --out metrics.json",
            "metrics_setup": "roborank metrics setup --environment <namespace>/<slug> --from result.json --out metrics.json",
            "metrics_validate": "roborank metrics validate --environment <namespace>/<slug> metrics.json --json",
            "evidence_upload": (
                "roborank evidence upload --rrd recording.rrd --metrics metrics.json "
                "--robot <namespace>/<slug> --environment <namespace>/<slug> --policy <namespace>/<slug> "
                "--license CC-BY-4.0 --yes --non-interactive --json"
            ),
            "eval_run": "roborank eval run <challenge_id> --policy-source robot_policy.py --out runs/local-001 --json",
        },
        "resource_management": {
            "kinds": sorted(RESOURCE_KINDS),
            "read_paths": [
                "roborank resources search --kind <kind> --query <text> --json",
                "roborank resources read <kind> <namespace>/<slug> --json",
                "roborank resources resolve <kind> <namespace>/<slug> --json",
                "roborank resources readme <kind> <namespace>/<slug> --out README.md",
            ],
            "write_paths": [
                "roborank resources create <kind> <namespace>/<slug> --title <title> --markdown README.md --yes --non-interactive --json",
                "roborank resources update <kind> <namespace>/<slug> --markdown README.md --yes --non-interactive --json",
            ],
            "boundaries": [
                "A robot resource describes the physical or simulated robot platform.",
                "An environment resource describes the task world, challenge, or metrics context.",
                "A policy resource describes one exact policy implementation or submitted behavior.",
                "A policy_family resource groups related policy variants and runs.",
            ],
        },
        "rules": [
            "One evidence run is one Rerun .rrd recording.",
            "Do not fabricate RoboRank resource IDs; search, read, or resolve them through the API.",
            "Validate metrics.json before upload when the environment has an active schema.",
            "Do not upload policy source code as public evidence through evidence upload.",
        ],
        "task": args.task,
        "environment": args.environment,
        "challenge": args.challenge,
    }


def command_prime(ctx: Context, args: argparse.Namespace) -> int:
    primer = build_agent_primer(args)
    if ctx.json_output or ctx.yaml_output:
        return emit_success(ctx, "prime", primer)
    text = f"""# RoboRank Agent Primer

Use this as the index for RoboRank automation. It is not a help screen; it names
the stable terms and standard procedures the model should follow before
guessing at IDs, schemas, or upload metadata.

## Terms

- Resource: a canonical robot, environment, policy, or policy_family record.
  Boundary: metadata, aliases, ownership, and README markdown only; not evidence
  artifacts or private policy source.
- Eval: a packaged RoboRank challenge execution against a policy. Boundary:
  local or hosted RoboRank challenges only; not arbitrary simulation jobs.
- Evidence: one observed run artifact, normally one `.rrd` recording plus
  optional `metrics.json`. Boundary: links resources to a run; public upload
  does not upload policy source code by default.
- Metrics: JSON values validated against an environment schema when one exists.
  Boundary: summary data for the run, not a replacement for the recording.

Do not fabricate RoboRank resource IDs. Use:

```text
roborank resources search --kind robot --query "<name>" --json
roborank resources read robot <namespace>/<slug> --json
roborank resources resolve robot <namespace>/<slug> --json
roborank resources readme robot <namespace>/<slug> --out README.md
```

## Standard Procedures

1. Authenticate with `roborank auth status --json`, then configure a token with
   the smallest required scopes.
2. Discover resources with `resources search`, read canonical payloads with
   `resources read`, and resolve aliases with `resources resolve`.
3. Create or update public resource records with README markdown:
   `roborank resources create robot <namespace>/<slug> --title "<title>" --markdown README.md --yes --non-interactive --json`
4. Before evidence upload, ensure the Rerun file is a single `.rrd` under 40 MB,
   check the environment metrics schema, validate metrics when required, and
   upload with an explicit artifact license.
5. For local evals, run:
   `roborank eval run <challenge_id> --policy-source robot_policy.py --out runs/local-001 --json`
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
                "token_source": ctx.options.token_source,
                "auth_path": ctx.options.auth_path,
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
        removed = remove_auth_profile(Path(ctx.options.auth_path or str(auth_path())), profile=ctx.options.profile)
        return emit_success(
            ctx,
            "auth.logout",
            {
                "message": "Removed saved CLI auth if present. Also unset ROBORANK_TOKEN or profile config tokens if configured.",
                "auth_path": ctx.options.auth_path,
                "profile": ctx.options.profile,
                "removed_auth_file_token": removed,
            },
        )
    login_url = f"{ctx.api_url.rstrip('/')}/api/auth/cli/login"
    opened_browser = False
    if not getattr(args, "no_browser", False):
        opened_browser = webbrowser.open(login_url)
    auth_file = Path(ctx.options.auth_path or str(auth_path()))
    saved = False
    if not (ctx.json_output or ctx.yaml_output or ctx.options.non_interactive):
        print(f"Open this URL to create a personal access token: {login_url}", file=ctx.stderr)
        print("Paste the generated token, then press Enter. Leave blank to skip saving.", file=ctx.stderr)
        token = ctx.stdin.readline().strip()
        if token:
            write_auth_file(auth_file, profile=ctx.options.profile, api_url=ctx.api_url, token=token)
            saved = True
    return emit_success(
        ctx,
        "auth.login",
        {
            "message": "Create a personal access token in the browser, then paste it into the CLI prompt to save it.",
            "login_url": login_url,
            "auth_path": str(auth_file),
            "profile": ctx.options.profile,
            "no_browser": bool(getattr(args, "no_browser", False)),
            "opened_browser": opened_browser,
            "saved": saved,
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


def read_resource(api: ApiClient, kind: str, resource_id: str) -> dict[str, Any]:
    namespace, slug = canonical_id_parts(resource_id)
    return api.request_json("GET", f"/api/resources/{kind}/{namespace}/{slug}")


def resource_payload_object(payload: dict[str, Any]) -> dict[str, Any]:
    resource = payload.get("resource")
    return resource if isinstance(resource, dict) else payload


def resource_markdown(payload: dict[str, Any]) -> str:
    resource = resource_payload_object(payload)
    for container in (resource, payload):
        markdown = container.get("markdown")
        if isinstance(markdown, str):
            return markdown
    return ""


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def command_resources(ctx: Context, args: argparse.Namespace) -> int:
    api = client(ctx)
    if args.resources_command in {"search", "list"}:
        return resource_result_command(ctx, args, f"resources.{args.resources_command}")
    if args.resources_command in {"show", "read", "resolve", "readme"}:
        if args.kind not in RESOURCE_KINDS:
            raise RoboRankError("Invalid resource kind.", code="usage_error", exit_code=2)
        if args.resources_command == "resolve":
            canonical_id_parts(args.resource_id)
            payload = api.request_json("GET", "/api/resources/resolve", params={"kind": args.kind, "id": args.resource_id})
        else:
            payload = read_resource(api, args.kind, args.resource_id)
        if args.resources_command == "readme":
            markdown = resource_markdown(payload)
            resource = resource_payload_object(payload)
            warnings = []
            if not markdown:
                warnings.append({"code": "resource_readme_empty", "message": "Resource has no README markdown."})
            result = {
                "resource": {
                    "kind": resource.get("kind", args.kind),
                    "canonicalId": resource.get("canonicalId") or resource.get("canonical_id") or args.resource_id,
                },
                "readme": markdown,
            }
            if args.out:
                write_text(Path(args.out), markdown)
                result["path"] = args.out
                return emit_success(ctx, "resources.readme", result, warnings=warnings)
            if ctx.json_output or ctx.yaml_output:
                return emit_success(ctx, "resources.readme", result, warnings=warnings)
            if markdown:
                ctx.stdout.write(markdown)
                if not markdown.endswith("\n"):
                    ctx.stdout.write("\n")
            for warning in warnings:
                print(f"warning: {warning['message']}", file=ctx.stderr)
            return 9 if warnings and ctx.options.fail_on_warning else 0
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
    for name in ("show", "read", "resolve"):
        cmd = resources_sub.add_parser(name)
        cmd.add_argument("kind", choices=sorted(RESOURCE_KINDS))
        cmd.add_argument("resource_id")
        cmd.set_defaults(handler=command_resources)
    readme = resources_sub.add_parser("readme")
    readme.add_argument("kind", choices=sorted(RESOURCE_KINDS))
    readme.add_argument("resource_id")
    readme.add_argument("--out")
    readme.set_defaults(handler=command_resources)
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
    ):
        value = getattr(args, attr, None)
        if value:
            parts.append(str(value))
            break
    return ".".join(parts)


def main(
    argv: list[str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    stdin: TextIO | None = None,
) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    stdin = stdin or sys.stdin
    command = "unknown"
    try:
        options, filtered = extract_global_options(argv)
        ctx = Context(options=options, stdout=stdout, stderr=stderr, stdin=stdin)
        parser = build_parser()
        args = parser.parse_args(filtered)
        command = command_name(args)
        handler: Callable[[Context, argparse.Namespace], int] = args.handler
        return handler(ctx, args)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    except RoboRankError as exc:
        ctx = Context(
            options=options if "options" in locals() else GlobalOptions(api_url=DEFAULT_API_URL),
            stdout=stdout,
            stderr=stderr,
            stdin=stdin,
        )
        return emit_error(ctx, command, exc)
    except Exception as exc:  # noqa: BLE001 - CLI boundary must return structured failures.
        ctx = Context(
            options=options if "options" in locals() else GlobalOptions(api_url=DEFAULT_API_URL),
            stdout=stdout,
            stderr=stderr,
            stdin=stdin,
        )
        return emit_error(ctx, command, RoboRankError(str(exc), code="internal_error", exit_code=1))


if __name__ == "__main__":
    raise SystemExit(main())
