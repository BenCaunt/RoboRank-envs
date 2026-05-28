from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import roborank.cli as cli_module
from roborank.cli import ApiClient, CHALLENGE_RESOURCE_CACHE, main


def run_cli(*argv: str) -> tuple[int, dict]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(list(argv), stdout=stdout, stderr=stderr)
    payload = json.loads(stdout.getvalue())
    return code, payload


def run_cli_raw(*argv: str) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(list(argv), stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


class CliTests(unittest.TestCase):
    def test_challenge_resource_cache_covers_env_catalog(self) -> None:
        catalog_ids = {challenge["id"] for challenge in cli_module.list_challenges()}

        self.assertEqual(set(CHALLENGE_RESOURCE_CACHE), catalog_ids)
        self.assertNotIn(("std/unknown", "std/unknown"), CHALLENGE_RESOURCE_CACHE.values())
        self.assertEqual(CHALLENGE_RESOURCE_CACHE["cart_pole"], ("roborank/cart-pole-cart-v1", "roborank/cart-pole"))

    def test_prime_agent_json_is_stable(self) -> None:
        code, payload = run_cli("prime", "--agent", "--json")

        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"]["primer_version"], "roborank-agent-primer-v0")
        self.assertEqual(payload["result"]["model"]["max_rrd_bytes"], 41943040)
        self.assertIn("robot", payload["result"]["model"]["required_tags"])

    def test_format_yaml_outputs_structured_envelope(self) -> None:
        code, stdout, stderr = run_cli_raw("--format", "yaml", "prime", "--agent")

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("ok: true\n", stdout)
        self.assertIn('command: "prime"', stdout)
        self.assertIn('primer_version: "roborank-agent-primer-v0"', stdout)
        self.assertNotEqual(stdout.strip()[0], "{")

    def test_format_yaml_errors_use_structured_stdout(self) -> None:
        code, stdout, stderr = run_cli_raw("--format", "yaml", "evidence", "show")

        self.assertEqual(code, 2)
        self.assertEqual(stderr, "")
        self.assertIn("ok: false\n", stdout)
        self.assertIn('command: "evidence.show"', stdout)
        self.assertIn('code: "usage_error"', stdout)

    def test_global_options_load_profile_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            config.write_text(
                """
api_url = "https://top.example"
token = "top-token"
format = "yaml"

[profiles.staging]
api_url = "https://staging.example"
token = "profile-token"
format = "json"
non_interactive = true
""".strip()
                + "\n"
            )
            with patch.dict(os.environ, {"ROBORANK_CONFIG": str(config)}, clear=True):
                def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
                    self.assertEqual(api.token, "profile-token")
                    self.assertEqual(method, "GET")
                    self.assertEqual(path, "/api/auth/cli/status")
                    return {"authenticated": True, "authSource": "access_token", "user": {"email": "dev@example.com"}}

                with patch.object(ApiClient, "request_json", fake_request_json):
                    code, payload = run_cli("auth", "status", "--profile", "staging")

        self.assertEqual(code, 0)
        self.assertEqual(payload["api_url"], "https://staging.example")
        self.assertEqual(payload["result"]["profile"], "staging")
        self.assertTrue(payload["result"]["authenticated"])

    def test_global_options_precedence_over_profile_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            config.write_text(
                """
[profiles.default]
api_url = "https://config.example"
token = "config-token"
format = "yaml"
""".strip()
                + "\n"
            )
            with patch.dict(
                os.environ,
                {
                    "ROBORANK_CONFIG": str(config),
                    "ROBORANK_API_URL": "https://env.example",
                    "ROBORANK_TOKEN": "env-token",
                    "ROBORANK_FORMAT": "json",
                },
                clear=True,
            ):
                def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
                    self.assertEqual(api.api_url, "https://flag.example")
                    self.assertEqual(api.token, "env-token")
                    self.assertEqual(method, "GET")
                    self.assertEqual(path, "/api/auth/cli/status")
                    return {"authenticated": True, "authSource": "access_token", "user": {"email": "dev@example.com"}}

                with patch.object(ApiClient, "request_json", fake_request_json):
                    code, payload = run_cli("--api-url", "https://flag.example", "auth", "status")

        self.assertEqual(code, 0)
        self.assertEqual(payload["api_url"], "https://flag.example")
        self.assertTrue(payload["result"]["authenticated"])

    def test_evidence_init_uses_project_defaults(self) -> None:
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "roborank.toml").write_text(
                """
[evidence]
robot = "benkant/flat-disk-robot"
environment = "benkant/warehouse-floor"
policy_family = "benkant/pure-pursuit"
license = "CC0-1.0"
visibility = "private"

[[evidence.source_links]]
kind = "source_repo"
url = "https://github.com/benkant/flat-disk-policy"
label = "Policy repo"
""".strip()
                + "\n"
            )
            try:
                os.chdir(workspace)
                code, payload = run_cli("evidence", "init", "--out", "evidence.json", "--policy", "benkant/policy-001", "--json")
                envelope = json.loads((workspace / "evidence.json").read_text())
            finally:
                os.chdir(old_cwd)

        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["path"], "evidence.json")
        self.assertEqual(envelope["robot"], "benkant/flat-disk-robot")
        self.assertEqual(envelope["environment"], "benkant/warehouse-floor")
        self.assertEqual(envelope["policy"], "benkant/policy-001")
        self.assertEqual(envelope["policy_family"], "benkant/pure-pursuit")
        self.assertEqual(envelope["license"], "CC0-1.0")
        self.assertEqual(envelope["visibility"], "private")
        self.assertEqual(envelope["source_links"][0]["url"], "https://github.com/benkant/flat-disk-policy")

    def test_evidence_validate_uses_project_defaults_below_flags(self) -> None:
        seen_metadata: dict = {}

        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            if path == "/api/resources/resolve":
                return {"resource": {"canonicalId": kwargs["params"]["id"]}}
            if path.endswith("/metrics-schema"):
                return {"schema": None}
            if method == "POST" and path == "/api/evidence-runs/validate":
                seen_metadata.update(kwargs["body"]["metadata"])
                return {"valid": True, "warnings": []}
            raise AssertionError((method, path))

        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, patch.object(ApiClient, "request_json", fake_request_json):
            workspace = Path(directory)
            (workspace / "recording.rrd").write_bytes(b"RRD")
            (workspace / "roborank.toml").write_text(
                """
[evidence]
robot = "benkant/config-robot"
environment = "std/unknown"
policy = "benkant/config-policy"
policy_family = "benkant/config-family"
license = "CC0-1.0"
visibility = "unlisted"

[[evidence.source_links]]
kind = "source_repo"
url = "https://github.com/benkant/config-policy"
label = "Config policy"
""".strip()
                + "\n"
            )
            try:
                os.chdir(workspace)
                code, payload = run_cli(
                    "evidence",
                    "validate",
                    "--rrd",
                    "recording.rrd",
                    "--policy",
                    "benkant/flag-policy",
                    "--json",
                )
            finally:
                os.chdir(old_cwd)

        self.assertEqual(code, 0)
        self.assertEqual(seen_metadata["tags"]["robot"], "benkant/config-robot")
        self.assertEqual(seen_metadata["tags"]["environment"], "std/unknown")
        self.assertEqual(seen_metadata["tags"]["policy"], "benkant/flag-policy")
        self.assertEqual(seen_metadata["tags"]["policy_family"], "benkant/config-family")
        self.assertEqual(seen_metadata["license"]["artifact"], "CC0-1.0")
        self.assertEqual(seen_metadata["visibility"], "unlisted")
        self.assertEqual(seen_metadata["links"][0]["url"], "https://github.com/benkant/config-policy")
        self.assertEqual(payload["result"]["tags"]["policy"], "benkant/flag-policy")

    def test_metrics_setup_extracts_metrics_from_result_json(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(method, "GET")
            self.assertTrue(path.endswith("/metrics-schema"))
            return {
                "schema": {
                    "id": "schema_1",
                    "schemaHash": "abc",
                    "jsonSchema": {
                        "type": "object",
                        "required": ["score", "success"],
                        "properties": {
                            "score": {"type": "number"},
                            "success": {"type": "boolean"},
                            "status": {"type": "string", "default": "unknown"},
                        },
                    },
                }
            }

        with tempfile.TemporaryDirectory() as directory, patch.object(ApiClient, "request_json", fake_request_json):
            result_path = Path(directory) / "result.json"
            out_path = Path(directory) / "metrics.json"
            result_path.write_text(json.dumps({"score": 1, "metrics": {"score": 0.91, "success": True}}))
            code, payload = run_cli(
                "metrics",
                "setup",
                "--environment",
                "roborank/diff-drive-reach-target",
                "--from",
                str(result_path),
                "--out",
                str(out_path),
                "--json",
            )
            metrics = json.loads(out_path.read_text())

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "metrics.setup")
        self.assertEqual(metrics, {"score": 0.91, "status": "unknown", "success": True})

    def test_metrics_setup_accepts_metrics_prefixed_set_for_root_schema(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(method, "GET")
            self.assertTrue(path.endswith("/metrics-schema"))
            return {
                "schema": {
                    "id": "schema_1",
                    "schemaHash": "abc",
                    "jsonSchema": {
                        "type": "object",
                        "required": ["success_rate"],
                        "properties": {"success_rate": {"type": "number"}},
                    },
                }
            }

        with tempfile.TemporaryDirectory() as directory, patch.object(ApiClient, "request_json", fake_request_json):
            out_path = Path(directory) / "metrics.json"
            code, payload = run_cli(
                "metrics",
                "setup",
                "--environment",
                "roborank/diff-drive-reach-target",
                "--set",
                "/metrics/success_rate=0.94",
                "--out",
                str(out_path),
                "--json",
            )
            metrics = json.loads(out_path.read_text())

        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["metrics_required"], True)
        self.assertEqual(metrics, {"success_rate": 0.94})

    def test_oversized_rrd_exits_6_before_upload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rrd = Path(directory) / "recording.rrd"
            with rrd.open("wb") as handle:
                handle.truncate(40 * 1024 * 1024 + 1)

            code, payload = run_cli(
                "evidence",
                "validate",
                "--rrd",
                str(rrd),
                "--robot",
                "roborank/differential-drive-cube-v1",
                "--environment",
                "roborank/diff-drive-reach-target",
                "--policy",
                "benkant/policy-001",
                "--json",
            )

        self.assertEqual(code, 6)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "rerun_artifact_too_large")

    def test_schema_backed_environment_requires_metrics(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            if path == "/api/resources/resolve":
                return {"resource": {"canonicalId": kwargs["params"]["id"]}}
            if path.endswith("/metrics-schema"):
                return {
                    "schema": {
                        "id": "schema_1",
                        "schemaHash": "abc",
                        "jsonSchema": {
                            "type": "object",
                            "required": ["score"],
                            "properties": {"score": {"type": "number"}},
                        },
                    }
                }
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as directory, patch.object(ApiClient, "request_json", fake_request_json):
            rrd = Path(directory) / "recording.rrd"
            rrd.write_bytes(b"RRD")
            code, payload = run_cli(
                "evidence",
                "validate",
                "--rrd",
                str(rrd),
                "--robot",
                "roborank/differential-drive-cube-v1",
                "--environment",
                "roborank/diff-drive-reach-target",
                "--policy",
                "benkant/policy-001",
                "--json",
            )

        self.assertEqual(code, 5)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "metrics_missing")

    def test_evidence_upload_sends_stable_client_upload_id(self) -> None:
        seen_validation_ids: list[str] = []
        seen_upload_ids: list[str] = []

        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            if path == "/api/resources/resolve":
                return {"resource": {"canonicalId": kwargs["params"]["id"]}}
            if path.endswith("/metrics-schema"):
                return {"schema": None}
            if method == "POST" and path == "/api/evidence-runs/validate":
                metadata = kwargs["body"]["metadata"]
                seen_validation_ids.append(metadata["client_upload_id"])
                self.assertEqual(metadata["notes"], "Calibration note")
                self.assertEqual(metadata["superseded_by_run_id"], "run_previous")
                return {"valid": True, "warnings": []}
            raise AssertionError((method, path))

        def fake_multipart(api: ApiClient, path: str, *, fields: dict[str, str], files: dict, attempts: int = 3):
            self.assertEqual(path, "/api/evidence-runs")
            self.assertEqual(attempts, 3)
            metadata = json.loads(fields["metadata"])
            seen_upload_ids.append(metadata["client_upload_id"])
            self.assertEqual(metadata["notes"], "Calibration note")
            self.assertEqual(metadata["superseded_by_run_id"], "run_previous")
            return {
                "runId": "run_123",
                "runUrl": "/runs/run_123",
                "clientUploadId": metadata["client_upload_id"],
                "warnings": [],
            }

        with tempfile.TemporaryDirectory() as directory, patch.object(ApiClient, "request_json", fake_request_json), patch.object(
            ApiClient, "multipart", fake_multipart
        ):
            rrd = Path(directory) / "recording.rrd"
            rrd.write_bytes(b"RRD")
            code, payload = run_cli(
                "evidence",
                "upload",
                "--rrd",
                str(rrd),
                "--robot",
                "roborank/differential-drive-cube-v1",
                "--environment",
                "std/unknown",
                "--policy",
                "benkant/policy-001",
                "--notes",
                "Calibration note",
                "--superseded-by-run-id",
                "run_previous",
                "--allow-new-policy",
                "--yes",
                "--non-interactive",
                "--json",
            )

        self.assertEqual(code, 0)
        self.assertEqual(len(seen_validation_ids), 1)
        self.assertEqual(seen_validation_ids, seen_upload_ids)
        self.assertTrue(seen_upload_ids[0].startswith("upload_"))
        self.assertEqual(payload["result"]["client_upload_id"], seen_upload_ids[0])

    def test_evidence_upload_preserves_migration_metadata_from_envelope(self) -> None:
        seen_validation_metadata: dict = {}
        seen_upload_metadata: dict = {}

        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            if path == "/api/resources/resolve":
                return {"resource": {"canonicalId": kwargs["params"]["id"]}}
            if path.endswith("/metrics-schema"):
                return {"schema": None}
            if method == "POST" and path == "/api/evidence-runs/validate":
                seen_validation_metadata.update(kwargs["body"]["metadata"])
                self.assertEqual(kwargs["body"]["metrics"], {"score": 88, "success": True})
                return {"valid": True, "warnings": []}
            raise AssertionError((method, path))

        def fake_multipart(api: ApiClient, path: str, *, fields: dict[str, str], files: dict, attempts: int = 3):
            self.assertEqual(path, "/api/evidence-runs")
            self.assertIn("recording", files)
            self.assertIn("metrics", files)
            seen_upload_metadata.update(json.loads(fields["metadata"]))
            return {"runId": "run_migrated", "runUrl": "/runs/run_migrated", "warnings": []}

        with tempfile.TemporaryDirectory() as directory, patch.object(ApiClient, "request_json", fake_request_json), patch.object(
            ApiClient, "multipart", fake_multipart
        ):
            root = Path(directory)
            (root / "recording.rrd").write_bytes(b"RRD")
            (root / "metrics.json").write_text(json.dumps({"score": 88, "success": True}))
            (root / "evidence.json").write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.rerun_evidence.v0",
                        "title": "Migrated run",
                        "robot": "roborank/differential-drive-cube-v1",
                        "environment": "std/unknown",
                        "policy": "migration/diff-drive-reach-target-sub-123",
                        "policy_family": "migration/diff-drive-reach-target",
                        "recording_path": "recording.rrd",
                        "metrics_path": "metrics.json",
                        "license": "CC-BY-4.0",
                        "source_kind": "migration",
                        "legacy_submission_id": "sub_123",
                        "migration_provenance": "regenerated_from_code",
                        "migration_status": "uploaded",
                        "legacy_score": 77,
                        "legacy_metrics_json": {"score": 77, "success": False},
                        "regenerated_score": 88,
                        "regenerated_metrics_json": {"score": 88, "success": True},
                        "migration_notes": "rerun completed under staging seed data",
                        "legacy_code_hash": "hash_123",
                    }
                )
            )
            code, payload = run_cli(
                "evidence",
                "upload",
                "--from",
                str(root / "evidence.json"),
                "--allow-new-policy",
                "--allow-new-policy-family",
                "--yes",
                "--non-interactive",
                "--json",
            )

        self.assertEqual(code, 0)
        self.assertEqual(seen_validation_metadata, seen_upload_metadata)
        self.assertEqual(seen_upload_metadata["source_kind"], "migration")
        self.assertEqual(seen_upload_metadata["legacy_submission_id"], "sub_123")
        self.assertEqual(seen_upload_metadata["migration_method"], "regenerated_from_code")
        self.assertEqual(seen_upload_metadata["legacy_score"], 77)
        self.assertEqual(seen_upload_metadata["legacy_metrics_json"], {"score": 77, "success": False})
        self.assertEqual(seen_upload_metadata["regenerated_score"], 88)
        self.assertEqual(seen_upload_metadata["regenerated_metrics_json"], {"score": 88, "success": True})
        self.assertEqual(seen_upload_metadata["migration_notes"], "rerun completed under staging seed data")
        self.assertEqual(seen_upload_metadata["legacy_code_hash"], "hash_123")
        self.assertEqual(payload["result"]["run_id"], "run_migrated")

    def test_evidence_validate_requires_policy_family_creation_flag(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            if path == "/api/resources/resolve":
                if kwargs["params"]["kind"] == "policy_family":
                    raise cli_module.RoboRankError("not found", code="resource_not_found", exit_code=4)
                return {"resource": {"canonicalId": kwargs["params"]["id"]}}
            if path.endswith("/metrics-schema"):
                return {"schema": None}
            raise AssertionError((method, path))

        with tempfile.TemporaryDirectory() as directory, patch.object(ApiClient, "request_json", fake_request_json):
            rrd = Path(directory) / "recording.rrd"
            rrd.write_bytes(b"RRD")
            code, payload = run_cli(
                "evidence",
                "validate",
                "--rrd",
                str(rrd),
                "--robot",
                "roborank/differential-drive-cube-v1",
                "--environment",
                "std/unknown",
                "--policy",
                "benkant/policy-001",
                "--policy-family",
                "benkant/policy-family",
                "--json",
            )

        self.assertEqual(code, 4)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "resource_not_found")

    def test_evidence_validate_sends_policy_family_creation_flag(self) -> None:
        seen_metadata: dict = {}

        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            if path == "/api/resources/resolve":
                if kwargs["params"]["kind"] == "policy_family":
                    raise cli_module.RoboRankError("not found", code="resource_not_found", exit_code=4)
                return {"resource": {"canonicalId": kwargs["params"]["id"]}}
            if path.endswith("/metrics-schema"):
                return {"schema": None}
            if method == "POST" and path == "/api/evidence-runs/validate":
                seen_metadata.update(kwargs["body"]["metadata"])
                return {"valid": True, "warnings": []}
            raise AssertionError((method, path))

        with tempfile.TemporaryDirectory() as directory, patch.object(ApiClient, "request_json", fake_request_json):
            rrd = Path(directory) / "recording.rrd"
            rrd.write_bytes(b"RRD")
            code, payload = run_cli(
                "evidence",
                "validate",
                "--rrd",
                str(rrd),
                "--robot",
                "roborank/differential-drive-cube-v1",
                "--environment",
                "std/unknown",
                "--policy",
                "benkant/policy-001",
                "--policy-family",
                "benkant/policy-family",
                "--allow-new-policy-family",
                "--json",
            )

        self.assertEqual(code, 0)
        self.assertTrue(seen_metadata["allow_new_policy_family"])
        self.assertEqual(payload["result"]["metadata"]["allow_new_policy_family"], True)
        self.assertIn("policy_family_will_be_created", [warning["code"] for warning in payload["warnings"]])

    def test_evidence_validate_warns_for_mutable_source_link(self) -> None:
        seen_metadata: dict = {}

        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            if path == "/api/resources/resolve":
                return {"resource": {"canonicalId": kwargs["params"]["id"]}}
            if path.endswith("/metrics-schema"):
                return {"schema": None}
            if method == "POST" and path == "/api/evidence-runs/validate":
                seen_metadata.update(kwargs["body"]["metadata"])
                return {"valid": True, "warnings": []}
            raise AssertionError((method, path))

        with tempfile.TemporaryDirectory() as directory, patch.object(ApiClient, "request_json", fake_request_json):
            rrd = Path(directory) / "recording.rrd"
            rrd.write_bytes(b"RRD")
            code, payload = run_cli(
                "evidence",
                "validate",
                "--rrd",
                str(rrd),
                "--robot",
                "roborank/differential-drive-cube-v1",
                "--environment",
                "std/unknown",
                "--policy",
                "benkant/policy-001",
                "--source-link",
                "github=https://github.com/benkant/flat-disk-policy",
                "--json",
            )

        self.assertEqual(code, 0)
        self.assertEqual(seen_metadata["links"][0]["kind"], "source_repo")
        warnings = [warning["code"] for warning in payload["warnings"]]
        self.assertIn("source_link_revision_missing", warnings)

    def test_evidence_validate_rejects_invalid_source_link_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rrd = Path(directory) / "recording.rrd"
            rrd.write_bytes(b"RRD")
            code, payload = run_cli(
                "evidence",
                "validate",
                "--rrd",
                str(rrd),
                "--robot",
                "roborank/differential-drive-cube-v1",
                "--environment",
                "std/unknown",
                "--policy",
                "benkant/policy-001",
                "--source-link",
                "source_repo=not-a-url",
                "--json",
            )

        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "usage_error")
        self.assertIn("http(s)", payload["error"]["message"])

    def test_evidence_validate_rejects_unknown_source_link_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rrd = Path(directory) / "recording.rrd"
            rrd.write_bytes(b"RRD")
            code, payload = run_cli(
                "evidence",
                "validate",
                "--rrd",
                str(rrd),
                "--robot",
                "roborank/differential-drive-cube-v1",
                "--environment",
                "std/unknown",
                "--policy",
                "benkant/policy-001",
                "--source-link",
                "random=https://example.com/source",
                "--json",
            )

        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "usage_error")
        self.assertEqual(payload["error"]["details"]["kind"], "random")

    def test_evidence_init_rejects_invalid_visibility_from_project_defaults(self) -> None:
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "roborank.toml").write_text(
                """
[evidence]
visibility = "secret"
""".strip()
                + "\n"
            )
            os.chdir(workspace)
            try:
                code, payload = run_cli("evidence", "init", "--out", "evidence.json", "--json")
            finally:
                os.chdir(old_cwd)

        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "usage_error")
        self.assertEqual(payload["error"]["details"]["visibility"], "secret")

    def test_evidence_validate_rejects_invalid_visibility_from_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "recording.rrd").write_bytes(b"RRD")
            envelope = workspace / "evidence.json"
            envelope.write_text(
                json.dumps(
                    {
                        "recording_path": "recording.rrd",
                        "robot": "roborank/differential-drive-cube-v1",
                        "environment": "std/unknown",
                        "policy": "benkant/policy-001",
                        "license": "CC-BY-4.0",
                        "visibility": "secret",
                    }
                )
            )
            code, payload = run_cli("evidence", "validate", "--from", str(envelope), "--json")

        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "usage_error")
        self.assertEqual(payload["error"]["details"]["visibility"], "secret")

    def test_evidence_validate_rejects_metrics_file_with_wrong_extension(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            if path == "/api/resources/resolve":
                return {"resource": {"canonicalId": kwargs["params"]["id"]}}
            if path.endswith("/metrics-schema"):
                return {"schema": None}
            raise AssertionError((method, path))

        with tempfile.TemporaryDirectory() as directory, patch.object(ApiClient, "request_json", fake_request_json):
            workspace = Path(directory)
            rrd = workspace / "recording.rrd"
            rrd.write_bytes(b"RRD")
            metrics = workspace / "metrics.txt"
            metrics.write_text("{}")
            code, payload = run_cli(
                "evidence",
                "validate",
                "--rrd",
                str(rrd),
                "--metrics",
                str(metrics),
                "--robot",
                "roborank/differential-drive-cube-v1",
                "--environment",
                "std/unknown",
                "--policy",
                "benkant/policy-001",
                "--json",
            )

        self.assertEqual(code, 5)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "metrics_invalid")

    def test_evidence_validate_rejects_oversized_metrics_file(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            if path == "/api/resources/resolve":
                return {"resource": {"canonicalId": kwargs["params"]["id"]}}
            if path.endswith("/metrics-schema"):
                return {"schema": None}
            raise AssertionError((method, path))

        with tempfile.TemporaryDirectory() as directory, patch.object(ApiClient, "request_json", fake_request_json):
            workspace = Path(directory)
            rrd = workspace / "recording.rrd"
            rrd.write_bytes(b"RRD")
            metrics = workspace / "metrics.json"
            metrics.write_bytes(b"{" + b" " * cli_module.MAX_METRICS_BYTES + b"}")
            code, payload = run_cli(
                "evidence",
                "validate",
                "--rrd",
                str(rrd),
                "--metrics",
                str(metrics),
                "--robot",
                "roborank/differential-drive-cube-v1",
                "--environment",
                "std/unknown",
                "--policy",
                "benkant/policy-001",
                "--json",
            )

        self.assertEqual(code, 5)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "metrics_too_large")
        self.assertEqual(payload["error"]["details"]["max_bytes"], cli_module.MAX_METRICS_BYTES)

    def test_evidence_show_can_recover_by_client_upload_id(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(method, "GET")
            self.assertEqual(path, "/api/evidence-runs")
            self.assertEqual(kwargs["params"], {"client_upload_id": "upload_retry_123"})
            return {"run": {"id": "run_123"}, "clientUploadId": "upload_retry_123"}

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli("evidence", "show", "--client-upload-id", "upload_retry_123", "--json")

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "evidence.show")
        self.assertEqual(payload["result"]["run"]["id"], "run_123")

    def test_evidence_show_requires_single_lookup_key(self) -> None:
        code, payload = run_cli("evidence", "show", "run_123", "--client-upload-id", "upload_retry_123", "--json")

        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "usage_error")

    def test_eval_list_uses_packaged_catalog(self) -> None:
        code, payload = run_cli("eval", "list", "--json")

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "eval.list")
        self.assertIn("diff_drive_reach_target", {challenge["id"] for challenge in payload["result"]})

    def test_eval_show_uses_packaged_catalog(self) -> None:
        code, payload = run_cli("eval", "show", "diff_drive_reach_target", "--json")

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "eval.show")
        self.assertEqual(payload["result"]["id"], "diff_drive_reach_target")

    def test_eval_submit_returns_evidence_link(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/api/runs")
            self.assertIn("code", kwargs["body"])
            self.assertTrue(kwargs["body"]["evidenceLicenseAccepted"])
            return {
                "score": 93,
                "status": "success",
                "metrics": {"score": 93},
                "evidence_run_id": "run_123",
                "evidence_url": "/runs/run_123",
            }

        with tempfile.TemporaryDirectory() as directory, patch.object(ApiClient, "request_json", fake_request_json):
            policy = Path(directory) / "policy.py"
            policy.write_text("class RobotPolicy: pass\n")
            code, payload = run_cli(
                "eval",
                "submit",
                "diff_drive_reach_target",
                "--policy-source",
                str(policy),
                "--yes",
                "--non-interactive",
                "--json",
            )

        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["evidence_run_id"], "run_123")
        self.assertEqual(payload["result"]["evidence_url"], "/runs/run_123")

    def test_migration_and_launch_commands_are_not_registered(self) -> None:
        for command in ("migration", "launch"):
            with self.subTest(command=command):
                argparse_stderr = io.StringIO()
                with patch("sys.stderr", argparse_stderr):
                    code, stdout, stderr = run_cli_raw(command, "--help")

                self.assertEqual(code, 2)
                self.assertEqual(stdout, "")
                self.assertEqual(stderr, "")
                self.assertIn("invalid choice", argparse_stderr.getvalue())
                self.assertIn(command, argparse_stderr.getvalue())

    def test_eval_run_writes_uploadable_local_bundle(self) -> None:
        def fake_run_local_challenge(challenge_id: str, policy_path: Path, max_steps: int | None, artifact_dir: Path):
            self.assertEqual(challenge_id, "diff_drive_reach_target")
            self.assertIn("RobotPolicy", policy_path.read_text())
            self.assertEqual(max_steps, 12)
            self.assertEqual(artifact_dir.name, "bundle")
            return {
                "challenge_id": challenge_id,
                "metrics": {"score": 91, "success": True, "status": "success"},
                "logs": ["status=success", "score=91"],
                "replay": {"artifacts": [{"type": "rerun_rrd", "url": "/artifacts/local.rrd"}]},
            }

        def fake_copy_local_rerun_artifact(result_payload: dict, out_dir: Path):
            recording = out_dir / "recording.rrd"
            recording.write_bytes(b"RRD")
            return recording

        with tempfile.TemporaryDirectory() as directory, patch.object(
            cli_module, "run_local_challenge", fake_run_local_challenge
        ), patch.object(cli_module, "copy_local_rerun_artifact", fake_copy_local_rerun_artifact):
            policy = Path(directory) / "policy.py"
            bundle = Path(directory) / "bundle"
            policy.write_text("class RobotPolicy: pass\n")
            code, payload = run_cli(
                "eval",
                "run",
                "diff_drive_reach_target",
                "--policy-source",
                str(policy),
                "--out",
                str(bundle),
                "--policy",
                "benkant/local-policy",
                "--policy-family",
                "benkant/diff-drive",
                "--max-steps",
                "12",
                "--json",
            )

            evidence = json.loads((bundle / "evidence.json").read_text())
            result = json.loads((bundle / "result.json").read_text())

        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["result_status"], "success")
        self.assertEqual(payload["result"]["recording_path"], "recording.rrd")
        self.assertEqual(evidence["recording_path"], "recording.rrd")
        self.assertEqual(evidence["metrics_path"], "metrics.json")
        self.assertEqual(evidence["policy"], "benkant/local-policy")
        self.assertEqual(result["environment"], "roborank/diff-drive-reach-target")
        self.assertNotIn("RobotPolicy", json.dumps(payload))

    def test_auth_token_create_posts_scopes(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/api/auth/tokens")
            self.assertEqual(kwargs["body"]["name"], "upload bot")
            self.assertEqual(kwargs["body"]["scopes"], ["evidence:write", "resources:read"])
            return {"token": "rrk_secret", "tokenId": "tok_123", "scopes": kwargs["body"]["scopes"]}

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli(
                "auth",
                "token",
                "create",
                "--scope",
                "evidence:write",
                "--scope",
                "resources:read",
                "--name",
                "upload bot",
                "--json",
            )

        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["token"], "rrk_secret")
        self.assertEqual(payload["result"]["scopes"], ["evidence:write", "resources:read"])

    def test_auth_login_opens_cli_login_route(self) -> None:
        with patch.object(cli_module.webbrowser, "open", return_value=True) as open_browser:
            code, payload = run_cli("--api-url", "https://roborank.example", "auth", "login", "--json")

        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["login_url"], "https://roborank.example/api/auth/cli/login")
        self.assertTrue(payload["result"]["opened_browser"])
        open_browser.assert_called_once_with("https://roborank.example/api/auth/cli/login")

    def test_auth_login_no_browser_returns_cli_login_route(self) -> None:
        with patch.object(cli_module.webbrowser, "open") as open_browser:
            code, payload = run_cli("--api-url", "https://roborank.example", "auth", "login", "--no-browser", "--json")

        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["login_url"], "https://roborank.example/api/auth/cli/login")
        self.assertFalse(payload["result"]["opened_browser"])
        open_browser.assert_not_called()

    def test_auth_token_create_rejects_invalid_scope(self) -> None:
        code, payload = run_cli("auth", "token", "create", "--scope", "admin", "--json")

        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "invalid_token_scope")

    def test_auth_token_create_rejects_migration_admin_scope(self) -> None:
        code, payload = run_cli("auth", "token", "create", "--scope", "migration:admin", "--json")

        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "invalid_token_scope")

    def test_resource_update_can_rename_canonical_id(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(method, "PATCH")
            self.assertEqual(path, "/api/resources/robot/benkant/old-bot")
            self.assertEqual(kwargs["body"]["namespace"], "benkant")
            self.assertEqual(kwargs["body"]["slug"], "new-bot")
            return {
                "resource": {
                    "kind": "robot",
                    "canonicalId": "benkant/new-bot",
                    "aliases": ["benkant/old-bot"],
                }
            }

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli(
                "resources",
                "update",
                "robot",
                "benkant/old-bot",
                "--new-id",
                "benkant/new-bot",
                "--yes",
                "--non-interactive",
                "--json",
            )

        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["resource"]["canonicalId"], "benkant/new-bot")
        self.assertEqual(payload["result"]["resource"]["aliases"], ["benkant/old-bot"])


if __name__ == "__main__":
    unittest.main()
