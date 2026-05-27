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
    def write_passing_privacy_report(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "roborank.privacy_scan.v0",
                    "source_exposure": False,
                    "checks": [{"name": name, "status": "pass"} for name in sorted(cli_module.GO_NO_GO_REQUIRED_PRIVACY_CHECKS)],
                }
            )
        )

    def write_passing_schema_guard_report(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "roborank.schema_guard.v0",
                    "environment": "roborank/diff-drive-reach-target",
                    "checks": [{"name": name, "status": "pass"} for name in sorted(cli_module.GO_NO_GO_REQUIRED_SCHEMA_GUARD_CHECKS)],
                }
            )
        )

    def write_passing_trust_actions_report(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "roborank.trust_actions.v0",
                    "resource": {"kind": "robot", "id": "roborank/differential-drive-cube-v1"},
                    "run_id": "run_123",
                    "checks": [{"name": name, "status": "pass"} for name in sorted(cli_module.GO_NO_GO_REQUIRED_TRUST_ACTION_CHECKS)],
                }
            )
        )

    def write_passing_resource_guard_report(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "roborank.resource_guard.v0",
                    "known_resources": {
                        "robot": "roborank/differential-drive-cube-v1",
                        "environment": "std/unknown",
                        "policy": "qa/resource-guard-policy",
                    },
                    "checks": [{"name": name, "status": "pass"} for name in sorted(cli_module.GO_NO_GO_REQUIRED_RESOURCE_GUARD_CHECKS)],
                }
            )
        )

    def test_migration_resource_cache_covers_env_catalog(self) -> None:
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

    def test_launch_smoke_read_only_checks_release_routes(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(method, "GET")
            if path == "/api/me":
                self.assertEqual(api.token, "token-123")
                return {"user": {"id": "user_123"}}
            if path == "/api/challenges":
                return [{"id": "diff_drive_reach_target"}]
            if path == "/api/leaderboard":
                return {"entries": [], "total": 0}
            if path == "/api/submissions/sub_123":
                self.assertEqual(api.token, "token-123")
                return {"submission": {"id": "sub_123", "challengeId": "diff_drive_reach_target", "status": "success"}}
            if path == "/api/explorer/runs":
                if api.token:
                    return {"runs": [{"id": "run_123"}, {"id": "run_456"}, {"id": "run_789"}], "hasMore": False}
                return {"runs": [{"id": "run_123"}, {"id": "run_456"}], "hasMore": False}
            if path == "/api/explorer/facets":
                self.assertEqual(kwargs["params"], {"robot": "roborank/differential-drive-cube-v1"})
                return {"facets": [{"kind": "environment", "canonicalId": "roborank/diff-drive-reach-target", "count": 2}]}
            if path == "/api/evidence-runs/run_123":
                return {"id": "run_123", "artifacts": [{"url": "/api/evidence-artifacts/runs/run_123/recording.rrd"}]}
            if path == "/api/resources/robot/roborank/differential-drive-cube-v1":
                return {"resource": {"canonicalId": "roborank/differential-drive-cube-v1"}}
            if path == "/api/leaderboards/roborank/diff-drive-reach-target":
                return {"environment": {"canonicalId": "roborank/diff-drive-reach-target"}, "entries": []}
            if path == "/api/compare":
                self.assertEqual(kwargs["params"], {"run": ["run_123", "run_456"]})
                return {"runs": [{"id": "run_123"}, {"id": "run_456"}]}
            raise AssertionError((api.token, method, path, kwargs))

        def fake_request_bytes(api: ApiClient, path_or_url: str):
            self.assertEqual(path_or_url, "/api/evidence-artifacts/runs/run_123/recording.rrd")
            return b"RRD"

        with patch.object(ApiClient, "request_json", fake_request_json), patch.object(ApiClient, "request_bytes", fake_request_bytes):
            code, payload = run_cli(
                "--token",
                "token-123",
                "launch",
                "smoke",
                "--run-id",
                "run_123",
                "--resource-kind",
                "robot",
                "--resource-id",
                "roborank/differential-drive-cube-v1",
                "--environment",
                "roborank/diff-drive-reach-target",
                "--legacy-submission-id",
                "sub_123",
                "--compare-run",
                "run_123",
                "--compare-run",
                "run_456",
                "--json",
            )

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.smoke")
        checks = {check["name"]: check["status"] for check in payload["result"]["checks"]}
        self.assertEqual(checks["auth"], "pass")
        self.assertEqual(checks["problem_list"], "pass")
        self.assertEqual(checks["artifact_download"], "pass")
        self.assertEqual(checks["explorer_facets"], "pass")
        self.assertEqual(checks["environment_leaderboard"], "pass")
        self.assertEqual(checks["legacy_submission_detail"], "pass")
        self.assertEqual(checks["problem_submit_evidence_creation"], "skipped")

    def test_launch_smoke_mutating_requires_yes(self) -> None:
        code, payload = run_cli("launch", "smoke", "--include-mutating", "--policy-source", "policy.py", "--json")

        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "confirmation_required")

    def test_launch_smoke_rejects_more_than_four_compare_runs(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(method, "GET")
            if path == "/api/challenges":
                return [{"id": "diff_drive_reach_target"}]
            if path == "/api/leaderboard":
                return {"entries": [], "total": 0}
            if path == "/api/explorer/runs":
                return {"runs": [{"id": "run_1"}, {"id": "run_2"}], "hasMore": False}
            if path == "/api/explorer/facets":
                return {"facets": [{"kind": "robot", "canonicalId": "roborank/differential-drive-cube-v1", "count": 2}]}
            raise AssertionError((method, path, kwargs))

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli(
                "launch",
                "smoke",
                "--compare-run",
                "run_1",
                "--compare-run",
                "run_2",
                "--compare-run",
                "run_3",
                "--compare-run",
                "run_4",
                "--compare-run",
                "run_5",
                "--json",
            )

        self.assertEqual(code, 9)
        checks = {check["name"]: check for check in payload["error"]["details"]["checks"]}
        self.assertEqual(checks["compare"]["status"], "fail")
        self.assertEqual(checks["compare"]["code"], "compare_too_many_runs")

    def test_launch_smoke_mutating_problem_submit_requires_evidence_side_effect(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            if method == "GET" and path == "/api/me":
                return {"user": {"id": "user_123"}}
            if method == "GET" and path == "/api/challenges":
                return [{"id": "diff_drive_reach_target"}]
            if method == "GET" and path == "/api/leaderboard":
                return {"entries": [], "total": 0}
            if method == "GET" and path == "/api/explorer/runs":
                return {"runs": [{"id": "run_1"}, {"id": "run_2"}], "hasMore": False}
            if method == "GET" and path == "/api/explorer/facets":
                return {"facets": [{"kind": "robot", "canonicalId": "roborank/differential-drive-cube-v1", "count": 2}]}
            if method == "POST" and path == "/api/runs":
                self.assertTrue(kwargs["body"]["evidenceLicenseAccepted"])
                return {
                    "score": 91,
                    "status": "success",
                    "metrics": {"score": 91},
                    "evidence_run_id": "run_smoke",
                    "evidence_url": "/runs/run_smoke",
                }
            raise AssertionError((api.token, method, path, kwargs))

        with tempfile.TemporaryDirectory() as directory, patch.object(ApiClient, "request_json", fake_request_json):
            policy = Path(directory) / "policy.py"
            policy.write_text("class RobotPolicy: pass\n")
            code, payload = run_cli(
                "--token",
                "token-123",
                "--yes",
                "launch",
                "smoke",
                "--include-mutating",
                "--policy-source",
                str(policy),
                "--json",
            )

        self.assertEqual(code, 0)
        checks = {check["name"]: check for check in payload["result"]["checks"]}
        self.assertEqual(checks["problem_submit_evidence_creation"]["status"], "pass")
        self.assertEqual(checks["problem_submit_evidence_creation"]["data"]["evidence_run_id"], "run_smoke")

    def test_launch_smoke_mutating_direct_evidence_upload(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            if method == "GET" and path == "/api/me":
                return {"user": {"id": "user_123"}}
            if method == "GET" and path == "/api/challenges":
                return [{"id": "diff_drive_reach_target"}]
            if method == "GET" and path == "/api/leaderboard":
                return {"entries": [], "total": 0}
            if method == "GET" and path == "/api/explorer/runs":
                return {"runs": [{"id": "run_1"}, {"id": "run_2"}], "hasMore": False}
            if method == "GET" and path == "/api/explorer/facets":
                return {"facets": [{"kind": "environment", "canonicalId": "std/unknown", "count": 2}]}
            if method == "GET" and path == "/api/leaderboards/std/unknown":
                return {"environment": {"canonicalId": "std/unknown"}, "entries": []}
            if method == "GET" and path == "/api/resources/resolve":
                return {"resource": {"canonicalId": kwargs["params"]["id"]}}
            if method == "GET" and path.endswith("/metrics-schema"):
                return {"schema": None}
            if method == "POST" and path == "/api/evidence-runs/validate":
                self.assertEqual(kwargs["body"]["metadata"]["tags"]["environment"], "std/unknown")
                return {"valid": True, "warnings": []}
            raise AssertionError((api.token, method, path, kwargs))

        def fake_multipart(api: ApiClient, path: str, **kwargs):
            self.assertEqual(path, "/api/evidence-runs")
            self.assertIn("recording", kwargs["files"])
            metadata = json.loads(kwargs["fields"]["metadata"])
            self.assertEqual(metadata["tags"]["policy"], "benkant/smoke-policy")
            return {"runId": "run_direct", "runUrl": "/runs/run_direct"}

        with tempfile.TemporaryDirectory() as directory, patch.object(ApiClient, "request_json", fake_request_json), patch.object(
            ApiClient, "multipart", fake_multipart
        ):
            rrd = Path(directory) / "smoke.rrd"
            rrd.write_bytes(b"RRD")
            code, payload = run_cli(
                "--token",
                "token-123",
                "--yes",
                "launch",
                "smoke",
                "--include-mutating",
                "--direct-rrd",
                str(rrd),
                "--robot",
                "roborank/differential-drive-cube-v1",
                "--environment",
                "std/unknown",
                "--policy",
                "benkant/smoke-policy",
                "--json",
            )

        self.assertEqual(code, 0)
        checks = {check["name"]: check for check in payload["result"]["checks"]}
        self.assertEqual(checks["direct_evidence_creation"]["status"], "pass")
        self.assertEqual(checks["direct_evidence_creation"]["data"]["run_id"], "run_direct")

    def test_launch_restore_check_validates_d1_objects_and_smoke_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            d1_sql = root / "backup.sql"
            d1_sql.write_text("\n".join(f"CREATE TABLE {name} (id TEXT);" for name in sorted(cli_module.D1_RESTORE_REQUIRED_TABLES)))
            d1_hash = cli_module.sha256_file(d1_sql)
            object_root = root / "bucket"
            artifact = object_root / "runs" / "run_123" / "recording.rrd"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"RRD")
            manifest = root / "objects.json"
            manifest.write_text(
                json.dumps(
                    {
                        "objects": [
                            {
                                "key": "runs/run_123/recording.rrd",
                                "sha256": cli_module.sha256_file(artifact),
                                "size_bytes": artifact.stat().st_size,
                            }
                        ]
                    }
                )
            )
            smoke = root / "smoke.json"
            smoke.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.launch_smoke.v0",
                        "checks": [
                            {"name": "auth", "status": "pass"},
                            {"name": "problem_list", "status": "pass"},
                            {"name": "artifact_download", "status": "pass"},
                            {"name": "explorer_anonymous_limit", "status": "pass"},
                            {"name": "explorer_facets", "status": "pass"},
                            {"name": "run_detail", "status": "pass"},
                            {"name": "resource_page", "status": "pass"},
                            {"name": "environment_leaderboard", "status": "pass"},
                            {"name": "legacy_leaderboard", "status": "pass"},
                            {"name": "legacy_submission_detail", "status": "pass"},
                        ],
                    }
                )
            )

            code, payload = run_cli(
                "launch",
                "restore-check",
                "--target",
                "staging",
                "--strict",
                "--d1-sql",
                str(d1_sql),
                "--object-manifest",
                str(manifest),
                "--object-root",
                str(object_root),
                "--required-object",
                "runs/run_123/recording.rrd",
                "--api-smoke-report",
                str(smoke),
                "--json",
            )

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.restore-check")
        checks = {check["name"]: check for check in payload["result"]["checks"]}
        self.assertEqual(checks["d1_sql_import"]["status"], "pass")
        self.assertEqual(checks["object_integrity"]["data"]["verified_count"], 1)
        self.assertEqual(checks["api_smoke_report"]["status"], "pass")

    def test_launch_restore_check_rejects_missing_restored_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            d1_sql = root / "backup.sql"
            d1_sql.write_text("\n".join(f"CREATE TABLE {name} (id TEXT);" for name in sorted(cli_module.D1_RESTORE_REQUIRED_TABLES)))
            manifest = root / "objects.json"
            manifest.write_text(json.dumps({"objects": [{"key": "runs/run_123/recording.rrd", "sha256": "0" * 64}]}))

            code, payload = run_cli(
                "launch",
                "restore-check",
                "--d1-sql",
                str(d1_sql),
                "--object-manifest",
                str(manifest),
                "--object-root",
                str(root / "bucket"),
                "--json",
            )

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "launch_restore_check_failed")
        checks = {check["name"]: check for check in payload["error"]["details"]["checks"]}
        self.assertEqual(checks["object_integrity"]["status"], "fail")
        self.assertEqual(checks["object_integrity"]["code"], "restore_object_integrity_failed")

    def test_launch_backup_manifest_hashes_and_verifies_backup_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            d1_sql = root / "backup.sql"
            d1_sql.write_text("\n".join(f"CREATE TABLE {name} (id TEXT);" for name in sorted(cli_module.D1_RESTORE_REQUIRED_TABLES)))
            d1_hash = cli_module.sha256_file(d1_sql)
            object_root = root / "bucket"
            artifact = object_root / "runs" / "run_123" / "recording.rrd"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"RRD")
            manifest = root / "objects.json"
            manifest.write_text(
                json.dumps(
                    {
                        "objects": [
                            {
                                "key": "runs/run_123/recording.rrd",
                                "sha256": cli_module.sha256_file(artifact),
                                "size_bytes": artifact.stat().st_size,
                            }
                        ]
                    }
                )
            )
            out = root / "backup-manifest.json"

            code, payload = run_cli(
                "launch",
                "backup-manifest",
                "--d1-sql",
                str(d1_sql),
                "--object-manifest",
                str(manifest),
                "--object-root",
                str(object_root),
                "--required-object",
                "runs/run_123/recording.rrd",
                "--label",
                "final-production-backup",
                "--out",
                str(out),
                "--json",
            )
            written = json.loads(out.read_text())

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.backup-manifest")
        self.assertEqual(payload["result"]["schema_version"], "roborank.backup_manifest.v0")
        self.assertEqual(payload["result"]["counts"], {"pass": 2})
        self.assertEqual(written["backups"]["object_manifest"]["object_count"], 1)
        self.assertEqual(written["backups"]["d1_sql"]["sha256"], d1_hash)

    def test_launch_evidence_examples_verifies_required_scenarios(self) -> None:
        runs = {
            "run_schema": {
                "id": "run_schema",
                "metricsSchemaId": "schema_1",
                "resultStatus": "success",
                "tags": {"environment": {"canonicalId": "roborank/diff-drive-reach-target"}},
                "artifacts": [{"viewerCompatibilityState": "viewer_ok"}],
                "metrics": [{"name": "score", "valueNumber": 91}],
                "warnings": [],
            },
            "run_optional": {
                "id": "run_optional",
                "metricsSchemaId": "schema_1",
                "resultStatus": "success",
                "tags": {"environment": {"canonicalId": "roborank/diff-drive-reach-target"}},
                "artifacts": [{"viewerCompatibilityState": "viewer_ok"}],
                "metrics": [{"name": "score", "valueNumber": 88}],
                "warnings": [],
            },
            "run_unknown": {
                "id": "run_unknown",
                "resultStatus": "success",
                "tags": {"environment": {"canonicalId": "std/unknown"}},
                "artifacts": [{"viewerCompatibilityState": "viewer_ok"}],
            },
            "run_failed": {
                "id": "run_failed",
                "resultStatus": "failure",
                "tags": {"environment": {"canonicalId": "roborank/diff-drive-reach-target"}},
                "artifacts": [{"viewerCompatibilityState": "viewer_ok"}],
            },
            "run_viewer": {
                "id": "run_viewer",
                "resultStatus": "success",
                "tags": {"environment": {"canonicalId": "roborank/diff-drive-reach-target"}},
                "artifacts": [{"viewerCompatibilityState": "viewer_unknown"}],
            },
        }

        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(method, "GET")
            run_id = path.rsplit("/", 1)[-1]
            return runs[run_id]

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli(
                "launch",
                "evidence-examples",
                "--example",
                "schema_backed=run_schema",
                "--example",
                "missing_optional_metrics=run_optional",
                "--example",
                "std_unknown=run_unknown",
                "--example",
                "failed_result=run_failed",
                "--example",
                "viewer_incompatible=run_viewer",
                "--json",
            )

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.evidence-examples")
        checks = {check["name"]: check["status"] for check in payload["result"]["checks"]}
        self.assertEqual(set(checks), set(cli_module.CURATED_EVIDENCE_SCENARIOS))
        self.assertTrue(all(status == "pass" for status in checks.values()))

    def test_launch_evidence_examples_fails_missing_required_scenario(self) -> None:
        code, payload = run_cli("launch", "evidence-examples", "--json")

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "launch_evidence_examples_failed")

    def test_launch_privacy_scan_passes_public_evidence_surfaces(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(method, "GET")
            if path == "/api/explorer/runs":
                self.assertIsNone(api.token)
                self.assertEqual(kwargs["params"], {"limit": 3, "offset": 0})
                return {"runs": [{"id": "run_123"}, {"id": "run_456"}]}
            if path == "/api/evidence-runs/run_123":
                self.assertIsNone(api.token)
                return {"id": "run_123", "artifacts": [{"url": "/api/evidence-artifacts/runs/run_123/recording.rrd"}]}
            if path == "/api/evidence-runs/run_456":
                self.assertIsNone(api.token)
                return {"id": "run_456", "artifacts": []}
            if path == "/api/resources/robot/roborank/differential-drive-cube-v1":
                self.assertIsNone(api.token)
                return {"resource": {"canonicalId": "roborank/differential-drive-cube-v1", "summary": "public robot"}}
            if path == "/api/compare":
                self.assertEqual(kwargs["params"], {"run": ["run_123", "run_456"]})
                if api.token is None:
                    raise cli_module.RoboRankError("Authentication required.", code="auth_failed", exit_code=3, details={"status": 401})
                self.assertEqual(api.token, "token-ignored")
                return {"runs": [{"id": "run_123"}, {"id": "run_456"}]}
            raise AssertionError((method, path, kwargs))

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli(
                "--token",
                "token-ignored",
                "launch",
                "privacy-scan",
                "--strict",
                "--run-id",
                "run_123",
                "--run-id",
                "run_456",
                "--resource-kind",
                "robot",
                "--resource-id",
                "roborank/differential-drive-cube-v1",
                "--json",
            )

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.privacy-scan")
        self.assertFalse(payload["result"]["source_exposure"])
        checks = {check["name"]: check["status"] for check in payload["result"]["checks"]}
        self.assertEqual(checks, {name: "pass" for name in cli_module.GO_NO_GO_REQUIRED_PRIVACY_CHECKS})

    def test_launch_privacy_scan_strict_requires_token_for_compare_payload(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(method, "GET")
            if path == "/api/explorer/runs":
                return {"runs": [{"id": "run_123"}, {"id": "run_456"}]}
            if path == "/api/evidence-runs/run_123":
                return {"id": "run_123"}
            if path == "/api/evidence-runs/run_456":
                return {"id": "run_456"}
            if path == "/api/resources/robot/roborank/differential-drive-cube-v1":
                return {"resource": {"canonicalId": "roborank/differential-drive-cube-v1"}}
            if path == "/api/compare":
                raise cli_module.RoboRankError("Authentication required.", code="auth_failed", exit_code=3, details={"status": 401})
            raise AssertionError((method, path, kwargs))

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli(
                "launch",
                "privacy-scan",
                "--strict",
                "--run-id",
                "run_123",
                "--run-id",
                "run_456",
                "--resource-kind",
                "robot",
                "--resource-id",
                "roborank/differential-drive-cube-v1",
                "--json",
            )

        self.assertEqual(code, 9)
        checks = {check["name"]: check["status"] for check in payload["error"]["details"]["checks"]}
        self.assertEqual(checks["compare_anonymous_denied"], "pass")
        self.assertEqual(checks["compare_authenticated"], "skipped")

    def test_launch_privacy_scan_rejects_more_than_four_compare_runs(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(method, "GET")
            if path == "/api/explorer/runs":
                return {"runs": [{"id": "run_123"}, {"id": "run_456"}]}
            raise AssertionError((method, path, kwargs))

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli(
                "launch",
                "privacy-scan",
                "--compare-run",
                "run_1",
                "--compare-run",
                "run_2",
                "--compare-run",
                "run_3",
                "--compare-run",
                "run_4",
                "--compare-run",
                "run_5",
                "--json",
            )

        self.assertEqual(code, 9)
        checks = {check["name"]: check for check in payload["error"]["details"]["checks"]}
        self.assertEqual(checks["compare_anonymous_denied"]["status"], "fail")
        self.assertEqual(checks["compare_anonymous_denied"]["code"], "compare_too_many_runs")

    def test_launch_privacy_scan_fails_when_public_payload_exposes_source(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            if path == "/api/explorer/runs":
                return {"runs": [{"id": "run_secret"}]}
            if path == "/api/evidence-runs/run_secret":
                return {"id": "run_secret", "policy_source": "class RobotPolicy:\n    pass\n"}
            if path == "/api/resources/robot/roborank/differential-drive-cube-v1":
                return {"resource": {"canonicalId": "roborank/differential-drive-cube-v1"}}
            if path == "/api/compare":
                return {"runs": [{"id": "run_secret"}, {"id": "run_other"}]}
            raise AssertionError((method, path, kwargs))

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli(
                "launch",
                "privacy-scan",
                "--strict",
                "--run-id",
                "run_secret",
                "--run-id",
                "run_other",
                "--resource-kind",
                "robot",
                "--resource-id",
                "roborank/differential-drive-cube-v1",
                "--json",
            )

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "privacy_scan_failed")
        self.assertTrue(payload["error"]["details"]["source_exposure"])
        checks = {check["name"]: check for check in payload["error"]["details"]["checks"]}
        self.assertEqual(checks["run_detail_public"]["status"], "fail")
        self.assertEqual(checks["run_detail_public"]["code"], "privacy_source_exposure")

    def test_launch_schema_guard_proves_missing_and_invalid_metrics_rejected(self) -> None:
        schema = {
            "type": "object",
            "required": ["metric_kind", "score"],
            "properties": {
                "metric_kind": {"const": "navigation"},
                "score": {"type": "number"},
            },
        }
        seen_metrics: list[dict | None] = []

        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(api.token, "token-123")
            if method == "GET" and path == "/api/environments/roborank/diff-drive-reach-target/metrics-schema":
                return {"schema": {"id": "schema_1", "schemaHash": "sha256:abc", "jsonSchema": schema}}
            if method == "POST" and path == "/api/evidence-runs/validate":
                body = kwargs["body"]
                self.assertEqual(body["metadata"]["tags"]["environment"], "roborank/diff-drive-reach-target")
                self.assertEqual(body["metadata"]["tags"]["policy"], "benkant/schema-guard-policy")
                seen_metrics.append(body.get("metrics"))
                if body.get("metrics") is None:
                    raise cli_module.RoboRankError("metrics.json is required for this environment.", code="api_error", exit_code=7, details={"status": 400})
                if body.get("metrics") == {}:
                    raise cli_module.RoboRankError("metrics.json did not validate.", code="api_error", exit_code=7, details={"status": 400})
            raise AssertionError((method, path, kwargs))

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli(
                "--token",
                "token-123",
                "launch",
                "schema-guard",
                "--policy",
                "benkant/schema-guard-policy",
                "--json",
            )

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.schema-guard")
        checks = {check["name"]: check["status"] for check in payload["result"]["checks"]}
        self.assertEqual(checks, {name: "pass" for name in cli_module.GO_NO_GO_REQUIRED_SCHEMA_GUARD_CHECKS})
        self.assertEqual(seen_metrics, [None, {}])

    def test_launch_schema_guard_fails_when_missing_metrics_are_accepted(self) -> None:
        schema = {
            "type": "object",
            "required": ["score"],
            "properties": {"score": {"type": "number"}},
        }

        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            if method == "GET" and path == "/api/environments/roborank/diff-drive-reach-target/metrics-schema":
                return {"schema": {"id": "schema_1", "schemaHash": "sha256:abc", "jsonSchema": schema}}
            if method == "POST" and path == "/api/evidence-runs/validate":
                if kwargs["body"].get("metrics") is None:
                    return {"valid": True}
                raise cli_module.RoboRankError("metrics.json did not validate.", code="api_error", exit_code=7, details={"status": 400})
            raise AssertionError((method, path, kwargs))

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli(
                "--token",
                "token-123",
                "launch",
                "schema-guard",
                "--policy",
                "benkant/schema-guard-policy",
                "--json",
            )

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "schema_guard_failed")
        checks = {check["name"]: check for check in payload["error"]["details"]["checks"]}
        self.assertEqual(checks["missing_metrics_rejected"]["status"], "fail")
        self.assertEqual(checks["missing_metrics_rejected"]["code"], "missing_metrics_accepted")

    def test_launch_resource_guard_proves_robot_environment_typos_rejected(self) -> None:
        seen_tags: list[dict] = []

        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(api.token, "token-123")
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/api/evidence-runs/validate")
            tags = kwargs["body"]["metadata"]["tags"]
            seen_tags.append(tags)
            if tags["robot"] == "qa/missing-robot-typo" or tags["environment"] == "qa/missing-environment-typo":
                raise cli_module.RoboRankError(
                    "Robot and environment resources must exist before evidence upload.",
                    code="api_error",
                    exit_code=7,
                    details={"status": 400},
                )
            return {
                "valid": True,
                "resolved": {
                    "robot": {"canonicalId": tags["robot"]},
                    "environment": {"canonicalId": tags["environment"]},
                    "policy": {"canonicalId": tags["policy"]},
                },
            }

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli(
                "--token",
                "token-123",
                "launch",
                "resource-guard",
                "--policy",
                "qa/resource-guard-policy",
                "--allow-new-policy",
                "--json",
            )

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.resource-guard")
        checks = {check["name"]: check["status"] for check in payload["result"]["checks"]}
        self.assertEqual(checks, {name: "pass" for name in cli_module.GO_NO_GO_REQUIRED_RESOURCE_GUARD_CHECKS})
        self.assertEqual([tags["robot"] for tags in seen_tags], ["roborank/differential-drive-cube-v1", "qa/missing-robot-typo", "roborank/differential-drive-cube-v1"])

    def test_launch_resource_guard_fails_when_missing_robot_is_accepted(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            tags = kwargs["body"]["metadata"]["tags"]
            if tags["environment"] == "qa/missing-environment-typo":
                raise cli_module.RoboRankError(
                    "Robot and environment resources must exist before evidence upload.",
                    code="api_error",
                    exit_code=7,
                    details={"status": 400},
                )
            return {"valid": True, "resolved": {"robot": {"canonicalId": tags["robot"]}}}

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli(
                "--token",
                "token-123",
                "launch",
                "resource-guard",
                "--policy",
                "qa/resource-guard-policy",
                "--allow-new-policy",
                "--json",
            )

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "resource_guard_failed")
        checks = {check["name"]: check for check in payload["error"]["details"]["checks"]}
        self.assertEqual(checks["missing_robot_rejected"]["status"], "fail")
        self.assertEqual(checks["missing_robot_rejected"]["code"], "missing_robot_accepted")

    def test_launch_trust_actions_endorses_and_disputes_tagged_evidence(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(api.token, "owner-token")
            calls.append((method, path))
            if method == "GET" and path == "/api/resources/robot/roborank/differential-drive-cube-v1":
                return {"resource": {"canonicalId": "roborank/differential-drive-cube-v1", "ownerUserId": "owner_1"}}
            if method == "GET" and path == "/api/evidence-runs/run_123":
                previous_actions = {call_path.rsplit("/", 1)[-1] for call_method, call_path in calls if call_method == "POST"}
                labels = ["self_reported"]
                if "endorse" in previous_actions and "dispute" not in previous_actions:
                    labels.append("resource_owner_endorsed")
                if "dispute" in previous_actions:
                    labels.append("disputed")
                return {"id": "run_123", "trustLabels": labels}
            if method == "GET" and path == "/api/explorer/runs":
                self.assertEqual(
                    kwargs["params"],
                    {
                        "limit": 50,
                        "offset": 0,
                        "robot": "roborank/differential-drive-cube-v1",
                        "resource_context_kind": "robot",
                        "resource_context": "roborank/differential-drive-cube-v1",
                    },
                )
                previous_actions = {call_path.rsplit("/", 1)[-1] for call_method, call_path in calls if call_method == "POST"}
                state = "community"
                if "dispute" in previous_actions:
                    state = "disputed"
                elif "endorse" in previous_actions:
                    state = "endorsed"
                return {
                    "runs": [
                        {"id": "run_other_endorsed", "resourceTrustState": "endorsed"},
                        {"id": "run_123", "resourceTrustState": state},
                    ]
                }
            if method == "POST" and path == "/api/resources/robot/roborank/differential-drive-cube-v1/runs/run_123/endorse":
                self.assertEqual(kwargs["body"]["note"], "looks valid")
                return {"ok": True}
            if method == "POST" and path == "/api/resources/robot/roborank/differential-drive-cube-v1/runs/run_123/dispute":
                self.assertEqual(kwargs["body"]["reason"], "wrong tag")
                return {"ok": True}
            raise AssertionError((method, path, kwargs))

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli(
                "--token",
                "owner-token",
                "--yes",
                "launch",
                "trust-actions",
                "--resource-kind",
                "robot",
                "--resource-id",
                "roborank/differential-drive-cube-v1",
                "--run-id",
                "run_123",
                "--note",
                "looks valid",
                "--reason",
                "wrong tag",
                "--json",
            )

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.trust-actions")
        checks = {check["name"]: check["status"] for check in payload["result"]["checks"]}
        self.assertEqual(checks, {name: "pass" for name in cli_module.GO_NO_GO_REQUIRED_TRUST_ACTION_CHECKS})

    def test_launch_trust_actions_requires_confirmation(self) -> None:
        code, payload = run_cli(
            "--token",
            "owner-token",
            "launch",
            "trust-actions",
            "--resource-kind",
            "robot",
            "--resource-id",
            "roborank/differential-drive-cube-v1",
            "--run-id",
            "run_123",
            "--json",
        )

        self.assertEqual(code, 2)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "confirmation_required")

    def test_launch_trust_actions_fails_when_labels_do_not_update(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            if method == "GET" and path == "/api/resources/robot/roborank/differential-drive-cube-v1":
                return {"resource": {"canonicalId": "roborank/differential-drive-cube-v1"}}
            if method == "GET" and path == "/api/evidence-runs/run_123":
                return {"id": "run_123", "trustLabels": ["self_reported"]}
            if method == "GET" and path == "/api/explorer/runs":
                previous_actions = {call_path.rsplit("/", 1)[-1] for call_method, call_path in fake_request_json.calls if call_method == "POST"}
                state = "disputed" if "dispute" in previous_actions else "endorsed"
                return {"runs": [{"id": "run_123", "resourceTrustState": state}]}
            if method == "POST" and path.endswith(("/endorse", "/dispute")):
                fake_request_json.calls.append((method, path))
                return {"ok": True}
            raise AssertionError((method, path, kwargs))

        fake_request_json.calls = []  # type: ignore[attr-defined]

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli(
                "--token",
                "owner-token",
                "--yes",
                "launch",
                "trust-actions",
                "--resource-kind",
                "robot",
                "--resource-id",
                "roborank/differential-drive-cube-v1",
                "--run-id",
                "run_123",
                "--json",
            )

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "trust_actions_failed")
        checks = {check["name"]: check for check in payload["error"]["details"]["checks"]}
        self.assertEqual(checks["trust_labels"]["status"], "fail")
        self.assertEqual(checks["trust_labels"]["code"], "trust_labels_missing")

    def test_launch_browser_qa_generates_report_from_screenshots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "browser-qa.json"
            args = ["launch", "browser-qa", "--target", "staging", "--out", str(out), "--json"]
            for surface in sorted(cli_module.GO_NO_GO_REQUIRED_BROWSER_SURFACES):
                screenshot = root / f"{surface}.png"
                screenshot.write_bytes(f"png-{surface}".encode("utf-8"))
                args.extend(["--surface", f"{surface}={screenshot}"])

            code, payload = run_cli(*args)
            written = json.loads(out.read_text())

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.browser-qa")
        self.assertEqual(payload["result"]["schema_version"], "roborank.browser_qa.v0")
        self.assertEqual(set(payload["result"]["surfaces"]), set(cli_module.GO_NO_GO_REQUIRED_BROWSER_SURFACES))
        self.assertTrue(all(value["status"] == "pass" for value in payload["result"]["surfaces"].values()))
        self.assertEqual(written["counts"], {"pass": len(cli_module.GO_NO_GO_REQUIRED_BROWSER_SURFACES)})

    def test_launch_browser_qa_fails_when_required_surface_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            screenshot = root / "problem.png"
            screenshot.write_bytes(b"PNG")
            code, payload = run_cli("launch", "browser-qa", "--surface", f"problem_page={screenshot}", "--json")

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "browser_qa_failed")
        checks = {check["name"]: check for check in payload["error"]["details"]["checks"]}
        self.assertEqual(checks["problem_page"]["status"], "pass")
        self.assertEqual(checks["explorer_desktop"]["code"], "browser_qa_surface_missing")

    def test_launch_known_issues_generates_empty_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "known-issues.json"
            code, payload = run_cli("launch", "known-issues", "--out", str(out), "--json")
            written = json.loads(out.read_text())

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.known-issues")
        self.assertEqual(payload["result"]["schema_version"], "roborank.known_issues.v0")
        self.assertEqual(payload["result"]["counts"]["open_blockers"], 0)
        self.assertEqual(written["issues"], [])

    def test_launch_known_issues_fails_for_open_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "known-issues.json"
            code, payload = run_cli(
                "launch",
                "known-issues",
                "--issue",
                '{"id":"privacy-bug","severity":"critical","status":"open"}',
                "--out",
                str(out),
                "--json",
            )
            written = json.loads(out.read_text())

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "known_issues_blocked")
        self.assertEqual(written["counts"]["open_blockers"], 1)

    def test_launch_triage_builds_post_launch_board_from_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            watch = root / "watch.json"
            progress = root / "progress.json"
            out = root / "triage.json"
            watch.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.launch_watch.v0",
                        "checks": [
                            {"name": "storage_failures", "status": "fail", "message": "storage failures 7d = 1, threshold = 0"},
                            {"name": "upload_failures", "status": "pass"},
                        ],
                        "recent_events": [{"severity": "error", "message": "artifact download failed", "type": "artifact_download"}],
                    }
                )
            )
            progress.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.migration_progress.v0",
                        "decision": "pause",
                        "pause_reasons": ["repeated failure keys reached threshold 3: storage_write_failed"],
                        "counts": {"failure_count": 3, "migration_remaining": 12},
                    }
                )
            )

            code, payload = run_cli(
                "launch",
                "triage",
                "--watch-report",
                str(watch),
                "--migration-progress-report",
                str(progress),
                "--out",
                str(out),
                "--json",
            )
            written = json.loads(out.read_text())

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "post_launch_triage_blocked")
        details = payload["error"]["details"]
        self.assertEqual(details["schema_version"], "roborank.post_launch_triage.v0")
        self.assertEqual(details["decision"], "investigate")
        self.assertIn("watch-storage-failures", details["columns"]["launch_blockers"])
        self.assertIn("migration-progress-pause", details["columns"]["migration"])
        self.assertEqual(written["counts"]["open_blockers"], details["counts"]["open_blockers"])

    def test_launch_triage_allows_nonblocking_board(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            known = root / "known.json"
            out = root / "triage.json"
            known.write_text(json.dumps({"issues": [{"id": "minor-ui", "severity": "p2", "status": "open", "title": "Minor UI polish"}]}))

            code, payload = run_cli(
                "launch",
                "triage",
                "--known-issues",
                str(known),
                "--out",
                str(out),
                "--json",
            )

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.triage")
        self.assertEqual(payload["result"]["decision"], "monitor")
        self.assertEqual(payload["result"]["counts"]["open_blockers"], 0)
        self.assertIn("minor-ui", payload["result"]["columns"]["known"])

    def test_launch_signoff_generates_required_approvals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "signoff.json"
            code, payload = run_cli(
                "launch",
                "signoff",
                "--data-storage",
                "approved",
                "--backend",
                "approved",
                "--frontend",
                "approved",
                "--product",
                "go",
                "--out",
                str(out),
                "--json",
            )
            written = json.loads(out.read_text())

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.signoff")
        self.assertEqual(payload["result"]["schema_version"], "roborank.launch_signoff.v0")
        self.assertEqual(payload["result"]["missing"], [])
        self.assertEqual(written["signoffs"]["product"], "go")

    def test_launch_signoff_fails_when_missing_required_role(self) -> None:
        code, payload = run_cli(
            "launch",
            "signoff",
            "--data-storage",
            "approved",
            "--backend",
            "approved",
            "--frontend",
            "approved",
            "--json",
        )

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "launch_signoff_missing")
        self.assertEqual(payload["error"]["details"]["missing"], ["product"])

    def test_launch_cli_release_builds_and_smoke_tests_artifacts(self) -> None:
        def fake_run(command, **kwargs):
            if command[1] == "build":
                dist_dir = Path(command[-1])
                dist_dir.mkdir(parents=True, exist_ok=True)
                (dist_dir / "roborank-0.1.0.tar.gz").write_bytes(b"sdist")
                (dist_dir / "roborank-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
                return cli_module.subprocess.CompletedProcess(command, 0, stdout="built\n", stderr="")
            if command[1] == "run":
                return cli_module.subprocess.CompletedProcess(command, 0, stdout=json.dumps({"ok": True, "command": "prime"}), stderr="")
            raise AssertionError(command)

        with tempfile.TemporaryDirectory() as directory, patch.object(cli_module.shutil, "which", lambda name: "uv"), patch.object(
            cli_module.subprocess, "run", fake_run
        ):
            root = Path(directory)
            cli_root = root / "cli"
            package = cli_root / "roborank"
            package.mkdir(parents=True)
            (cli_root / "README.md").write_text("CLI\n")
            (cli_root / "pyproject.toml").write_text(
                """
[project]
name = "roborank"
version = "0.1.0"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
""".strip()
                + "\n"
            )
            (package / "__init__.py").write_text('__version__ = "0.1.0"\n')
            dist = root / "dist"
            out = root / "cli-release.json"
            code, payload = run_cli(
                "launch",
                "cli-release",
                "--cli-root",
                str(cli_root),
                "--dist-dir",
                str(dist),
                "--out",
                str(out),
                "--json",
            )
            written = json.loads(out.read_text())

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.cli-release")
        self.assertEqual(payload["result"]["schema_version"], "roborank.cli_release.v0")
        self.assertTrue(payload["result"]["publish_ready"])
        self.assertEqual(payload["result"]["counts"], {"pass": 4})
        self.assertEqual(len(written["artifacts"]), 2)

    def test_launch_cli_release_fails_on_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(cli_module.shutil, "which", lambda name: None):
            root = Path(directory)
            cli_root = root / "cli"
            package = cli_root / "roborank"
            package.mkdir(parents=True)
            (cli_root / "pyproject.toml").write_text('[project]\nname = "roborank"\nversion = "0.1.0"\n')
            (package / "__init__.py").write_text('__version__ = "0.2.0"\n')
            code, payload = run_cli("launch", "cli-release", "--cli-root", str(cli_root), "--dist-dir", str(root / "dist"), "--json")

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "cli_release_failed")
        checks = {check["name"]: check for check in payload["error"]["details"]["checks"]}
        self.assertEqual(checks["version_consistency"]["status"], "fail")

    def test_launch_preflight_generates_local_readiness_report(self) -> None:
        seen: list[tuple[str, Path, int]] = []

        def fake_run_launch_preflight_check(name: str, root: Path, timeout: int) -> dict:
            seen.append((name, root, timeout))
            return {"name": name, "status": "pass", "duration_ms": 0, "command": cli_module.LAUNCH_PREFLIGHT_COMMANDS[name]}

        with tempfile.TemporaryDirectory() as directory, patch.object(cli_module, "run_launch_preflight_check", fake_run_launch_preflight_check):
            root = Path(directory)
            out = root / "preflight.json"
            code, payload = run_cli(
                "launch",
                "preflight",
                "--target",
                "staging",
                "--repo-root",
                str(root),
                "--check",
                "npm_test",
                "--check",
                "d1_schema_import",
                "--timeout",
                "7",
                "--out",
                str(out),
                "--json",
            )
            written = json.loads(out.read_text())

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.preflight")
        self.assertEqual(payload["result"]["schema_version"], "roborank.launch_preflight.v0")
        self.assertEqual([item[0] for item in seen], ["npm_test", "d1_schema_import"])
        self.assertTrue(all(item[2] == 7 for item in seen))
        self.assertEqual(written["counts"], {"pass": 2})

    def test_launch_preflight_fails_on_failed_check(self) -> None:
        def fake_run_launch_preflight_check(name: str, root: Path, timeout: int) -> dict:
            return {
                "name": name,
                "status": "fail",
                "code": "preflight_command_failed",
                "message": "typecheck failed",
                "duration_ms": 0,
                "command": cli_module.LAUNCH_PREFLIGHT_COMMANDS[name],
                "returncode": 2,
            }

        with patch.object(cli_module, "run_launch_preflight_check", fake_run_launch_preflight_check):
            code, payload = run_cli("launch", "preflight", "--check", "cloudflare_typecheck", "--json")

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "launch_preflight_failed")
        self.assertEqual(payload["error"]["details"]["counts"], {"fail": 1})

    def test_launch_watch_passes_clean_admin_dashboard(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(method, "GET")
            self.assertEqual(path, "/api/admin/dashboard")
            self.assertEqual(kwargs["params"], {"limit": 1, "offset": 0})
            return {
                "stats": {
                    "totalEvidenceRuns": 12,
                    "uploadFailures7d": 0,
                    "metricsValidationFailures7d": 0,
                    "storageFailures7d": 0,
                    "viewerFailures7d": 0,
                    "explorerSlowEvents7d": 0,
                    "migrationFailures7d": 0,
                    "rateLimitEvents7d": 0,
                },
                "monitoring": {"recentEvents": [{"severity": "info", "message": "ok"}]},
            }

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli("launch", "watch", "--min-evidence-runs", "1", "--json")

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.watch")
        self.assertEqual(payload["result"]["schema_version"], "roborank.launch_watch.v0")
        self.assertTrue(all(check["status"] == "pass" for check in payload["result"]["checks"]))

    def test_launch_watch_fails_on_storage_failure(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            return {
                "stats": {
                    "totalEvidenceRuns": 12,
                    "uploadFailures7d": 0,
                    "metricsValidationFailures7d": 0,
                    "storageFailures7d": 1,
                    "viewerFailures7d": 0,
                    "explorerSlowEvents7d": 0,
                    "migrationFailures7d": 0,
                    "rateLimitEvents7d": 0,
                },
                "monitoring": {"recentEvents": []},
            }

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli("launch", "watch", "--json")

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "launch_watch_failed")
        checks = {check["name"]: check for check in payload["error"]["details"]["checks"]}
        self.assertEqual(checks["storage_failures"]["status"], "fail")

    def test_launch_watch_fails_on_recent_error_event(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            return {
                "stats": {
                    "uploadFailures7d": 0,
                    "metricsValidationFailures7d": 0,
                    "storageFailures7d": 0,
                    "viewerFailures7d": 0,
                    "explorerSlowEvents7d": 0,
                    "migrationFailures7d": 0,
                    "rateLimitEvents7d": 0,
                },
                "monitoring": {"recentEvents": [{"severity": "error", "message": "storage write failed"}]},
            }

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli("launch", "watch", "--json")

        self.assertEqual(code, 9)
        checks = {check["name"]: check for check in payload["error"]["details"]["checks"]}
        self.assertEqual(checks["recent_monitoring_event_severity"]["status"], "fail")

    def test_launch_go_no_go_passes_with_required_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight = root / "preflight.json"
            smoke = root / "smoke.json"
            restore = root / "restore.json"
            examples = root / "examples.json"
            privacy = root / "privacy.json"
            schema_guard = root / "schema-guard.json"
            trust_actions = root / "trust-actions.json"
            resource_guard = root / "resource-guard.json"
            migration = root / "migration.json"
            browser = root / "browser.json"
            issues = root / "issues.json"
            signoff = root / "signoff.json"
            out = root / "go-no-go.json"
            preflight.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.launch_preflight.v0",
                        "checks": [{"name": name, "status": "pass"} for name in sorted(cli_module.LAUNCH_PREFLIGHT_COMMANDS)],
                    }
                )
            )
            smoke.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.launch_smoke.v0",
                        "checks": [{"name": name, "status": "pass"} for name in sorted(cli_module.GO_NO_GO_REQUIRED_SMOKE_CHECKS)],
                    }
                )
            )
            restore.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.launch_restore_check.v0",
                        "checks": [
                            {"name": "d1_sql_import", "status": "pass"},
                            {"name": "object_integrity", "status": "pass"},
                            {"name": "api_smoke_report", "status": "pass"},
                        ],
                    }
                )
            )
            examples.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.launch_evidence_examples.v0",
                        "checks": [{"name": name, "status": "pass"} for name in sorted(cli_module.CURATED_EVIDENCE_SCENARIOS)],
                    }
                )
            )
            self.write_passing_privacy_report(privacy)
            self.write_passing_schema_guard_report(schema_guard)
            self.write_passing_trust_actions_report(trust_actions)
            self.write_passing_resource_guard_report(resource_guard)
            migration.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.migration_dry_run_report.v0",
                        "privacy_checks": [{"name": "no_preserved_source_code_in_reports", "status": "pass"}],
                        "validation_failures": [],
                        "rollback_plan": {
                            "d1_restore_point": "d1-backup",
                            "object_storage_restore_point": "r2-snapshot",
                            "worker_rollback_version": "worker-v1",
                            "frontend_rollback_version": "frontend-v1",
                            "migration_pause_resume_state_path": "migration-state.json",
                        },
                    }
                )
            )
            browser.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.browser_qa.v0",
                        "surfaces": {name: "pass" for name in sorted(cli_module.GO_NO_GO_REQUIRED_BROWSER_SURFACES)},
                    }
                )
            )
            issues.write_text(json.dumps({"issues": [{"id": "minor", "severity": "p2", "status": "open"}]}))
            signoff.write_text(
                json.dumps(
                    {
                        "signoffs": {
                            "data_storage": {"status": "approved"},
                            "backend": "approved",
                            "frontend": True,
                            "product": "go",
                        }
                    }
                )
            )

            code, payload = run_cli(
                "launch",
                "go-no-go",
                "--target",
                "staging",
                "--preflight-report",
                str(preflight),
                "--smoke-report",
                str(smoke),
                "--restore-report",
                str(restore),
                "--evidence-examples-report",
                str(examples),
                "--privacy-report",
                str(privacy),
                "--schema-guard-report",
                str(schema_guard),
                "--trust-actions-report",
                str(trust_actions),
                "--resource-guard-report",
                str(resource_guard),
                "--migration-report",
                str(migration),
                "--browser-report",
                str(browser),
                "--known-issues",
                str(issues),
                "--signoff",
                str(signoff),
                "--out",
                str(out),
                "--json",
            )
            written = json.loads(out.read_text())

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.go-no-go")
        self.assertEqual(payload["result"]["decision"], "go")
        self.assertEqual(written["decision"], "go")

    def test_launch_go_no_go_fails_for_failed_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight = root / "preflight.json"
            smoke = root / "smoke.json"
            restore = root / "restore.json"
            examples = root / "examples.json"
            privacy = root / "privacy.json"
            schema_guard = root / "schema-guard.json"
            trust_actions = root / "trust-actions.json"
            resource_guard = root / "resource-guard.json"
            migration = root / "migration.json"
            browser = root / "browser.json"
            issues = root / "issues.json"
            signoff = root / "signoff.json"
            preflight.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.launch_preflight.v0",
                        "checks": [
                            {"name": "npm_test", "status": "pass"},
                            {"name": "cloudflare_typecheck", "status": "fail"},
                            {"name": "cloudflare_build", "status": "pass"},
                            {"name": "d1_schema_import", "status": "pass"},
                        ],
                    }
                )
            )
            smoke.write_text(json.dumps({"schema_version": "roborank.launch_smoke.v0", "checks": [{"name": name, "status": "pass"} for name in sorted(cli_module.GO_NO_GO_REQUIRED_SMOKE_CHECKS)]}))
            restore.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.launch_restore_check.v0",
                        "checks": [
                            {"name": "d1_sql_import", "status": "pass"},
                            {"name": "object_integrity", "status": "pass"},
                            {"name": "api_smoke_report", "status": "pass"},
                        ],
                    }
                )
            )
            examples.write_text(json.dumps({"schema_version": "roborank.launch_evidence_examples.v0", "checks": [{"name": name, "status": "pass"} for name in sorted(cli_module.CURATED_EVIDENCE_SCENARIOS)]}))
            self.write_passing_privacy_report(privacy)
            self.write_passing_schema_guard_report(schema_guard)
            self.write_passing_trust_actions_report(trust_actions)
            self.write_passing_resource_guard_report(resource_guard)
            migration.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.migration_dry_run_report.v0",
                        "privacy_checks": [{"name": "no_preserved_source_code_in_reports", "status": "pass"}],
                        "validation_failures": [],
                        "rollback_plan": {
                            "d1_restore_point": "d1-backup",
                            "object_storage_restore_point": "r2-snapshot",
                            "worker_rollback_version": "worker-v1",
                            "frontend_rollback_version": "frontend-v1",
                            "migration_pause_resume_state_path": "migration-state.json",
                        },
                    }
                )
            )
            browser.write_text(json.dumps({"schema_version": "roborank.browser_qa.v0", "surfaces": {name: "pass" for name in sorted(cli_module.GO_NO_GO_REQUIRED_BROWSER_SURFACES)}}))
            issues.write_text(json.dumps({"issues": []}))
            signoff.write_text(json.dumps({"signoffs": {"data_storage": "approved", "backend": "approved", "frontend": "approved", "product": "approved"}}))

            code, payload = run_cli(
                "launch",
                "go-no-go",
                "--preflight-report",
                str(preflight),
                "--smoke-report",
                str(smoke),
                "--restore-report",
                str(restore),
                "--evidence-examples-report",
                str(examples),
                "--privacy-report",
                str(privacy),
                "--schema-guard-report",
                str(schema_guard),
                "--trust-actions-report",
                str(trust_actions),
                "--resource-guard-report",
                str(resource_guard),
                "--migration-report",
                str(migration),
                "--browser-report",
                str(browser),
                "--known-issues",
                str(issues),
                "--signoff",
                str(signoff),
                "--json",
            )

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        checks = {check["name"]: check for check in payload["error"]["details"]["checks"]}
        self.assertEqual(checks["preflight_report"]["status"], "fail")
        self.assertEqual(checks["preflight_report"]["evidence"]["failed"], ["cloudflare_typecheck"])

    def test_launch_go_no_go_fails_for_open_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight = root / "preflight.json"
            smoke = root / "smoke.json"
            restore = root / "restore.json"
            examples = root / "examples.json"
            privacy = root / "privacy.json"
            schema_guard = root / "schema-guard.json"
            trust_actions = root / "trust-actions.json"
            resource_guard = root / "resource-guard.json"
            migration = root / "migration.json"
            browser = root / "browser.json"
            issues = root / "issues.json"
            signoff = root / "signoff.json"
            preflight.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.launch_preflight.v0",
                        "checks": [{"name": name, "status": "pass"} for name in sorted(cli_module.LAUNCH_PREFLIGHT_COMMANDS)],
                    }
                )
            )
            smoke.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.launch_smoke.v0",
                        "checks": [{"name": name, "status": "pass"} for name in sorted(cli_module.GO_NO_GO_REQUIRED_SMOKE_CHECKS)],
                    }
                )
            )
            restore.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.launch_restore_check.v0",
                        "checks": [
                            {"name": "d1_sql_import", "status": "pass"},
                            {"name": "object_integrity", "status": "pass"},
                            {"name": "api_smoke_report", "status": "pass"},
                        ],
                    }
                )
            )
            examples.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.launch_evidence_examples.v0",
                        "checks": [{"name": name, "status": "pass"} for name in sorted(cli_module.CURATED_EVIDENCE_SCENARIOS)],
                    }
                )
            )
            self.write_passing_privacy_report(privacy)
            self.write_passing_schema_guard_report(schema_guard)
            self.write_passing_trust_actions_report(trust_actions)
            self.write_passing_resource_guard_report(resource_guard)
            migration.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.migration_dry_run_report.v0",
                        "privacy_checks": [{"name": "no_preserved_source_code_in_reports", "status": "pass"}],
                        "validation_failures": [],
                        "rollback_plan": {
                            "d1_restore_point": "d1-backup",
                            "object_storage_restore_point": "r2-snapshot",
                            "worker_rollback_version": "worker-v1",
                            "frontend_rollback_version": "frontend-v1",
                            "migration_pause_resume_state_path": "migration-state.json",
                        },
                    }
                )
            )
            browser.write_text(json.dumps({"schema_version": "roborank.browser_qa.v0", "surfaces": {name: "pass" for name in sorted(cli_module.GO_NO_GO_REQUIRED_BROWSER_SURFACES)}}))
            issues.write_text(json.dumps({"issues": [{"id": "privacy-bug", "severity": "critical", "status": "open"}]}))
            signoff.write_text(json.dumps({"signoffs": {"data_storage": "approved", "backend": "approved", "frontend": "approved", "product": "approved"}}))

            code, payload = run_cli(
                "launch",
                "go-no-go",
                "--preflight-report",
                str(preflight),
                "--smoke-report",
                str(smoke),
                "--restore-report",
                str(restore),
                "--evidence-examples-report",
                str(examples),
                "--privacy-report",
                str(privacy),
                "--schema-guard-report",
                str(schema_guard),
                "--trust-actions-report",
                str(trust_actions),
                "--resource-guard-report",
                str(resource_guard),
                "--migration-report",
                str(migration),
                "--browser-report",
                str(browser),
                "--known-issues",
                str(issues),
                "--signoff",
                str(signoff),
                "--json",
            )

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "launch_no_go")
        checks = {check["name"]: check for check in payload["error"]["details"]["checks"]}
        self.assertEqual(checks["known_issues"]["status"], "fail")

    def test_launch_cutover_releases_with_required_production_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            go_no_go = root / "go-no-go.json"
            smoke = root / "production-smoke.json"
            privacy = root / "production-privacy.json"
            schema_guard = root / "production-schema-guard.json"
            resource_guard = root / "production-resource-guard.json"
            backup = root / "backup.json"
            restore = root / "restore.json"
            progress = root / "progress.json"
            watch = root / "watch.json"
            triage = root / "triage.json"
            cli_release = root / "cli-release.json"
            out = root / "production-release.json"
            go_no_go.write_text(json.dumps({"schema_version": "roborank.launch_go_no_go.v0", "decision": "go", "checks": [{"name": "all", "status": "pass"}]}))
            smoke.write_text(json.dumps({"schema_version": "roborank.launch_smoke.v0", "checks": [{"name": name, "status": "pass"} for name in sorted(cli_module.GO_NO_GO_REQUIRED_SMOKE_CHECKS)]}))
            self.write_passing_privacy_report(privacy)
            self.write_passing_schema_guard_report(schema_guard)
            self.write_passing_resource_guard_report(resource_guard)
            backup.write_text(json.dumps({"schema_version": "roborank.backup_manifest.v0", "checks": [{"name": "d1_sql_import", "status": "pass"}, {"name": "object_integrity", "status": "pass"}]}))
            restore.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.launch_restore_check.v0",
                        "checks": [
                            {"name": "d1_sql_import", "status": "pass"},
                            {"name": "object_integrity", "status": "pass"},
                            {"name": "api_smoke_report", "status": "pass"},
                        ],
                    }
                )
            )
            progress.write_text(json.dumps({"schema_version": "roborank.migration_progress.v0", "decision": "continue", "pause_recommended": False}))
            watch.write_text(json.dumps({"schema_version": "roborank.launch_watch.v0", "checks": [{"name": "storage_failures", "status": "pass"}]}))
            triage.write_text(json.dumps({"schema_version": "roborank.post_launch_triage.v0", "decision": "monitor", "counts": {"open_blockers": 0}}))
            cli_release.write_text(json.dumps({"schema_version": "roborank.cli_release.v0", "publish_ready": True, "checks": [{"name": "entrypoint_smoke", "status": "pass"}]}))

            code, payload = run_cli(
                "launch",
                "cutover",
                "--go-no-go-report",
                str(go_no_go),
                "--production-smoke-report",
                str(smoke),
                "--production-privacy-report",
                str(privacy),
                "--production-schema-guard-report",
                str(schema_guard),
                "--production-resource-guard-report",
                str(resource_guard),
                "--backup-report",
                str(backup),
                "--restore-report",
                str(restore),
                "--migration-progress-report",
                str(progress),
                "--watch-report",
                str(watch),
                "--triage-report",
                str(triage),
                "--cli-release-report",
                str(cli_release),
                "--worker-version",
                "worker-v2",
                "--frontend-version",
                "frontend-v2",
                "--backend-version",
                "backend-v2",
                "--d1-backup",
                "d1-backup-final",
                "--object-backup",
                "r2-snapshot-final",
                "--out",
                str(out),
                "--json",
            )
            written = json.loads(out.read_text())

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "launch.cutover")
        self.assertEqual(payload["result"]["schema_version"], "roborank.production_release.v0")
        self.assertEqual(payload["result"]["decision"], "released")
        self.assertEqual(written["deployment"]["worker_version"], "worker-v2")

    def test_launch_cutover_holds_on_failed_watch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            go_no_go = root / "go-no-go.json"
            smoke = root / "production-smoke.json"
            privacy = root / "production-privacy.json"
            schema_guard = root / "production-schema-guard.json"
            resource_guard = root / "production-resource-guard.json"
            backup = root / "backup.json"
            restore = root / "restore.json"
            progress = root / "progress.json"
            watch = root / "watch.json"
            triage = root / "triage.json"
            cli_release = root / "cli-release.json"
            go_no_go.write_text(json.dumps({"schema_version": "roborank.launch_go_no_go.v0", "decision": "go", "checks": [{"name": "all", "status": "pass"}]}))
            smoke.write_text(json.dumps({"schema_version": "roborank.launch_smoke.v0", "checks": [{"name": name, "status": "pass"} for name in sorted(cli_module.GO_NO_GO_REQUIRED_SMOKE_CHECKS)]}))
            self.write_passing_privacy_report(privacy)
            self.write_passing_schema_guard_report(schema_guard)
            self.write_passing_resource_guard_report(resource_guard)
            backup.write_text(json.dumps({"schema_version": "roborank.backup_manifest.v0", "checks": [{"name": "d1_sql_import", "status": "pass"}, {"name": "object_integrity", "status": "pass"}]}))
            restore.write_text(json.dumps({"schema_version": "roborank.launch_restore_check.v0", "checks": [{"name": "d1_sql_import", "status": "pass"}, {"name": "object_integrity", "status": "pass"}, {"name": "api_smoke_report", "status": "pass"}]}))
            progress.write_text(json.dumps({"schema_version": "roborank.migration_progress.v0", "decision": "continue"}))
            watch.write_text(json.dumps({"schema_version": "roborank.launch_watch.v0", "checks": [{"name": "storage_failures", "status": "fail"}]}))
            triage.write_text(json.dumps({"schema_version": "roborank.post_launch_triage.v0", "decision": "monitor", "counts": {"open_blockers": 0}}))
            cli_release.write_text(json.dumps({"schema_version": "roborank.cli_release.v0", "publish_ready": True, "checks": [{"name": "entrypoint_smoke", "status": "pass"}]}))

            code, payload = run_cli(
                "launch",
                "cutover",
                "--go-no-go-report",
                str(go_no_go),
                "--production-smoke-report",
                str(smoke),
                "--production-privacy-report",
                str(privacy),
                "--production-schema-guard-report",
                str(schema_guard),
                "--production-resource-guard-report",
                str(resource_guard),
                "--backup-report",
                str(backup),
                "--restore-report",
                str(restore),
                "--migration-progress-report",
                str(progress),
                "--watch-report",
                str(watch),
                "--triage-report",
                str(triage),
                "--cli-release-report",
                str(cli_release),
                "--worker-version",
                "worker-v2",
                "--frontend-version",
                "frontend-v2",
                "--backend-version",
                "backend-v2",
                "--d1-backup",
                "d1-backup-final",
                "--object-backup",
                "r2-snapshot-final",
                "--json",
            )

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "production_release_hold")
        checks = {check["name"]: check for check in payload["error"]["details"]["checks"]}
        self.assertEqual(checks["watch_report"]["status"], "fail")

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

    def test_migration_rerun_writes_private_bundle(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            if method == "GET" and path == "/api/submissions/sub_123":
                return {
                    "submission": {
                        "id": "sub_123",
                        "challengeId": "diff_drive_reach_target",
                        "language": "python",
                        "score": 76,
                        "metrics": {"score": 76, "success": False},
                        "code": "class RobotPolicy: pass\n",
                    }
                }
            raise AssertionError((method, path))

        def fake_run_local_challenge(challenge_id: str, policy_path: Path, max_steps: int | None, artifact_dir: Path):
            self.assertEqual(challenge_id, "diff_drive_reach_target")
            self.assertEqual(policy_path.read_text(), "class RobotPolicy: pass\n")
            self.assertIsNone(max_steps)
            self.assertEqual(artifact_dir.name, "sub_123")
            return {
                "score": 88,
                "status": "success",
                "metrics": {"score": 88, "success": True},
                "logs": ["ok"],
                "replay": {"artifacts": [{"type": "rerun_rrd", "url": "/artifacts/run.rrd"}]},
            }

        def fake_copy_local_rerun_artifact(result_payload: dict, out_dir: Path):
            target = out_dir / "recording.rrd"
            target.write_bytes(b"RRD")
            return target

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(ApiClient, "request_json", fake_request_json), patch.object(
                cli_module, "run_local_challenge", fake_run_local_challenge
            ), patch.object(cli_module, "copy_local_rerun_artifact", fake_copy_local_rerun_artifact):
                code, payload = run_cli(
                    "migration",
                    "rerun",
                    "--submission-id",
                    "sub_123",
                    "--out",
                    directory,
                    "--json",
                )

            bundle = Path(directory) / "sub_123"
            self.assertEqual(code, 0)
            self.assertTrue((bundle / "recording.rrd").exists())
            self.assertTrue((bundle / "metrics.json").exists())
            evidence = json.loads((bundle / "evidence.json").read_text())
            self.assertEqual(evidence["legacy_submission_id"], "sub_123")
            self.assertEqual(evidence["migration_provenance"], "regenerated_from_code")
            self.assertEqual(evidence["source_kind"], "migration")
            self.assertEqual(evidence["legacy_score"], 76)
            self.assertEqual(evidence["legacy_metrics_json"], {"score": 76, "success": False})
            self.assertEqual(evidence["regenerated_score"], 88)
            self.assertEqual(evidence["regenerated_metrics_json"], {"score": 88, "success": True})
            self.assertNotIn("RobotPolicy", json.dumps(payload))

    def test_migration_recover_writes_imported_artifact_bundle(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(method, "GET")
            self.assertEqual(path, "/api/submissions/sub_123")
            return {
                "submission": {
                    "id": "sub_123",
                    "challengeId": "diff_drive_reach_target",
                    "status": "success",
                    "score": 91,
                    "metrics": {"score": 91, "success": True},
                    "logs": ["legacy log"],
                    "code": "class RobotPolicy: pass\n",
                    "codeHash": "hash_123",
                }
            }

        with tempfile.TemporaryDirectory() as directory, patch.object(ApiClient, "request_json", fake_request_json):
            artifact = Path(directory) / "legacy.rrd"
            artifact.write_bytes(b"RRD")
            out = Path(directory) / "migration"
            code, payload = run_cli(
                "migration",
                "recover",
                "--submission-id",
                "sub_123",
                "--artifact",
                str(artifact),
                "--out",
                str(out),
                "--json",
            )

            bundle = out / "sub_123"
            evidence = json.loads((bundle / "evidence.json").read_text())
            result = json.loads((bundle / "result.json").read_text())
            bundle_text = "\n".join(path.read_text(errors="ignore") for path in bundle.glob("*.json"))
            recording_bytes = (bundle / "recording.rrd").read_bytes()
            run_log = (bundle / "run.log").read_text()

        self.assertEqual(code, 0)
        self.assertEqual(recording_bytes, b"RRD")
        self.assertEqual(run_log, "legacy log\n")
        self.assertEqual(evidence["migration_provenance"], "original_artifact_imported")
        self.assertEqual(evidence["legacy_code_hash"], "hash_123")
        self.assertEqual(evidence["source_kind"], "migration")
        self.assertEqual(evidence["legacy_score"], 91)
        self.assertEqual(evidence["legacy_metrics_json"], {"score": 91, "success": True})
        self.assertEqual(evidence["regenerated_score"], 91)
        self.assertEqual(evidence["regenerated_metrics_json"], {"score": 91, "success": True})
        self.assertEqual(result["provenance"], "original_artifact_imported")
        self.assertEqual(result["legacy_score"], 91)
        self.assertEqual(payload["result"]["result"]["provenance"], "original_artifact_imported")
        self.assertNotIn("RobotPolicy", json.dumps(payload))
        self.assertNotIn("RobotPolicy", bundle_text)

    def test_migration_map_marks_unknown_challenge_unmapped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inventory_path = Path(directory) / "migration-inventory.json"
            map_path = Path(directory) / "migration-map.json"
            inventory_path.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.migration_inventory.v0",
                        "limit": 2,
                        "offset": 0,
                        "has_more": False,
                        "submissions": [
                            {
                                "id": "sub_known",
                                "userId": "user_1",
                                "challengeId": "diff_drive_reach_target",
                                "score": 91,
                                "status": "success",
                                "createdAt": "2026-05-27T00:00:00Z",
                                "codeHash": "hash_known",
                                "metrics": {"score": 91, "success": True},
                                "logs": ["legacy ok"],
                                "code": "class RobotPolicy: pass\n",
                            },
                            {
                                "id": "sub_retired",
                                "userId": "user_2",
                                "challengeId": "retired_cart_pole_experiment",
                                "score": 17,
                                "status": "failed",
                                "code": "class RobotPolicy: pass\n",
                            },
                        ],
                    }
                )
            )

            code, payload = run_cli(
                "migration",
                "map",
                "--from",
                str(inventory_path),
                "--out",
                str(map_path),
                "--json",
            )
            written_report = json.loads(map_path.read_text())

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "migration.map")
        self.assertEqual(payload["result"]["counts"], {"total": 2, "mapped": 1, "legacy_unmapped": 1})
        known, retired = payload["result"]["mappings"]
        self.assertEqual(known["mapping_status"], "mapped")
        self.assertEqual(known["robot"], "roborank/differential-drive-cube-v1")
        self.assertEqual(known["environment"], "roborank/diff-drive-reach-target")
        self.assertEqual(known["policy"], "migration/diff-drive-reach-target-sub-known")
        self.assertEqual(known["policy_family"], "migration/diff-drive-reach-target")
        self.assertEqual(known["legacy"]["code_hash"], "hash_known")
        self.assertEqual(known["legacy"]["logs"], ["legacy ok"])
        self.assertEqual(retired["mapping_status"], "legacy_unmapped")
        self.assertEqual(retired["robot"], "std/unknown")
        self.assertEqual(retired["environment"], "std/unknown")
        self.assertEqual(retired["reason"], "challenge resource mapping missing")
        self.assertEqual(written_report["schema_version"], "roborank.migration_mapping.v0")
        self.assertNotIn("RobotPolicy", json.dumps(payload))
        self.assertNotIn("RobotPolicy", json.dumps(written_report))

    def test_migration_map_fetches_submissions_from_api(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            self.assertEqual(method, "GET")
            self.assertEqual(path, "/api/submissions")
            self.assertEqual(
                kwargs["params"],
                {"limit": 1, "offset": 5, "scope": "all", "challenge_id": "diff_drive_reach_target"},
            )
            return {
                "submissions": [
                    {
                        "id": "sub_api",
                        "challengeId": "diff_drive_reach_target",
                        "code": "class RobotPolicy: pass\n",
                    }
                ],
                "hasMore": True,
                "limit": 1,
                "offset": 5,
            }

        with patch.object(ApiClient, "request_json", fake_request_json):
            code, payload = run_cli(
                "migration",
                "map",
                "--challenge",
                "diff_drive_reach_target",
                "--limit",
                "1",
                "--offset",
                "5",
                "--json",
            )

        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["source"], "api/submissions")
        self.assertTrue(payload["result"]["has_more"])
        self.assertEqual(payload["result"]["counts"], {"total": 1, "mapped": 1, "legacy_unmapped": 0})
        self.assertNotIn("RobotPolicy", json.dumps(payload))

    def test_migration_report_summarizes_dry_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / "inventory.json"
            mapping = root / "mapping.json"
            state = root / "state.json"
            verify = root / "verify.json"
            upload = root / "upload.json"
            smoke = root / "smoke.json"
            out = root / "report.json"
            inventory.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.migration_inventory.v0",
                        "submissions": [
                            {"id": "sub_imported", "challengeId": "diff_drive_reach_target", "codeHash": "hash_1"},
                            {"id": "sub_regenerated", "challengeId": "diff_drive_reach_target", "codeHash": "hash_2"},
                            {"id": "sub_unknown", "challengeId": "retired_challenge", "codeHash": "hash_3"},
                        ],
                    }
                )
            )
            mapping.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.migration_mapping.v0",
                        "counts": {"total": 3, "mapped": 2, "legacy_unmapped": 1},
                        "mappings": [
                            {
                                "submission_id": "sub_imported",
                                "user_id": "user_1",
                                "challenge_id": "diff_drive_reach_target",
                                "mapping_status": "mapped",
                                "robot": "roborank/differential-drive-cube-v1",
                                "environment": "roborank/diff-drive-reach-target",
                                "policy": "migration/diff-drive-reach-target-sub-import",
                                "policy_family": "migration/diff-drive-reach-target",
                            },
                            {
                                "submission_id": "sub_regenerated",
                                "user_id": "user_2",
                                "challenge_id": "diff_drive_reach_target",
                                "mapping_status": "mapped",
                                "robot": "roborank/differential-drive-cube-v1",
                                "environment": "roborank/diff-drive-reach-target",
                                "policy": "migration/diff-drive-reach-target-sub-regene",
                                "policy_family": "migration/diff-drive-reach-target",
                            },
                            {
                                "submission_id": "sub_unknown",
                                "user_id": "user_3",
                                "challenge_id": "retired_challenge",
                                "mapping_status": "legacy_unmapped",
                                "robot": "std/unknown",
                                "environment": "std/unknown",
                                "policy": "migration/retired-challenge-sub-unknown",
                                "policy_family": "migration/retired-challenge",
                                "reason": "challenge resource mapping missing",
                            },
                        ],
                    }
                )
            )
            state.write_text(
                json.dumps(
                    {
                        "schema_version": cli_module.MIGRATION_RERUN_STATE_VERSION,
                        "processed": {
                            "sub_imported": {"submission_id": "sub_imported", "provenance": "original_artifact_imported"},
                            "sub_regenerated": {"submission_id": "sub_regenerated", "provenance": "regenerated_from_code"},
                        },
                        "failed": {"sub_failed": {"message": "simulation failed"}},
                        "completed": True,
                    }
                )
            )
            verify.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.migration_verify.v0",
                        "valid": True,
                        "evidence": {"legacy_submission_id": "sub_imported"},
                    }
                )
            )
            upload.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "license": "CC-BY-4.0",
                            "server": {"runId": "run_123", "licenseArtifact": "CC-BY-4.0"},
                        },
                    }
                )
            )
            smoke.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.launch_smoke.v0",
                        "counts": {"pass": 8},
                        "checks": [{"name": "legacy_leaderboard", "status": "pass"}],
                    }
                )
            )

            code, payload = run_cli(
                "migration",
                "report",
                "--inventory",
                str(inventory),
                "--mapping",
                str(mapping),
                "--rerun-state",
                str(state),
                "--verify-report",
                str(verify),
                "--upload-report",
                str(upload),
                "--smoke-report",
                str(smoke),
                "--environment",
                "staging",
                "--rollback-d1",
                "d1-backup-001",
                "--rollback-objects",
                "r2-snapshot-001",
                "--out",
                str(out),
                "--json",
            )
            written = json.loads(out.read_text())

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "migration.report")
        self.assertEqual(payload["result"]["summary"]["legacy_submissions_inspected"], 3)
        self.assertEqual(payload["result"]["summary"]["submissions_with_recovered_rrd_artifacts"], 1)
        self.assertEqual(payload["result"]["summary"]["submissions_regenerated_from_preserved_code"], 1)
        self.assertEqual(payload["result"]["summary"]["legacy_unmapped_submissions"], 1)
        self.assertEqual(payload["result"]["summary"]["regeneration_failed_submissions"], 1)
        self.assertEqual(payload["result"]["privacy_checks"][0]["status"], "pass")
        self.assertEqual(written["schema_version"], "roborank.migration_dry_run_report.v0")

    def test_migration_report_fails_when_inputs_expose_source_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / "inventory.json"
            mapping = root / "mapping.json"
            inventory.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.migration_inventory.v0",
                        "submissions": [{"id": "sub_secret", "challengeId": "diff_drive_reach_target", "code": "class RobotPolicy: pass\n"}],
                    }
                )
            )
            mapping.write_text(json.dumps({"schema_version": "roborank.migration_mapping.v0", "counts": {"total": 0}, "mappings": []}))

            code, payload = run_cli("migration", "report", "--inventory", str(inventory), "--mapping", str(mapping), "--json")

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "migration_report_failed")
        privacy = payload["error"]["details"]["privacy_checks"]
        self.assertEqual(privacy[0]["status"], "fail")

    def test_migration_progress_reports_next_controlled_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = root / "mapping.json"
            state = root / "state.json"
            verify = root / "verify.json"
            upload = root / "upload.json"
            out = root / "progress.json"
            mapping.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.migration_mapping.v0",
                        "counts": {"total": 4, "mapped": 3, "legacy_unmapped": 1},
                        "mappings": [
                            {"submission_id": "sub_1", "mapping_status": "mapped"},
                            {"submission_id": "sub_2", "mapping_status": "mapped"},
                            {"submission_id": "sub_3", "mapping_status": "mapped"},
                            {"submission_id": "sub_4", "mapping_status": "legacy_unmapped"},
                        ],
                    }
                )
            )
            state.write_text(
                json.dumps(
                    {
                        "schema_version": cli_module.MIGRATION_RERUN_STATE_VERSION,
                        "processed": {
                            "sub_1": {"submission_id": "sub_1", "provenance": "original_artifact_imported"},
                            "sub_2": {"submission_id": "sub_2", "provenance": "regenerated_from_code"},
                        },
                        "failed": {"sub_3": {"code": "simulation_failed", "message": "simulation failed"}},
                        "next_offset": 3,
                        "completed": False,
                    }
                )
            )
            verify.write_text(json.dumps({"schema_version": "roborank.migration_verify.v0", "valid": True, "evidence": {"legacy_submission_id": "sub_1"}}))
            upload.write_text(json.dumps({"ok": True, "result": {"run_id": "run_1", "server": {"runId": "run_1"}}}))

            code, payload = run_cli(
                "migration",
                "progress",
                "--mapping",
                str(mapping),
                "--rerun-state",
                str(state),
                "--verify-report",
                str(verify),
                "--upload-report",
                str(upload),
                "--batch-size",
                "25",
                "--max-failures",
                "2",
                "--max-failure-rate",
                "1.0",
                "--out",
                str(out),
                "--json",
            )
            written = json.loads(out.read_text())

        self.assertEqual(code, 0)
        self.assertEqual(payload["command"], "migration.progress")
        self.assertEqual(payload["result"]["schema_version"], "roborank.migration_progress.v0")
        self.assertEqual(payload["result"]["decision"], "continue")
        self.assertEqual(payload["result"]["counts"]["migration_remaining"], 1)
        self.assertEqual(payload["result"]["batch"]["next_offset"], 3)
        self.assertEqual(written["counts"]["evidence_uploaded"], 1)

    def test_migration_progress_pauses_on_repeated_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = root / "mapping.json"
            state = root / "state.json"
            mapping.write_text(
                json.dumps(
                    {
                        "schema_version": "roborank.migration_mapping.v0",
                        "counts": {"total": 3, "mapped": 3, "legacy_unmapped": 0},
                        "mappings": [{"submission_id": f"sub_{index}", "mapping_status": "mapped"} for index in range(3)],
                    }
                )
            )
            state.write_text(
                json.dumps(
                    {
                        "schema_version": cli_module.MIGRATION_RERUN_STATE_VERSION,
                        "processed": {},
                        "failed": {
                            "sub_0": {"code": "storage_write_failed", "message": "storage failed"},
                            "sub_1": {"code": "storage_write_failed", "message": "storage failed"},
                            "sub_2": {"code": "storage_write_failed", "message": "storage failed"},
                        },
                        "completed": False,
                    }
                )
            )

            code, payload = run_cli(
                "migration",
                "progress",
                "--mapping",
                str(mapping),
                "--rerun-state",
                str(state),
                "--max-failures",
                "10",
                "--max-failure-rate",
                "1.0",
                "--json",
            )

        self.assertEqual(code, 9)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "migration_progress_pause")
        self.assertEqual(payload["error"]["details"]["decision"], "pause")
        self.assertEqual(payload["error"]["details"]["health"]["repeated_failures"], ["storage_write_failed"])

    def test_migration_rerun_resume_skips_processed_submission(self) -> None:
        def fake_request_json(api: ApiClient, method: str, path: str, **kwargs):
            if method == "GET" and path == "/api/submissions":
                self.assertEqual(kwargs["params"]["offset"], 0)
                return {
                    "submissions": [{"id": "sub_done"}, {"id": "sub_new"}],
                    "hasMore": False,
                    "limit": 2,
                    "offset": 0,
                }
            if method == "GET" and path == "/api/submissions/sub_done":
                raise AssertionError("processed submissions must be skipped during resume")
            if method == "GET" and path == "/api/submissions/sub_new":
                return {
                    "submission": {
                        "id": "sub_new",
                        "challengeId": "diff_drive_reach_target",
                        "language": "python",
                        "code": "class RobotPolicy: pass\n",
                    }
                }
            raise AssertionError((method, path))

        def fake_run_local_challenge(challenge_id: str, policy_path: Path, max_steps: int | None, artifact_dir: Path):
            self.assertEqual(challenge_id, "diff_drive_reach_target")
            self.assertEqual(policy_path.read_text(), "class RobotPolicy: pass\n")
            self.assertIsNone(max_steps)
            self.assertEqual(artifact_dir.name, "sub_new")
            return {
                "score": 82,
                "status": "success",
                "metrics": {"score": 82, "success": True},
                "logs": ["ok"],
                "replay": {"artifacts": [{"type": "rerun_rrd", "url": "/artifacts/sub_new.rrd"}]},
            }

        def fake_copy_local_rerun_artifact(result_payload: dict, out_dir: Path):
            target = out_dir / "recording.rrd"
            target.write_bytes(b"RRD")
            return target

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "migration-state.json"
            out = Path(directory) / "migration"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": cli_module.MIGRATION_RERUN_STATE_VERSION,
                        "created_at": "2026-05-27T00:00:00Z",
                        "updated_at": "2026-05-27T00:00:00Z",
                        "request": {"limit": 2, "initial_offset": 0, "out": str(out)},
                        "next_offset": 0,
                        "completed": False,
                        "processed": {"sub_done": {"submission_id": "sub_done", "bundle_dir": str(out / "sub_done")}},
                        "failed": {},
                    }
                )
            )
            with patch.object(ApiClient, "request_json", fake_request_json), patch.object(
                cli_module, "run_local_challenge", fake_run_local_challenge
            ), patch.object(cli_module, "copy_local_rerun_artifact", fake_copy_local_rerun_artifact):
                code, payload = run_cli(
                    "migration",
                    "rerun",
                    "--all",
                    "--resume",
                    "--state",
                    str(state_path),
                    "--out",
                    str(out),
                    "--limit",
                    "2",
                    "--json",
                )

            state = json.loads(state_path.read_text())

        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["count"], 2)
        self.assertEqual(payload["result"]["failed_count"], 0)
        self.assertEqual(payload["result"]["next_offset"], 2)
        self.assertTrue(state["completed"])
        self.assertIn("sub_done", state["processed"])
        self.assertIn("sub_new", state["processed"])
        self.assertEqual(state["next_offset"], 2)
        self.assertNotIn("RobotPolicy", json.dumps(payload))

    def test_migration_verify_rejects_code_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "evidence.json").write_text(json.dumps({"legacy_submission_id": "sub_123"}))
            (bundle / "metrics.json").write_text("{}")
            (bundle / "result.json").write_text('{"code": "secret"}')

            code, payload = run_cli("migration", "verify", "--from", directory, "--json")

        self.assertEqual(code, 8)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "migration_verify_failed")


if __name__ == "__main__":
    unittest.main()
