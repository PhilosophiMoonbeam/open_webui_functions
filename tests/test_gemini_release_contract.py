"""Coordinated Gemini suite release and archive-policy contract."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import re
import subprocess
import tarfile
from pathlib import Path
from types import ModuleType
from typing import cast

import yaml

ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "release" / "gemini-suite-v3.0.0.yaml"
BUILDER_PATH = ROOT / ".github" / "scripts" / "build_gemini_suite.py"


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_gemini_suite", BUILDER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_hashes_every_coordinated_artifact() -> None:
    raw = cast(dict[str, object], yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")))
    artifacts = cast(list[dict[str, object]], raw["artifacts"])

    assert raw["tag"] == "gemini-suite/v3.0.0"
    assert raw["release_policy"] == "coordinated_only"
    assert cast(dict[str, object], raw["protocols"]) == {
        "google_genai": "2.11.0",
        "grounding_envelope": 1,
        "model_catalog": 1,
    }
    assert {artifact["role"] for artifact in artifacts} >= {"pipe", "companion", "catalog"}
    for artifact in artifacts:
        content = (ROOT / cast(str, artifact["path"])).read_bytes()
        assert hashlib.sha256(content).hexdigest() == artifact["sha256"]
        assert len(content) == artifact["bytes"]


def test_bundle_is_byte_reproducible_and_has_only_manifested_members(tmp_path: Path) -> None:
    builder = _load_builder()
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    builder.build_bundle(MANIFEST_PATH, first, expected_tag="gemini-suite/v3.0.0")
    builder.build_bundle(MANIFEST_PATH, second, expected_tag="gemini-suite/v3.0.0")

    assert first.read_bytes() == second.read_bytes()
    raw = cast(dict[str, object], yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")))
    artifacts = cast(list[dict[str, object]], raw["artifacts"])
    expected = {f"manifest/{MANIFEST_PATH.name}"} | {
        cast(str, artifact["path"]) for artifact in artifacts
    }
    with tarfile.open(first, mode="r:gz") as archive:
        assert set(archive.getnames()) == expected
        assert all(member.mtime == 0 for member in archive.getmembers())


def test_release_workflow_forbids_individual_v3_suite_member_tags() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    helper = (ROOT / "dev" / "release.sh").read_text(encoding="utf-8")
    raw = cast(dict[str, object], yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")))
    artifacts = cast(list[dict[str, object]], raw["artifacts"])
    deployable_basenames = {
        Path(cast(str, artifact["path"])).stem
        for artifact in artifacts
        if artifact["role"] not in {"catalog", "release_notes"}
    }
    for basename in deployable_basenames:
        assert basename in workflow
        assert basename in helper
    assert "gemini-suite/v3.0.0" in workflow
    assert "build_gemini_suite.py" in workflow


def test_generate_content_archive_is_explicitly_excluded_from_suite() -> None:
    archive_readme = (ROOT / "plugins" / "archived" / "README.md").read_text(encoding="utf-8")
    manifest = MANIFEST_PATH.read_text(encoding="utf-8")

    assert "unsupported" in archive_readme.lower()
    assert "thinking_gemini.py" in archive_readme
    assert "thinking_gemini.py" not in manifest


def test_ci_and_release_workflows_cover_all_contract_surfaces() -> None:
    ci_path = ROOT / ".github" / "workflows" / "python-ci.yml"
    release_path = ROOT / ".github" / "workflows" / "release.yml"
    live_path = ROOT / ".github" / "workflows" / "gemini-live-smoke.yml"
    for path in (ci_path, release_path, live_path):
        parsed = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        assert isinstance(parsed, dict)
        assert "on" in parsed
        assert "jobs" in parsed

    ci = ci_path.read_text(encoding="utf-8")
    for surface in (
        '".github/**"',
        '"dev/**"',
        '"docs/**"',
        '"plugins/filters/**"',
        '"plugins/pipes/**"',
        '"release/**"',
        '"tests/**"',
        '"utils/**"',
        '".python-version"',
        '"pyproject.toml"',
        '"uv.lock"',
    ):
        assert surface in ci
    assert "make check" in ci
    assert "ubuntu-latest, macos-latest, windows-latest" in ci
    assert 'python-version: ["3.11", "3.12", "3.13"]' in ci

    release = release_path.read_text(encoding="utf-8")
    assert "make check" in release
    assert "cmp dist/gemini-suite-v3.0.0.tar.gz" in release
    assert "dist/SHA256SUMS" in release
    assert "release/gemini-suite-v3.0.0.yaml" in release

    live = live_path.read_text(encoding="utf-8")
    assert 'test -n "$GEMINI_API_KEY"' in live
    assert "Enterprise policy contract (no network)" in live
    assert "test_service_specific_interactions_paths" in live
    assert "test_public_pipe_denies_unverified_enterprise_before_create" in live
    assert "GCP_WORKLOAD_IDENTITY_PROVIDER" not in live

    live_tests = (ROOT / "tests" / "live" / "test_gemini_interactions_live.py").read_text(
        encoding="utf-8"
    )
    assert 'pytest.fail("RUN_GEMINI_LIVE_TESTS=1 requires GEMINI_API_KEY")' in live_tests
    assert "enterprise_api" not in live_tests


def test_release_tag_parser_rejects_malformed_tags(tmp_path: Path) -> None:
    parser = ROOT / ".github" / "scripts" / "parse_tag.sh"
    output = tmp_path / "output"
    environment = {**os.environ, "GITHUB_OUTPUT": str(output)}

    valid = subprocess.run(
        ["bash", str(parser)],
        cwd=ROOT,
        env={**environment, "GITHUB_REF_NAME": "gemini-suite/v3.0.0"},
        check=False,
        capture_output=True,
        text=True,
    )
    malformed = subprocess.run(
        ["bash", str(parser)],
        cwd=ROOT,
        env={**environment, "GITHUB_REF_NAME": "gemini-suite/latest"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert valid.returncode == 0
    assert malformed.returncode != 0


def test_live_continuation_stores_and_deletes_provider_interactions() -> None:
    live_path = ROOT / "tests" / "live" / "test_gemini_interactions_live.py"
    tree = ast.parse(live_path.read_text(encoding="utf-8"), filename=str(live_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "test_developer_previous_interaction_live"
    )
    continuation_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create"
        and any(keyword.arg == "previous_interaction_id" for keyword in node.keywords)
    ]
    assert len(continuation_calls) == 1
    continuation_keywords = {
        keyword.arg: keyword.value
        for keyword in continuation_calls[0].keywords
        if keyword.arg is not None
    }
    store_node = continuation_keywords.get("store")
    assert isinstance(store_node, ast.Constant)
    assert store_node.value is True

    cleanup_calls = [
        nested
        for node in ast.walk(function)
        if isinstance(node, ast.Try)
        for final_node in node.finalbody
        for nested in ast.walk(final_node)
        if isinstance(nested, ast.Call)
        and isinstance(nested.func, ast.Attribute)
        and nested.func.attr == "delete"
    ]
    assert len(cleanup_calls) == 1


def test_gemini_docs_have_valid_local_links_and_no_stale_companion_contract() -> None:
    documents = (
        ROOT / "README.md",
        ROOT / "docs" / "plugins" / "pipes" / "gemini_manifold.md",
        ROOT / "docs" / "plugins" / "filters" / "gemini_manifold_companion.md",
        ROOT / "docs" / "plugins" / "filters" / "gemini_reasoning_toggle.md",
        ROOT / "docs" / "plugins" / "filters" / "gemini_url_context_toggle.md",
        ROOT / "docs" / "plugins" / "filters" / "gemini_map_grounding_toggle.md",
        ROOT / "docs" / "plugins" / "filters" / "gemini_enterprise_toggle.md",
        ROOT / "docs" / "plugins" / "filters" / "gemini_paid_api.md",
        ROOT / "docs" / "development" / "gemini-suite-release.md",
        ROOT / "release" / "gemini-suite-v3.0.0.md",
        ROOT / "tests" / "live" / "README.md",
    )
    markdown_link = re.compile(r"\[[^]]+]\(([^)]+)\)")
    for document in documents:
        content = document.read_text(encoding="utf-8")
        for target in markdown_link.findall(content):
            path_text = target.split("#", 1)[0]
            if not path_text or "://" in path_text or path_text.startswith("mailto:"):
                continue
            assert (document.parent / path_text).resolve().exists(), f"{document}: {target}"

    companion = documents[2].read_text(encoding="utf-8")
    for stale in (
        "Set Temp To Zero",
        "Grounding Dynamic Retrieval Threshold",
        "ALLOWED_GROUNDING_MODELS",
        "grounding_supports",
        "grounding_chunks",
    ):
        assert stale not in companion

    live_readme = documents[-1].read_text(encoding="utf-8")
    assert "catalog-gated skip" not in live_readme
    assert "Enterprise authentication uses" not in live_readme
