"""Coordinated Gemini suite release and archive-policy contract."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "release" / "gemini-suite-v3.0.0.yaml"
BUILDER_PATH = ROOT / ".github" / "scripts" / "build_gemini_suite.py"


def _bash_executable() -> str:
    if os.name != "nt":
        return "bash"
    git_executable = shutil.which("git")
    assert git_executable is not None
    git_directory = Path(git_executable).parent
    candidates = (git_directory / "bash.exe", git_directory.parent / "bin" / "bash.exe")
    bash = next((candidate for candidate in candidates if candidate.is_file()), None)
    assert bash is not None, "Git for Windows Bash is required by the release contract"
    return str(bash)


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_gemini_suite", BUILDER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_live_tests() -> ModuleType:
    path = ROOT / "tests" / "live" / "test_gemini_interactions_live.py"
    spec = importlib.util.spec_from_file_location("gemini_live_contract", path)
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
        "model_catalog": 3,
    }
    assert {artifact["role"] for artifact in artifacts} >= {
        "pipe",
        "companion",
        "catalog",
        "provenance",
    }
    for artifact in artifacts:
        content = (ROOT / cast(str, artifact["path"])).read_bytes()
        assert hashlib.sha256(content).hexdigest() == artifact["sha256"]
        assert len(content) == artifact["bytes"]


def test_git_checkout_preserves_canonical_release_bytes() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "* text=auto eol=lf" in attributes


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
        if artifact["role"] not in {"catalog", "provenance", "release_notes"}
    }
    for basename in deployable_basenames:
        assert basename in workflow
        assert basename in helper
    assert "gemini-suite/v3.0.0" in workflow
    assert "build_gemini_suite.py" in workflow


def test_release_loader_rejects_duplicate_and_merge_keys(tmp_path: Path) -> None:
    builder = _load_builder()
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    merged = tmp_path / "merged.yaml"
    merged.write_text("base: &base\n  value: 1\ncopy:\n  <<: *base\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        builder.load_yaml(duplicate)
    with pytest.raises(ValueError, match="merge"):
        builder.load_yaml(merged)


def test_generate_content_archive_is_explicitly_excluded_from_suite() -> None:
    archive_readme = (ROOT / "plugins" / "archived" / "README.md").read_text(encoding="utf-8")
    manifest = MANIFEST_PATH.read_text(encoding="utf-8")

    assert "unsupported" in archive_readme.lower()
    assert "thinking_gemini.py" in archive_readme
    assert "thinking_gemini.py" not in manifest


def test_greenfield_release_excludes_retired_dependency_and_catalog_tools() -> None:
    assert not (ROOT / "requirements.txt").exists()
    assert not (ROOT / "utils" / "gemini_yaml_wizard.py").exists()


def test_ci_and_release_workflows_cover_all_contract_surfaces() -> None:
    ci_path = ROOT / ".github" / "workflows" / "python-ci.yml"
    release_path = ROOT / ".github" / "workflows" / "release.yml"
    live_path = ROOT / ".github" / "workflows" / "gemini-live-smoke.yml"
    enterprise_live_path = ROOT / ".github" / "workflows" / "gemini-enterprise-live-smoke.yml"
    for path in (ci_path, release_path, live_path, enterprise_live_path):
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
        '".gitattributes"',
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

    enterprise_live = enterprise_live_path.read_text(encoding="utf-8")
    assert "environment: gemini-live-enterprise" in enterprise_live
    assert "id-token: write" in enterprise_live
    assert "google-github-actions/auth@" in enterprise_live
    assert 'test -n "$GCP_WORKLOAD_IDENTITY_PROVIDER"' in enterprise_live
    assert 'test -n "$GEMINI_ENTERPRISE_PROJECT"' in enterprise_live
    assert 'test -n "$GEMINI_ENTERPRISE_LOCATION"' in enterprise_live
    assert "live and enterprise_api" in enterprise_live
    assert "if: github.ref" not in enterprise_live
    assert "Require trusted default-branch dispatch" in enterprise_live
    assert "ACTUAL_REF" in enterprise_live
    assert "EXPECTED_REF" in enterprise_live
    assert "must run from the repository default branch" in enterprise_live
    assert enterprise_live.index("Require trusted default-branch dispatch") < enterprise_live.index(
        "actions/checkout@"
    )
    assert "persist-credentials: false" in enterprise_live
    assert "timeout-minutes: 15" in enterprise_live
    assert "cancel-in-progress: false" in enterprise_live
    assert "group: gemini-enterprise-live-${{ inputs.api_version }}" in enterprise_live
    assert "gemini-enterprise-live-${{ inputs.model }}" not in enterprise_live
    assert "Enterprise model must be a bare model ID" in enterprise_live
    assert '[[ ! "$GEMINI_ENTERPRISE_MODEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]' in (
        enterprise_live
    )
    checkout_position = enterprise_live.index("actions/checkout@")
    auth_position = enterprise_live.index("google-github-actions/auth@")
    for masked_name in (
        "GCP_WORKLOAD_IDENTITY_PROVIDER",
        "GCP_SERVICE_ACCOUNT",
        "GEMINI_ENTERPRISE_PROJECT",
        "GEMINI_ENTERPRISE_LOCATION",
    ):
        mask_position = enterprise_live.index(f'echo "::add-mask::${masked_name}"')
        assert mask_position < checkout_position
        assert mask_position < auth_position
    assert "--junitxml=enterprise-live-results.xml" in enterprise_live
    assert '"tests": 3, "failures": 0, "errors": 0, "skipped": 0' in enterprise_live
    action_refs = re.findall(r"uses:\s+[^@\s]+@([^\s#]+)", enterprise_live)
    assert len(action_refs) == 4
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs)

    live_tests = (ROOT / "tests" / "live" / "test_gemini_interactions_live.py").read_text(
        encoding="utf-8"
    )
    assert 'pytest.fail("RUN_GEMINI_LIVE_TESTS=1 requires GEMINI_API_KEY")' in live_tests
    assert "RUN_GEMINI_ENTERPRISE_LIVE_TESTS=1 requires" in live_tests
    assert "test_enterprise_text_interaction_live" in live_tests
    assert "test_enterprise_previous_interaction_live" in live_tests


def test_maintained_workflows_pin_current_node24_actions() -> None:
    expected = {
        "actions/checkout": "9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0",
        "actions/setup-python": "ece7cb06caefa5fff74198d8649806c4678c61a1",
        "astral-sh/setup-uv": "11f9893b081a58869d3b5fccaea48c9e9e46f990",
    }
    seen: set[str] = set()
    for path in (ROOT / ".github" / "workflows").glob("*.yml"):
        source = path.read_text(encoding="utf-8")
        for action, revision in re.findall(
            r"uses:\s+(actions/(?:checkout|setup-python)|astral-sh/setup-uv)@([^\s#]+)",
            source,
        ):
            seen.add(action)
            assert revision == expected[action], path
    assert seen == set(expected)


def test_release_tag_parser_rejects_malformed_tags(tmp_path: Path) -> None:
    parser = ROOT / ".github" / "scripts" / "parse_tag.sh"
    output = tmp_path / "output"
    environment = {**os.environ, "GITHUB_OUTPUT": str(output)}

    valid = subprocess.run(
        [_bash_executable(), str(parser)],
        cwd=ROOT,
        env={**environment, "GITHUB_REF_NAME": "gemini-suite/v3.0.0"},
        check=False,
        capture_output=True,
        text=True,
    )
    malformed = subprocess.run(
        [_bash_executable(), str(parser)],
        cwd=ROOT,
        env={**environment, "GITHUB_REF_NAME": "gemini-suite/latest"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert valid.returncode == 0
    assert malformed.returncode != 0


def test_live_continuation_is_semantic_redacted_and_cleanup_owned() -> None:
    live_path = ROOT / "tests" / "live" / "test_gemini_interactions_live.py"
    tree = ast.parse(live_path.read_text(encoding="utf-8"), filename=str(live_path))
    probe = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_previous_interaction_probe"
    )
    continuation_calls = [
        node
        for node in ast.walk(probe)
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

    probe_text = ast.unparse(probe)
    assert "temporary token I asked you to remember" in probe_text
    assert "ALPHA" not in probe_text
    assert "second.output_text" not in probe_text or "pytest.fail" in probe_text
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(probe))

    text_probe = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_text_interaction_probe"
    )
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(text_probe))
    assert "provider details were redacted" in ast.unparse(text_probe)

    cleanup = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_cleanup_stored_interactions"
    )
    cleanup_attributes = {
        node.attr for node in ast.walk(cleanup) if isinstance(node, ast.Attribute)
    }
    assert {"delete", "aclose"} <= cleanup_attributes
    assert "asyncio.shield" in ast.unparse(probe)

    test_names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name.endswith("previous_interaction_live")
    }
    assert test_names == {
        "test_developer_previous_interaction_live",
        "test_enterprise_previous_interaction_live",
    }


@pytest.mark.asyncio
async def test_live_continuation_cleanup_survives_semantic_and_delete_failures(monkeypatch) -> None:
    live = _load_live_tests()
    monkeypatch.setattr(live.secrets, "token_hex", lambda _size: "ABC123")
    first = live.interactions.Interaction(id="first", status="completed", steps=[])
    second = live.interactions.Interaction(
        id="second",
        status="completed",
        steps=[
            live.interactions.ModelOutputStep(content=[live.interactions.TextContent(text="wrong")])
        ],
    )
    client = MagicMock()
    client.aio.interactions.create = AsyncMock(side_effect=[first, second])
    client.aio.interactions.delete = AsyncMock(side_effect=[RuntimeError("redacted"), None])
    client.aio.aclose = AsyncMock()

    with pytest.raises(pytest.fail.Exception, match="semantic continuation"):
        await live._run_previous_interaction_probe(client, "model")

    assert [call.args[0] for call in client.aio.interactions.delete.await_args_list] == [
        "second",
        "first",
    ]
    client.aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_probes_redact_provider_exception_details(monkeypatch) -> None:
    live = _load_live_tests()
    monkeypatch.setattr(live.secrets, "token_hex", lambda _size: "ABC123")
    sensitive_detail = "projects/private-project/locations/private-location provider body"

    continuation_client = MagicMock()
    continuation_client.aio.interactions.create = AsyncMock(
        side_effect=RuntimeError(sensitive_detail)
    )
    continuation_client.aio.aclose = AsyncMock()
    with pytest.raises(pytest.fail.Exception) as continuation_failure:
        await live._run_previous_interaction_probe(continuation_client, "model")
    assert sensitive_detail not in str(continuation_failure.value)
    continuation_client.aio.aclose.assert_awaited_once()

    text_client = MagicMock()
    text_client.aio.interactions.create = AsyncMock(side_effect=RuntimeError(sensitive_detail))
    text_client.aio.aclose = AsyncMock()
    with pytest.raises(pytest.fail.Exception) as text_failure:
        await live._run_text_interaction_probe(text_client, "model", False)
    assert sensitive_detail not in str(text_failure.value)
    text_client.aio.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_continuation_cleanup_survives_cancellation() -> None:
    live = _load_live_tests()
    first = live.interactions.Interaction(id="first", status="completed", steps=[])
    second_create_started = asyncio.Event()

    async def create_interaction(**_kwargs):
        if not second_create_started.is_set():
            second_create_started.set()
            return first
        await asyncio.Future()

    client = MagicMock()
    client.aio.interactions.create = AsyncMock(side_effect=create_interaction)
    client.aio.interactions.delete = AsyncMock()
    client.aio.aclose = AsyncMock()
    probe = asyncio.create_task(live._run_previous_interaction_probe(client, "model"))
    await second_create_started.wait()
    await asyncio.sleep(0)
    probe.cancel()

    with pytest.raises(asyncio.CancelledError):
        await probe

    client.aio.interactions.delete.assert_awaited_once_with("first")
    client.aio.aclose.assert_awaited_once()


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
