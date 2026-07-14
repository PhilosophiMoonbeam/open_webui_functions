"""Negative guards for the maintained Gemini Interactions implementation."""

import ast
import base64
from pathlib import Path

ROOT = Path(__file__).parents[1]
MAINTAINED = (
    ROOT / "plugins" / "pipes" / "gemini_manifold.py",
    ROOT / "plugins" / "filters" / "gemini_manifold_companion.py",
    ROOT / "utils" / "manifold_types.py",
)


def _attribute_path(node: ast.Attribute) -> str:
    values = [node.attr]
    current: ast.expr = node.value
    while isinstance(current, ast.Attribute):
        values.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        values.append(current.id)
    return ".".join(reversed(values))


def test_maintained_generation_has_no_models_generate_content_calls() -> None:
    forbidden_suffixes = (
        ".models." + "generate_content",
        ".models." + "generate_content_stream",
    )
    findings: list[str] = []
    for path in MAINTAINED:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                dotted = _attribute_path(node.func)
                if dotted.endswith(forbidden_suffixes):
                    findings.append(f"{path.relative_to(ROOT)}:{node.lineno}:{dotted}")
    assert findings == []


def test_maintained_generation_has_no_sdk_chat_afc_or_unsupported_request_attributes() -> None:
    forbidden_attributes = {
        "automatic_function_calling",
        "automatic_function_calling_config",
        "cached_content",
        "batch_api",
        "batch_config",
        "video_metadata",
    }
    findings: list[str] = []
    for path in MAINTAINED:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            dotted = _attribute_path(node)
            if ".aio.chats" in dotted or node.attr in forbidden_attributes:
                findings.append(f"{path.relative_to(ROOT)}:{node.lineno}:{dotted}")
    assert findings == []


def test_maintained_contract_has_no_removed_generate_content_storage_types() -> None:
    removed_names = {
        "Generate" + "ContentConfig",
        "Generate" + "ContentResponse",
        "gemini_" + "parts",
        "Grounding" + "Metadata",
    }
    findings: list[str] = []
    for path in MAINTAINED:
        source = path.read_text(encoding="utf-8")
        for name in removed_names:
            if name in source:
                findings.append(f"{path.relative_to(ROOT)}:{name}")
    assert findings == []


def test_maintained_contract_has_no_custom_safety_configuration_surface() -> None:
    removed_names = {
        "USE_" + "PERMISSIVE_SAFETY",
        "safety_" + "settings",
        "Safety" + "Setting",
        "permissive " + "safety",
    }
    findings: list[str] = []
    for path in MAINTAINED:
        source = path.read_text(encoding="utf-8")
        for name in removed_names:
            if name in source:
                findings.append(f"{path.relative_to(ROOT)}:{name}")
    assert findings == []


def test_maintained_response_path_has_no_candidate_or_usage_metadata_access() -> None:
    removed_attributes = {"candidates", "usage_" + "metadata"}
    findings: list[str] = []
    for path in MAINTAINED:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in removed_attributes:
                findings.append(f"{path.relative_to(ROOT)}:{node.lineno}:{_attribute_path(node)}")
    assert findings == []


def test_vertex_toggle_and_identifier_are_removed_from_maintained_surface() -> None:
    old_toggle = ROOT / "plugins" / "filters" / ("gemini_" + "vertex_ai_toggle.py")
    assert not old_toggle.exists()
    removed_identifier = "gemini_" + "vertex_ai_toggle"
    findings = [
        str(path.relative_to(ROOT))
        for path in MAINTAINED
        if removed_identifier in path.read_text(encoding="utf-8")
    ]
    assert findings == []


def test_enterprise_toggle_embedded_asset_has_no_vertex_branding() -> None:
    toggle_path = ROOT / "plugins" / "filters" / "gemini_enterprise_toggle.py"
    tree = ast.parse(toggle_path.read_text(encoding="utf-8"), filename=str(toggle_path))
    icon_values = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "icon" for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    assert len(icon_values) == 1
    prefix = "data:image/svg+xml;base64,"
    assert icon_values[0].startswith(prefix)
    decoded = base64.b64decode(icon_values[0][len(prefix) :], validate=True).decode("utf-8")
    assert "vertex" not in decoded.lower()
    assert "gemini enterprise" in decoded.lower()


def test_maintained_failure_paths_forbid_raw_traceback_and_value_logging() -> None:
    forbidden = {
        "log." + "exception(": "raw exception traceback logging",
        "difflib." + "unified_diff": "conversation diff logging",
        "diff_" + "str": "conversation diff materialization",
        "uri[" + ":64]": "URI prefix disclosure",
        "at {config_" + "path}: {e}": "catalog URL and exception disclosure",
    }
    findings: list[str] = []
    for path in MAINTAINED:
        source = path.read_text(encoding="utf-8")
        for snippet, description in forbidden.items():
            if snippet in source:
                findings.append(f"{path.relative_to(ROOT)}:{description}")
    assert findings == []


def test_pipe_persists_only_the_versioned_interaction_envelope() -> None:
    pipe_path = MAINTAINED[0]
    tree = ast.parse(pipe_path.read_text(encoding="utf-8"), filename=str(pipe_path))
    payload_key_sets: list[set[str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "upsert_message_to_chat_by_id_and_message_id"
        ):
            continue
        message_keyword = next(
            (keyword for keyword in node.keywords if keyword.arg == "message"), None
        )
        assert message_keyword is not None
        assert isinstance(message_keyword.value, ast.Dict)
        keys = {
            key.value
            for key in message_keyword.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        payload_key_sets.append(keys)
    assert payload_key_sets == [{"gemini_interaction"}]


def test_catalog_protocol_2_and_split_app_state_keys_are_removed() -> None:
    surfaces = (
        ROOT / "plugins" / "pipes" / "gemini_manifold.py",
        ROOT / "plugins" / "filters" / "gemini_manifold_companion.py",
        ROOT / "docs" / "development" / "gemini-suite-release.md",
        ROOT / "docs" / "plugins" / "pipes" / "gemini_manifold.md",
        ROOT / "release" / "gemini-suite-v3.0.0.md",
        ROOT / "release" / "gemini-suite-v3.0.0.yaml",
    )
    forbidden = (
        "protocol" + "-2",
        "protocol " + "2",
        "schema " + "2",
        "gemini_model_" + "config",
        "gemini_model_catalog_" + "schema_version",
    )
    findings = [
        f"{path.relative_to(ROOT)}:{value}"
        for path in surfaces
        for value in forbidden
        if value in path.read_text(encoding="utf-8")
    ]
    assert findings == []


def test_removed_model_aliases_are_absent_from_maintained_and_release_surfaces() -> None:
    surfaces = MAINTAINED + (
        ROOT / "docs" / "development" / "gemini-suite-release.md",
        ROOT / "docs" / "plugins" / "pipes" / "gemini_manifold.md",
        ROOT / "release" / "gemini-suite-v3.0.0.md",
        ROOT / "release" / "gemini-suite-v3.0.0.yaml",
        ROOT / "plugins" / "pipes" / "gemini_models.yaml",
    )
    removed_ids = (
        "gemini-3.1-flash-lite-" + "preview",
        "gemini-3.1-flash-image-" + "preview",
        "gemini-3-pro-image-" + "preview",
        "gemini-2.5-flash-" + "image",
    )
    findings = [
        f"{path.relative_to(ROOT)}:{model_id}"
        for path in surfaces
        for model_id in removed_ids
        if model_id in path.read_text(encoding="utf-8")
    ]
    assert findings == []
