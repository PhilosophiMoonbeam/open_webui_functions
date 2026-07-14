"""Negative guards for the maintained Gemini Interactions implementation."""

import ast
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
