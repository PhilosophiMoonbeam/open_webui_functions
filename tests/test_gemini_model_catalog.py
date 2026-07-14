from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# The companion is a standalone Open WebUI plugin. Catalog tests only need its
# pure validation models, so provide the one host module required at import.
sys.modules.setdefault("open_webui.models.chats", MagicMock())

from plugins.filters.gemini_manifold_companion import (
    MODEL_CATALOG_SCHEMA_VERSION,
    Filter,
    ModelCatalog,
    ModelCatalogError,
)
from pydantic import ValidationError

CATALOG_PATH = Path(__file__).parents[1] / "plugins" / "pipes" / "gemini_models.yaml"


def _raw_catalog() -> dict[str, object]:
    loaded = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_catalog_schema_is_the_expected_protocol() -> None:
    catalog = ModelCatalog.model_validate(_raw_catalog())

    assert catalog.schema_version == MODEL_CATALOG_SCHEMA_VERSION == 1
    assert len(catalog.models) == 11
    assert all(model.services.developer == "supported" for model in catalog.models.values())
    assert all(model.services.enterprise == "unverified" for model in catalog.models.values())


def test_catalog_semantics_cover_media_tools_thinking_and_pricing() -> None:
    catalog = ModelCatalog.model_validate(_raw_catalog())

    for model_id, model in catalog.models.items():
        assert model.limits.input_tokens > 0
        assert model.limits.output_tokens > 0
        assert model.content.inputs
        assert model.content.outputs
        assert model.pricing.input[-1].up_to_tokens is None
        assert model.pricing.output[-1].up_to_tokens is None
        assert model.interactions.thinking.supported == bool(model.interactions.thinking.levels)
        assert ("image" in model.content.outputs) == (model.pricing.image_output is not None)
        # The audited Interactions function-calling guide uses this exact model.
        assert model.interactions.custom_function_calling is (model_id == "gemini-3.5-flash")
        assert model.interactions.tools.file_search is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update(schema_version=2), "schema_version"),
        (
            lambda raw: raw["models"]["gemini-2.5-flash"]["interactions"]["thinking"].update(
                supported=False
            ),
            "thinking levels",
        ),
        (
            lambda raw: raw["models"]["gemini-2.5-flash"]["pricing"]["input"].__setitem__(
                -1, {"up_to_tokens": 100, "price_per_million": 0.3}
            ),
            "unbounded tier",
        ),
    ],
)
def test_catalog_rejects_protocol_and_semantic_mismatches(mutation, message: str) -> None:
    raw = _raw_catalog()
    mutation(raw)

    with pytest.raises(ValidationError, match=message):
        ModelCatalog.model_validate(raw)


def test_companion_loader_validates_and_flattens_catalog_for_both_plugins() -> None:
    Filter._load_model_config.cache_clear()

    with (
        CATALOG_PATH.open("rb") as catalog_file,
        patch(
            "plugins.filters.gemini_manifold_companion.urllib.request.urlopen",
            return_value=catalog_file,
        ),
    ):
        models = Filter._load_model_config("https://example.test/gemini_models.yaml")

    assert len(models) == 11
    assert Filter._check_model_capability("gemini-2.5-flash", models, "search_grounding") is True
    assert Filter._check_model_capability("gemini-future-unknown", models, "thinking") is False
    assert Filter._check_model_capability("gemini-2.5-flash", models, "future_tool") is False


def test_companion_loader_fails_visibly_instead_of_returning_empty_policy() -> None:
    Filter._load_model_config.cache_clear()

    with pytest.raises(ModelCatalogError, match="must not be empty"):
        Filter._load_model_config("")

    with pytest.raises(ModelCatalogError, match=r"HTTP\(S\) URL"):
        Filter._load_model_config(str(CATALOG_PATH))
