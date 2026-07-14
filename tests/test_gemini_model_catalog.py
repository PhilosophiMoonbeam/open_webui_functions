from __future__ import annotations

import sys
from pathlib import Path
from typing import cast
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


def _service_policy(raw: dict[str, object], model_id: str, service: str) -> dict[str, object]:
    models = cast(dict[str, object], raw["models"])
    model = cast(dict[str, object], models[model_id])
    services = cast(dict[str, object], model["services"])
    return cast(dict[str, object], services[service])


def test_catalog_schema_is_the_expected_protocol() -> None:
    catalog = ModelCatalog.model_validate(_raw_catalog())

    assert catalog.schema_version == MODEL_CATALOG_SCHEMA_VERSION == 2
    assert len(catalog.models) == 11
    assert all(
        model.services.developer.availability == "supported" for model in catalog.models.values()
    )
    assert all(
        model.services.enterprise.availability == "unverified" for model in catalog.models.values()
    )


def test_catalog_semantics_cover_media_tools_thinking_and_pricing() -> None:
    catalog = ModelCatalog.model_validate(_raw_catalog())

    for model_id, model in catalog.models.items():
        developer = model.services.developer
        assert developer.availability == "supported"
        assert developer.limits.input_tokens > 0
        assert developer.limits.output_tokens > 0
        assert developer.content.inputs
        assert developer.content.outputs
        assert developer.pricing.input[-1].up_to_tokens is None
        assert developer.pricing.output[-1].up_to_tokens is None
        assert developer.interactions.thinking.supported == bool(
            developer.interactions.thinking.levels
        )
        assert ("image" in developer.content.outputs) == (
            developer.pricing.image_output is not None
        )
        # The audited Interactions function-calling guide uses this exact model.
        assert developer.interactions.custom_function_calling is (model_id == "gemini-3.5-flash")
        assert developer.interactions.tools.file_search is False
        assert model.services.enterprise.model_dump() == {
            "availability": "unverified",
            "reason": "No credential-backed Enterprise Interactions model and capability evidence is recorded.",
        }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update(schema_version=1), "schema_version"),
        (
            lambda raw: raw["models"]["gemini-2.5-flash"]["services"]["developer"]["interactions"][
                "thinking"
            ].update(supported=False),
            "thinking levels",
        ),
        (
            lambda raw: raw["models"]["gemini-2.5-flash"]["services"]["developer"]["pricing"][
                "input"
            ].__setitem__(-1, {"up_to_tokens": 100, "price_per_million": 0.3}),
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
    assert (
        Filter._check_model_capability(
            "gemini-2.5-flash", models, "search_grounding", service="enterprise"
        )
        is False
    )
    assert Filter._check_model_capability("gemini-future-unknown", models, "thinking") is False
    assert Filter._check_model_capability("gemini-2.5-flash", models, "future_tool") is False


def test_unverified_service_cannot_carry_capabilities_or_pricing() -> None:
    raw = _raw_catalog()
    enterprise = _service_policy(raw, "gemini-2.5-flash", "enterprise")
    enterprise["interactions"] = {"store": True}

    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelCatalog.model_validate(raw)


def test_supported_service_requires_complete_policy_and_known_source_refs() -> None:
    raw = _raw_catalog()
    developer = _service_policy(raw, "gemini-2.5-flash", "developer")
    developer.pop("pricing")
    with pytest.raises(ValidationError, match="pricing"):
        ModelCatalog.model_validate(raw)

    raw = _raw_catalog()
    developer = _service_policy(raw, "gemini-2.5-flash", "developer")
    source_refs = cast(list[str], developer["source_refs"])
    source_refs.append("enterprise_only_source")
    with pytest.raises(ValidationError, match="unknown sources"):
        ModelCatalog.model_validate(raw)


def test_companion_loader_fails_visibly_instead_of_returning_empty_policy() -> None:
    Filter._load_model_config.cache_clear()

    with pytest.raises(ModelCatalogError, match="must not be empty"):
        Filter._load_model_config("")

    with pytest.raises(ModelCatalogError, match=r"HTTP\(S\) URL"):
        Filter._load_model_config(str(CATALOG_PATH))
