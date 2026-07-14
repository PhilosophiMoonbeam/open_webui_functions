from __future__ import annotations

import copy
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError

sys.modules.setdefault("open_webui.models.chats", MagicMock())

from plugins.filters.gemini_manifold_companion import (
    MODEL_CATALOG_SCHEMA_VERSION,
    CatalogAppStateEnvelope,
    CatalogPricedRate,
    CatalogUnpricedRate,
    Filter,
    ModelCatalog,
    ModelCatalogError,
    _UniqueKeyLoader,
    canonical_catalog_bytes,
)

CATALOG_PATH = Path(__file__).parents[1] / "plugins" / "pipes" / "gemini_models.yaml"
EXPECTED_IDS = {
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-image",
    "gemini-3.1-flash-image",
}


def _raw_catalog() -> dict[str, object]:
    loaded = yaml.load(CATALOG_PATH.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    assert isinstance(loaded, dict)
    return loaded


def test_protocol_3_catalog_is_evidence_bound_and_actionable() -> None:
    catalog = ModelCatalog.model_validate(_raw_catalog())

    assert catalog.schema_version == MODEL_CATALOG_SCHEMA_VERSION == 3
    assert set(catalog.provider_claims) == EXPECTED_IDS
    assert set(catalog.product_authorizations) == EXPECTED_IDS
    assert set(catalog.runtime_models()) == EXPECTED_IDS
    assert catalog.provenance_sha256 == (
        "9ac80b5e8fdb19e969d1684079376fc240f26ea544aa599c4e0bc6ff2566fe10"
    )


def test_pricing_is_explicit_by_modality_cache_state_and_whole_prompt_threshold() -> None:
    catalog = ModelCatalog.model_validate(_raw_catalog())
    flash_lite = catalog.provider_claims["gemini-3.1-flash-lite"].pricing
    pro = catalog.provider_claims["gemini-3.1-pro-preview"].pricing
    image = catalog.provider_claims["gemini-3-pro-image"].pricing

    audio_input = flash_lite.input["audio"]
    audio_cache = flash_lite.cached_input["audio"]
    pro_text = pro.input["text"]
    assert isinstance(audio_input, CatalogPricedRate)
    assert isinstance(audio_cache, CatalogPricedRate)
    assert isinstance(pro_text, CatalogPricedRate)
    assert audio_input.tiers[0].price_per_million == 0.5
    assert audio_cache.tiers[0].price_per_million == 0.05
    assert isinstance(flash_lite.input["document"], CatalogUnpricedRate)
    assert pro_text.tiers[0].up_to_prompt_tokens == 200_000
    assert pro_text.tiers[1].price_per_million == 4.0
    assert isinstance(image.cached_input["image"], CatalogUnpricedRate)


@pytest.mark.parametrize(
    "text",
    [
        "schema_version: 3\nschema_version: 3\n",
        "schema_version: 3\nnested:\n  claim: one\n  claim: two\n",
        "base: &base\n  claim: one\nmerged:\n  <<: *base\n",
    ],
)
def test_yaml_loader_rejects_duplicate_and_merge_keys_before_parsing(text: str) -> None:
    with pytest.raises(ModelCatalogError, match="duplicate|merge"):
        yaml.load(text, Loader=_UniqueKeyLoader)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update(schema_version=2), "schema_version"),
        (
            lambda raw: raw["evidence"]["developer_pricing"].update(
                content_digest="sha256:" + "0" * 64
            ),
            "digest mismatch",
        ),
        (
            lambda raw: raw["provider_claims"]["gemini-2.5-flash"]["evidence"].update(
                pricing="thinking_controls"
            ),
            "wrong evidence kind",
        ),
        (
            lambda raw: raw["provider_claims"]["gemini-2.5-flash"]["evidence"].update(
                pricing="missing_evidence"
            ),
            "unknown evidence",
        ),
        (
            lambda raw: raw["sources"]["developer_pricing"].update(retrieved_at="2026-07-13"),
            "stale",
        ),
        (
            lambda raw: raw["provider_claims"]["gemini-2.5-flash"].update(
                model_id="gemini-2.5-flash-alias"
            ),
            "exact claim model IDs",
        ),
        (
            lambda raw: raw["freshness"].update(expires_after="2026-07-15"),
            "catalog expiry",
        ),
        (lambda raw: raw.update(unexpected=True), "extra_forbidden"),
        (
            lambda raw: raw["provider_claims"]["gemini-2.5-flash"]["pricing"]["cached_input"].pop(
                "audio"
            ),
            "cached pricing",
        ),
        (
            lambda raw: raw["provider_claims"]["gemini-2.5-flash"]["capabilities"].update(
                google_search=False
            ),
            "exceeds provider capability",
        ),
        (
            lambda raw: raw["product_authorizations"]["gemini-2.5-flash"]["interactions"].update(
                thinking={
                    "supported": True,
                    "control": "known",
                    "levels": ["low"],
                    "summaries": True,
                }
            ),
            "contradict",
        ),
    ],
)
def test_catalog_mutations_fail_closed(mutation, message: str) -> None:
    raw = copy.deepcopy(_raw_catalog())
    mutation(raw)

    with pytest.raises(ValidationError, match=message):
        ModelCatalog.model_validate(raw)


def test_canonical_digest_is_order_independent_and_semantic_mutations_change_it() -> None:
    raw = _raw_catalog()
    catalog = ModelCatalog.model_validate(raw)
    reordered = {key: raw[key] for key in reversed(raw)}
    reordered_catalog = ModelCatalog.model_validate(reordered)

    assert canonical_catalog_bytes(catalog) == canonical_catalog_bytes(reordered_catalog)
    envelope = CatalogAppStateEnvelope.from_catalog(catalog)
    mutated = envelope.model_dump(mode="json", exclude_none=False)
    mutated["payload"]["provider_claims"]["gemini-2.5-flash"]["limits"]["output_tokens"] += 1
    with pytest.raises(ValidationError, match="canonical digest mismatch"):
        CatalogAppStateEnvelope.model_validate(mutated)


def test_companion_loader_publishes_atomic_full_catalog_envelope() -> None:
    Filter._load_model_config.cache_clear()
    with (
        CATALOG_PATH.open("rb") as catalog_file,
        patch(
            "plugins.filters.gemini_manifold_companion.urllib.request.urlopen",
            return_value=catalog_file,
        ),
    ):
        envelope = Filter._load_model_config("https://example.test/gemini_models.yaml")

    assert envelope.schema_version == 3
    assert set(envelope.payload.runtime_models()) == EXPECTED_IDS
    assert envelope.canonical_digest.startswith("sha256:")


def test_companion_loader_fails_visibly() -> None:
    Filter._load_model_config.cache_clear()
    with pytest.raises(ModelCatalogError, match="must not be empty"):
        Filter._load_model_config("")
    with pytest.raises(ModelCatalogError, match=r"HTTP\(S\) URL"):
        Filter._load_model_config(str(CATALOG_PATH))
