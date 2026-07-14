from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import yaml

PROVENANCE_PATH = (
    Path(__file__).parents[1] / "docs" / "development" / "gemini-model-provenance-v1.yaml"
)
CATALOG_PATH = Path(__file__).parents[1] / "plugins" / "pipes" / "gemini_models.yaml"

RETAINED_MODEL_IDS = {
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
REMOVED_CURRENT_IDS = {
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
}
SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def _load(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return cast(dict[str, object], loaded)


def test_provenance_sources_are_dated_exact_and_digest_bound() -> None:
    provenance = _load(PROVENANCE_PATH)
    assert provenance["schema_version"] == 1
    assert provenance["researched_at"] == "2026-07-14"

    sources = cast(dict[str, dict[str, object]], provenance["sources"])
    assert len(sources) >= 20
    for source in sources.values():
        assert source["kind"]
        assert source["url"]
        assert source["section"]
        assert source["retrieved_at"] == "2026-07-14"
        assert SHA256_PATTERN.fullmatch(cast(str, source["content_digest"]))

    evidence_policy = cast(dict[str, object], provenance["evidence_policy"])
    assert evidence_policy["freshness_days"] == 30
    assert evidence_policy["expires_after"] == "2026-08-13"
    assert len(cast(list[str], evidence_policy["renewal_procedure"])) >= 5


def test_every_retained_model_has_independent_claim_evidence() -> None:
    provenance = _load(PROVENANCE_PATH)
    sources = cast(dict[str, object], provenance["sources"])
    models = cast(dict[str, dict[str, object]], provenance["models"])
    assert set(models) == RETAINED_MODEL_IDS

    for model in models.values():
        assert model["disposition"] == "retain"
        evidence = cast(dict[str, object], model["evidence"])
        assert set(evidence) == {"availability", "model", "thinking", "pricing", "product"}
        refs = [
            evidence["availability"],
            evidence["model"],
            evidence["thinking"],
            evidence["pricing"],
        ]
        refs.extend(cast(list[str], evidence["product"]))
        assert all(ref in sources for ref in refs)

        provider = cast(dict[str, object], model["provider"])
        assert provider["lifecycle"] in {"stable", "preview"}
        assert cast(list[str], provider["input_modalities"])
        assert cast(list[str], provider["output_modalities"])
        limits = cast(dict[str, int], provider["limits"])
        assert limits["input_tokens"] > 0
        assert limits["output_tokens"] > 0
        assert cast(dict[str, object], provider["capabilities"])

        thinking = cast(dict[str, object], model["thinking"])
        assert thinking["control"] in {"known", "unknown"}
        if thinking["control"] == "unknown":
            assert thinking["levels"] == []
            assert thinking["reason"]
        else:
            assert cast(list[str], thinking["levels"])

        pricing = cast(dict[str, object], model["standard_pricing"])
        assert pricing["threshold_semantics"] in {"none", "whole_prompt_length_not_marginal"}
        assert cast(dict[str, float], pricing["input_per_million"])
        assert cast(dict[str, float], pricing["output_per_million"])
        assert isinstance(pricing["free_tier"], bool)
        free_tier_tools = cast(dict[str, str], pricing["free_tier_tools"])
        assert set(free_tier_tools) == {"google_search", "google_maps"}

        authorization = cast(dict[str, object], model["product_authorization"])
        assert authorization["discovery"] == "allow"
        assert authorization["reasoning_controls"] in {"allow", "deny"}
        assert authorization["response_format"] in {"allow", "deny"}
        assert authorization["custom_function_calling"] in {"allow", "deny"}
        tools = cast(dict[str, str], authorization["tools"])
        assert set(tools) == {
            "google_search",
            "google_maps",
            "code_execution",
            "url_context",
            "file_search",
        }
        assert tools["file_search"] == "deny"


def test_interactions_table_and_product_scope_are_not_conflated() -> None:
    provenance = _load(PROVENANCE_PATH)
    supported = cast(dict[str, object], provenance["interactions_supported_ids"])
    provider_models = set(cast(list[str], supported["models"]))
    provider_agents = set(cast(list[str], supported["agents"]))
    exclusions = cast(dict[str, str], provenance["product_scope_exclusions"])

    assert RETAINED_MODEL_IDS < provider_models
    assert set(exclusions) == (provider_models - RETAINED_MODEL_IDS) | provider_agents
    assert all(exclusions.values())


def test_every_ledger_evidence_reference_resolves() -> None:
    provenance = _load(PROVENANCE_PATH)
    sources = cast(dict[str, object], provenance["sources"])

    def assert_refs(refs: object) -> None:
        assert all(ref in sources for ref in cast(list[str], refs))

    shared = cast(dict[str, dict[str, object]], provenance["shared_product_claims"])
    for claim in shared.values():
        assert_refs(claim["evidence"])

    reconciliation = cast(
        dict[str, dict[str, object]], provenance["current_catalog_reconciliation"]
    )
    for decision in reconciliation.values():
        if "evidence" in decision:
            assert_refs(decision["evidence"])

    new_entries = cast(dict[str, dict[str, object]], provenance["new_catalog_entries"])
    for decision in new_entries.values():
        assert_refs(decision["evidence"])

    usage = cast(dict[str, dict[str, object]], provenance["unpriced_usage_ledger"])
    for claim in usage.values():
        assert_refs(claim["evidence"])


def test_current_catalog_has_an_explicit_reconciliation_decision() -> None:
    provenance = _load(PROVENANCE_PATH)
    catalog = _load(CATALOG_PATH)
    current_models = set(cast(dict[str, object], catalog["provider_claims"]))
    reconciliation = cast(
        dict[str, dict[str, object]], provenance["current_catalog_reconciliation"]
    )
    new_entries = cast(dict[str, dict[str, object]], provenance["new_catalog_entries"])

    retained = {
        model_id
        for model_id, model in cast(dict[str, dict[str, object]], provenance["models"]).items()
        if model["disposition"] == "retain"
    }
    assert current_models == retained
    assert current_models <= set(reconciliation) | set(new_entries)
    assert {
        model_id for model_id, record in reconciliation.items() if record["disposition"] == "remove"
    } == REMOVED_CURRENT_IDS
    assert set(new_entries) == {"gemini-3-pro-image", "gemini-3.1-flash-image"}


def test_known_catalog_mismatches_are_durable_handoff_inputs() -> None:
    provenance = _load(PROVENANCE_PATH)
    models = cast(dict[str, dict[str, object]], provenance["models"])
    reconciliation = cast(
        dict[str, dict[str, object]], provenance["current_catalog_reconciliation"]
    )

    assert models["gemini-3.1-flash-lite"]["thinking"] == {
        "control": "unknown",
        "levels": [],
        "reason": "No exact gemini-3.1-flash-lite row exists in the current Interactions controlling-thinking table.",
    }
    assert cast(dict[str, object], models["gemini-2.5-pro"]["thinking"])["levels"] == [
        "low",
        "medium",
        "high",
    ]
    assert cast(dict[str, object], models["gemini-2.5-flash"]["thinking"])["levels"] == [
        "low",
        "medium",
        "high",
    ]
    assert cast(dict[str, object], models["gemini-2.5-flash-lite"]["thinking"])["levels"] == [
        "low",
        "medium",
        "high",
    ]
    assert cast(dict[str, object], models["gemini-2.5-flash-lite"]["thinking"])["default"] == "off"

    flash_pricing = cast(dict[str, object], models["gemini-2.5-flash"]["standard_pricing"])
    assert flash_pricing["input_per_million"] == {"text_image_video": 0.30, "audio": 1.00}
    lite_pricing = cast(dict[str, object], models["gemini-2.5-flash-lite"]["standard_pricing"])
    assert lite_pricing["input_per_million"] == {"text_image_video": 0.10, "audio": 0.30}
    assert "remove_unproven_document_input" in cast(
        list[str], reconciliation["gemini-2.5-flash"]["changes"]
    )


def test_threshold_and_unpriced_usage_semantics_are_explicit() -> None:
    provenance = _load(PROVENANCE_PATH)
    models = cast(dict[str, dict[str, object]], provenance["models"])
    for model_id in ("gemini-3.1-pro-preview", "gemini-2.5-pro"):
        pricing = cast(dict[str, object], models[model_id]["standard_pricing"])
        assert pricing["threshold_semantics"] == "whole_prompt_length_not_marginal"
        assert pricing["threshold_tokens"] == 200000

    usage = cast(dict[str, dict[str, object]], provenance["unpriced_usage_ledger"])
    assert usage["document_input_tokens"]["state"] == "unknown"
    assert usage["tool_use_prompt_tokens"]["state"] == "unknown"
    assert usage["cached_tokens"]["state"] == "known_by_model_and_input_modality"
    assert usage["image_output_tokens"]["state"] == "known_for_retained_image_models"


def _expected_tiers(
    pricing: dict[str, object],
    rates: dict[str, float],
    group: str,
) -> list[dict[str, object]]:
    if pricing["threshold_semantics"] == "none":
        return [{"up_to_prompt_tokens": None, "price_per_million": rates[group]}]
    return [
        {
            "up_to_prompt_tokens": pricing["threshold_tokens"],
            "price_per_million": rates["at_or_below_threshold"],
        },
        {
            "up_to_prompt_tokens": None,
            "price_per_million": rates["above_threshold"],
        },
    ]


def _pricing_group(modality: str, rates: dict[str, float]) -> str | None:
    if modality == "document":
        return None
    if "at_or_below_threshold" in rates:
        return "at_or_below_threshold"
    if "all_supported_input_modalities" in rates:
        return "all_supported_input_modalities"
    if modality == "audio" and "audio" in rates:
        return "audio"
    if modality in {"text", "image", "video"} and "text_image_video" in rates:
        return "text_image_video"
    if modality in {"text", "image"} and "text_image" in rates:
        return "text_image"
    if modality == "image" and "image" in rates:
        return "image"
    if modality == "text" and "text_and_thinking" in rates:
        return "text_and_thinking"
    return None


def test_protocol_3_catalog_matches_every_provenance_claim_model_by_model() -> None:
    provenance = _load(PROVENANCE_PATH)
    catalog = _load(CATALOG_PATH)
    models = cast(dict[str, dict[str, object]], provenance["models"])
    providers = cast(dict[str, dict[str, object]], catalog["provider_claims"])
    products = cast(dict[str, dict[str, object]], catalog["product_authorizations"])

    assert set(providers) == set(products) == set(models)
    for model_id, model in models.items():
        provider = cast(dict[str, object], model["provider"])
        claim = providers[model_id]
        authorization = cast(dict[str, object], model["product_authorization"])
        product = products[model_id]
        interactions = cast(dict[str, object], product["interactions"])

        assert claim["model_id"] == product["model_id"] == model_id
        assert claim["availability"] == "supported"
        assert claim["lifecycle"] == provider["lifecycle"]
        assert claim["limits"] == provider["limits"]
        assert claim["content"] == {
            "inputs": provider["input_modalities"],
            "outputs": provider["output_modalities"],
        }
        provider_capabilities = cast(dict[str, object], provider["capabilities"])
        assert claim["capabilities"] == {
            key: value is not False
            for key, value in provider_capabilities.items()
            if key != "caching"
        }

        thinking = cast(dict[str, object], model["thinking"])
        expected_thinking = {
            "supported": thinking["control"] == "known",
            "control": thinking["control"],
            "levels": thinking["levels"],
            "summaries": thinking["control"] == "known",
        }
        assert claim["thinking"] == expected_thinking
        assert interactions["thinking"] == expected_thinking

        assert product["discovery"] == authorization["discovery"]
        assert interactions["store"] is True
        assert interactions["files"] is True
        assert interactions["external_urls"] is bool(provider_capabilities["url_context"])
        assert interactions["response_format"] is (authorization["response_format"] == "allow")
        assert interactions["custom_function_calling"] is (
            authorization["custom_function_calling"] == "allow"
        )
        assert interactions["tools"] == {
            key: value == "allow"
            for key, value in cast(dict[str, str], authorization["tools"]).items()
        }

        pricing = cast(dict[str, object], model["standard_pricing"])
        catalog_pricing = cast(dict[str, object], claim["pricing"])
        assert catalog_pricing["free_tier"] == pricing["free_tier"]
        assert catalog_pricing["threshold_basis"] == "total_input_tokens_including_cached"
        free_tier_tools = cast(dict[str, str], pricing["free_tier_tools"])
        expected_exclusions = [
            tool
            for tool, state in free_tier_tools.items()
            if state == "paid_only" and provider_capabilities[tool]
        ]
        assert catalog_pricing["excluded_features"] == expected_exclusions

        for catalog_key, provenance_key in (
            ("input", "input_per_million"),
            ("cached_input", "cached_input_per_million"),
            ("output", "output_per_million"),
        ):
            catalog_rates = cast(dict[str, dict[str, object]], catalog_pricing[catalog_key])
            raw_rates = pricing[provenance_key]
            for modality, rate in catalog_rates.items():
                rates = cast(dict[str, float], raw_rates) if raw_rates is not None else {}
                group = _pricing_group(modality, rates)
                if group is None:
                    assert rate["state"] == "unpriced"
                    assert rate["reason"]
                else:
                    assert rate == {
                        "state": "priced",
                        "tiers": _expected_tiers(pricing, rates, group),
                    }
