"""
title: Gemini Manifold Companion
id: gemini_manifold_companion
description: A companion filter for "Gemini Manifold google_genai" pipe providing enhanced functionality.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 3.0.0
"""

VERSION = "3.0.0"

# This filter can detect that a feature like web search or code execution is enabled in the front-end,
# set the feature back to False so Open WebUI does not run it's own logic and then
# pass custom values to "Gemini Manifold google_genai" that signal which feature was enabled and intercepted.

import asyncio
import copy
import functools
import hashlib
import json
import sys
import time
import urllib.request
from collections.abc import Awaitable, Callable
from datetime import date, timedelta
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import aiohttp
import pydantic_core
import yaml
from fastapi import Request
from fastapi.datastructures import State
from loguru import logger
from open_webui.models.chats import Chats
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

if TYPE_CHECKING:
    from loguru import Record
    from loguru._handler import Handler  # type: ignore
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.

# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main log.
log = logger.bind(auditable=False)

DEFAULT_MODEL_CONFIG_PATH = "https://raw.githubusercontent.com/suurt8ll/open_webui_functions/gemini-suite/v3.0.0/plugins/pipes/gemini_models.yaml"
MODEL_CATALOG_SCHEMA_VERSION = 3
MODEL_CATALOG_PROVENANCE_SHA256 = "5684d034b820bf5a99b28342263035ea3684f132bc0b6bfa3721678b55d9536e"
GROUNDING_ENVELOPE_PROTOCOL_VERSION = 1

# Default timeout for URL resolution
# TODO: Move to Pipe.Valves.
DEFAULT_URL_TIMEOUT = aiohttp.ClientTimeout(total=10)  # 10 seconds total timeout


class ModelCatalogError(ValueError):
    """Raised when the remote model policy is unavailable or incompatible."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys and merge-key expansion."""

    def flatten_mapping(self, node: yaml.MappingNode) -> None:
        if any(key.tag == "tag:yaml.org,2002:merge" for key, _ in node.value):
            raise ModelCatalogError("YAML merge keys are forbidden in the model catalog")
        super().flatten_mapping(node)


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ModelCatalogError(f"duplicate YAML key is forbidden: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class _CatalogModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


CatalogContentKind = Literal["text", "image", "video", "audio", "document"]
CatalogImageResolution = Literal["512", "1K", "2K", "4K"]
CatalogImageAspectRatio = Literal[
    "1:1",
    "1:4",
    "1:8",
    "2:3",
    "3:2",
    "3:4",
    "4:1",
    "4:3",
    "4:5",
    "5:4",
    "8:1",
    "9:16",
    "16:9",
    "21:9",
]
CatalogEvidenceKind = Literal[
    "provider_interactions_availability",
    "provider_interactions_feature",
    "provider_model_lifecycle",
    "provider_interactions_thinking",
    "provider_developer_pricing",
    "provider_lifecycle_changelog",
    "provider_interactions_input_transport",
    "provider_model_properties",
    "repository_implementation",
    "repository_test_evidence",
]


class CatalogFreshness(_CatalogModel):
    freshness_days: int = Field(gt=0)
    retrieved_at: date
    expires_after: date

    @model_validator(mode="after")
    def validate_window(self) -> "CatalogFreshness":
        if self.expires_after != self.retrieved_at + timedelta(days=self.freshness_days):
            raise ValueError("catalog expiry must equal retrieved_at plus freshness_days")
        if self.retrieved_at > date.today():
            raise ValueError("catalog retrieval date must not be in the future")
        if date.today() > self.expires_after:
            raise ValueError("catalog evidence is expired")
        return self


class CatalogSource(_CatalogModel):
    kind: CatalogEvidenceKind
    url: str = Field(min_length=1)
    section: str = Field(min_length=1)
    published_last_updated: date | None
    retrieved_at: date
    content_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    subject_model_ids: frozenset[str] = Field(default_factory=frozenset)

    @model_validator(mode="after")
    def validate_dates(self) -> "CatalogSource":
        if self.retrieved_at > date.today():
            raise ValueError("source retrieval date must not be in the future")
        if self.published_last_updated is not None:
            if self.published_last_updated > self.retrieved_at:
                raise ValueError("source publication date must not follow retrieval")
        return self


class CatalogEvidence(_CatalogModel):
    source: str = Field(min_length=1)
    content_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class CatalogLimits(_CatalogModel):
    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)


class CatalogContent(_CatalogModel):
    inputs: frozenset[CatalogContentKind]
    outputs: frozenset[CatalogContentKind]


class CatalogImageOutput(_CatalogModel):
    resolutions: tuple[CatalogImageResolution, ...] = Field(min_length=1)
    aspect_ratios: tuple[CatalogImageAspectRatio, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_options(self) -> "CatalogImageOutput":
        if len(self.resolutions) != len(set(self.resolutions)):
            raise ValueError("image resolutions must be unique")
        if len(self.aspect_ratios) != len(set(self.aspect_ratios)):
            raise ValueError("image aspect ratios must be unique")
        return self


class CatalogThinking(_CatalogModel):
    supported: bool
    control: Literal["known", "unknown"]
    levels: frozenset[Literal["minimal", "low", "medium", "high"]]
    summaries: bool

    @model_validator(mode="after")
    def validate_support(self) -> "CatalogThinking":
        if self.supported != bool(self.levels):
            raise ValueError("thinking levels must be present exactly when thinking is supported")
        if self.summaries and not self.supported:
            raise ValueError("thinking summaries require thinking support")
        if (self.control == "known") != self.supported:
            raise ValueError("known thinking control must match supported controls")
        return self


class CatalogTools(_CatalogModel):
    google_search: bool
    google_maps: bool
    code_execution: bool
    url_context: bool
    file_search: bool


class CatalogInteractions(_CatalogModel):
    store: bool
    response_format: bool
    thinking: CatalogThinking
    custom_function_calling: bool
    files: bool
    external_urls: bool
    tools: CatalogTools


class CatalogPriceTier(_CatalogModel):
    up_to_prompt_tokens: int | None = Field(default=None, gt=0)
    price_per_million: float = Field(ge=0)


class CatalogPricedRate(_CatalogModel):
    state: Literal["priced"]
    tiers: tuple[CatalogPriceTier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tiers(self) -> "CatalogPricedRate":
        thresholds = [tier.up_to_prompt_tokens for tier in self.tiers]
        if thresholds[-1] is not None:
            raise ValueError("priced rate must end with an unbounded tier")
        bounded = [threshold for threshold in thresholds[:-1] if threshold is not None]
        if len(bounded) != len(thresholds) - 1 or bounded != sorted(set(bounded)):
            raise ValueError("pricing thresholds must be unique and ascending")
        return self


class CatalogUnpricedRate(_CatalogModel):
    state: Literal["unpriced"]
    reason: str = Field(min_length=1)


CatalogRate = Annotated[
    CatalogPricedRate | CatalogUnpricedRate,
    Field(discriminator="state"),
]


class CatalogPricing(_CatalogModel):
    free_tier: bool
    excluded_features: frozenset[Literal["google_search", "google_maps"]]
    threshold_basis: Literal["total_input_tokens_including_cached"]
    input: dict[CatalogContentKind, CatalogRate]
    cached_input: dict[CatalogContentKind, CatalogRate]
    output: dict[CatalogContentKind, CatalogRate]

    @model_validator(mode="after")
    def validate_modalities(self) -> "CatalogPricing":
        if not self.input or not self.output:
            raise ValueError("pricing must declare input and output modalities")
        if set(self.cached_input) != set(self.input):
            raise ValueError("cached pricing must cover every input modality")
        return self


class CatalogSupportedService(_CatalogModel):
    availability: Literal["supported"]
    lifecycle: Literal["stable", "preview"]
    limits: CatalogLimits
    content: CatalogContent
    image_output: CatalogImageOutput | None = None
    interactions: CatalogInteractions
    pricing: CatalogPricing

    @model_validator(mode="after")
    def validate_capability_dependencies(self) -> "CatalogSupportedService":
        if set(self.content.inputs) != set(self.pricing.input):
            raise ValueError("input pricing must exactly cover authorized input modalities")
        if set(self.content.outputs) != set(self.pricing.output):
            raise ValueError("output pricing must exactly cover authorized output modalities")
        if self.interactions.external_urls and not self.interactions.tools.url_context:
            raise ValueError("external URL input requires the URL-context tool")
        if ("image" in self.content.outputs) != (self.image_output is not None):
            raise ValueError("image output options must exist exactly for image-output models")
        return self


class CatalogUnavailableService(_CatalogModel):
    availability: Literal["unsupported", "unverified"]
    reason: str = Field(min_length=1)


CatalogService = Annotated[
    CatalogSupportedService | CatalogUnavailableService,
    Field(discriminator="availability"),
]


class CatalogServices(_CatalogModel):
    developer: CatalogService
    enterprise: CatalogService


class CatalogModel(_CatalogModel):
    services: CatalogServices

    @model_validator(mode="after")
    def validate_somewhere_actionable(self) -> "CatalogModel":
        if all(
            service.availability == "unsupported"
            for service in (self.services.developer, self.services.enterprise)
        ):
            raise ValueError("a catalog model must be supported or pending verification somewhere")
        return self


class CatalogClaimEvidence(_CatalogModel):
    availability: str
    properties: str
    thinking: str
    pricing: str
    image_output: str | None = None


class CatalogProviderCapabilities(_CatalogModel):
    function_calling: bool
    structured_outputs: bool
    thinking: bool
    google_search: bool
    google_maps: bool
    code_execution: bool
    url_context: bool
    file_search: bool


class CatalogProviderClaim(_CatalogModel):
    model_id: str
    availability: Literal["supported"]
    lifecycle: Literal["stable", "preview"]
    limits: CatalogLimits
    content: CatalogContent
    image_output: CatalogImageOutput | None = None
    capabilities: CatalogProviderCapabilities
    thinking: CatalogThinking
    pricing: CatalogPricing
    evidence: CatalogClaimEvidence


class CatalogProductAuthorization(_CatalogModel):
    model_id: str
    discovery: Literal["allow"]
    interactions: CatalogInteractions
    evidence: tuple[str, ...] = Field(min_length=1)


class ModelCatalog(_CatalogModel):
    schema_version: Literal[3]
    provenance_sha256: Literal["5684d034b820bf5a99b28342263035ea3684f132bc0b6bfa3721678b55d9536e"]
    freshness: CatalogFreshness
    sources: dict[str, CatalogSource] = Field(min_length=1)
    evidence: dict[str, CatalogEvidence] = Field(min_length=1)
    provider_claims: dict[str, CatalogProviderClaim] = Field(min_length=1)
    product_authorizations: dict[str, CatalogProductAuthorization] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_claims(self) -> "ModelCatalog":
        if set(self.provider_claims) != set(self.product_authorizations):
            raise ValueError("provider claims and product authorizations must have identical IDs")
        invalid = [
            model_id for model_id in self.provider_claims if not model_id.startswith("gemini-")
        ]
        if invalid:
            raise ValueError(f"invalid Gemini model ids: {invalid}")
        expected_kinds: dict[str, CatalogEvidenceKind] = {
            "availability": "provider_interactions_availability",
            "properties": "provider_model_properties",
            "thinking": "provider_interactions_thinking",
            "pricing": "provider_developer_pricing",
        }
        for evidence_id, evidence in self.evidence.items():
            source = self.sources.get(evidence.source)
            if source is None:
                raise ValueError(f"evidence '{evidence_id}' references an unknown source")
            if evidence.content_digest != source.content_digest:
                raise ValueError(f"evidence '{evidence_id}' digest mismatch")
            if source.retrieved_at < self.freshness.retrieved_at:
                raise ValueError(f"evidence '{evidence_id}' is stale")
        for model_id, claim in self.provider_claims.items():
            authorization = self.product_authorizations[model_id]
            if claim.model_id != model_id or authorization.model_id != model_id:
                raise ValueError("catalog keys and exact claim model IDs must match")
            for field_name, expected_kind in expected_kinds.items():
                evidence_id = getattr(claim.evidence, field_name)
                evidence = self.evidence.get(evidence_id)
                if evidence is None:
                    raise ValueError(f"model '{model_id}' references unknown evidence")
                source = self.sources[evidence.source]
                if source.kind != expected_kind:
                    raise ValueError(f"model '{model_id}' has wrong evidence kind for {field_name}")
                if source.subject_model_ids and model_id not in source.subject_model_ids:
                    raise ValueError(f"model '{model_id}' evidence has an exact-ID mismatch")
            image_evidence_id = claim.evidence.image_output
            if (claim.image_output is None) != (image_evidence_id is None):
                raise ValueError("image output claims require exact image-output evidence")
            if image_evidence_id is not None:
                image_evidence = self.evidence.get(image_evidence_id)
                if image_evidence is None:
                    raise ValueError(f"model '{model_id}' references unknown image evidence")
                image_source = self.sources[image_evidence.source]
                if image_source.kind != "provider_interactions_feature":
                    raise ValueError(f"model '{model_id}' has wrong image evidence kind")
                if model_id not in image_source.subject_model_ids:
                    raise ValueError(f"model '{model_id}' image evidence has an exact-ID mismatch")
            for evidence_id in authorization.evidence:
                evidence = self.evidence.get(evidence_id)
                if evidence is None:
                    raise ValueError(f"model '{model_id}' references unknown product evidence")
                if self.sources[evidence.source].kind not in {
                    "repository_implementation",
                    "repository_test_evidence",
                }:
                    raise ValueError(f"model '{model_id}' product evidence has wrong kind")
            if authorization.interactions.thinking != claim.thinking:
                raise ValueError("provider thinking and product authorization contradict")
            tools = authorization.interactions.tools
            capabilities = claim.capabilities
            if claim.thinking.supported and not capabilities.thinking:
                raise ValueError("thinking controls exceed provider capability")
            if authorization.interactions.response_format and not capabilities.structured_outputs:
                raise ValueError("product response format exceeds provider capability")
            if (
                authorization.interactions.custom_function_calling
                and not capabilities.function_calling
            ):
                raise ValueError("product function calling exceeds provider capability")
            for tool_name in (
                "google_search",
                "google_maps",
                "code_execution",
                "url_context",
                "file_search",
            ):
                if getattr(tools, tool_name) and not getattr(capabilities, tool_name):
                    raise ValueError(f"product {tool_name} exceeds provider capability")
            if tools.file_search:
                raise ValueError("file_search is not product-authorized")
            if authorization.interactions.external_urls and not tools.url_context:
                raise ValueError("external URL input requires URL context")
            if authorization.interactions.files and not claim.content.inputs:
                raise ValueError("files require an authorized input modality")
        return self

    def runtime_models(self) -> dict[str, CatalogModel]:
        unavailable = CatalogUnavailableService(
            availability="unverified",
            reason="No credential-backed Enterprise Interactions model and capability evidence is recorded.",
        )
        return {
            model_id: CatalogModel(
                services=CatalogServices(
                    developer=CatalogSupportedService(
                        availability="supported",
                        lifecycle=claim.lifecycle,
                        limits=claim.limits,
                        content=claim.content,
                        image_output=claim.image_output,
                        interactions=self.product_authorizations[model_id].interactions,
                        pricing=claim.pricing,
                    ),
                    enterprise=unavailable,
                )
            )
            for model_id, claim in self.provider_claims.items()
        }


def _canonicalize_catalog(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _canonicalize_catalog(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_canonicalize_catalog(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_canonicalize_catalog(item) for item in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def canonical_catalog_bytes(catalog: ModelCatalog) -> bytes:
    normalized = _canonicalize_catalog(catalog.model_dump(mode="python", exclude_none=False))
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class CatalogAppStateEnvelope(_CatalogModel):
    schema_version: Literal[3]
    canonical_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    payload: ModelCatalog

    @model_validator(mode="after")
    def validate_digest(self) -> "CatalogAppStateEnvelope":
        expected = "sha256:" + hashlib.sha256(canonical_catalog_bytes(self.payload)).hexdigest()
        if self.canonical_digest != expected:
            raise ValueError("catalog canonical digest mismatch")
        return self

    @classmethod
    def from_catalog(cls, catalog: ModelCatalog) -> "CatalogAppStateEnvelope":
        digest = "sha256:" + hashlib.sha256(canonical_catalog_bytes(catalog)).hexdigest()
        return cls(schema_version=3, canonical_digest=digest, payload=catalog)


class GroundingTextBlock(_CatalogModel):
    step_index: int
    content_index: int
    text: str


class GroundingReviewSnippet(_CatalogModel):
    review_id: str | None = None
    title: str | None = None
    uri: str | None = None


class GroundingSource(_CatalogModel):
    id: str
    kind: Literal["url", "file", "place"]
    uri: str | None = None
    title: str | None = None
    file_name: str | None = None
    media_id: str | None = None
    page_number: int | None = None
    source: str | None = None
    place_id: str | None = None
    custom_metadata: dict[str, JsonValue] | None = None
    review_snippets: list[GroundingReviewSnippet] = Field(default_factory=list)


class GroundingCitation(_CatalogModel):
    source_id: str
    block_index: int
    start: int
    end: int
    index_unit: Literal["provider", "utf8_bytes", "unicode_codepoints"]


class GroundingToolRecord(_CatalogModel):
    tool: Literal["google_search", "url_context", "google_maps", "file_search", "retrieval"]
    phase: Literal["call", "result"]
    step_index: int
    call_id: str | None = None
    queries: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    search_type: str | None = None
    retrieval_type: str | None = None
    statuses: list[str] = Field(default_factory=list)
    search_suggestions: list[str] = Field(default_factory=list)
    places: list[GroundingSource] = Field(default_factory=list)
    widget_context_tokens: list[str] = Field(default_factory=list)
    is_error: bool | None = None


class GroundingDiagnostic(_CatalogModel):
    code: str
    detail: str
    step_index: int | None = None


class GroundingEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1]
    visible_content_sha256: str
    grounded_text_sha256: str
    text_blocks: list[GroundingTextBlock] = Field(default_factory=list)
    sources: list[GroundingSource] = Field(default_factory=list)
    citations: list[GroundingCitation] = Field(default_factory=list)
    tool_records: list[GroundingToolRecord] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    tool_errors: list[str] = Field(default_factory=list)
    diagnostics: list[GroundingDiagnostic] = Field(default_factory=list)


class EventEmitter:
    """
    A unified, thread-safe event emitter for Open WebUI plugins.
    Uses internal queues to guarantee ordered, non-blocking delivery of websocket events.
    Includes an idle timeout to prevent memory leaks from orphaned instances.
    """

    def __init__(
        self,
        event_emitter: Callable[["Event"], Awaitable[None]] | None,
        *,
        status_mode: str = "visible",
        idle_timeout: float = 3600.0,
    ):
        self._emitter = event_emitter
        self.status_mode = status_mode
        self.start_time = time.monotonic()

        # Used by external garbage collection to detect dead instances
        self.is_abandoned: bool = False
        self._idle_timeout = idle_timeout

        self._queue: asyncio.Queue[Event | None] = asyncio.Queue()
        self._toast_queue: asyncio.Queue[Event | None] = asyncio.Queue()

        self._worker_task: asyncio.Task | None = None
        self._toast_worker_task: asyncio.Task | None = None

        if self._emitter is not None:
            self._worker_task = asyncio.create_task(self._process_queue(self._queue))
            self._toast_worker_task = asyncio.create_task(self._process_queue(self._toast_queue))

    async def _process_queue(self, queue: asyncio.Queue["Event | None"]) -> None:
        """
        A generic consumer for event queues.
        Processes items sequentially until a None poison pill is encountered
        or the idle timeout is reached.
        """
        while True:
            try:
                # The timeout only applies to the waiting period for new events.
                # If an event takes a long time to process below, it won't trigger this.
                event = await asyncio.wait_for(queue.get(), timeout=self._idle_timeout)
            except TimeoutError:
                # If no events arrive within the timeout window, assume the parent
                # request was unexpectedly dropped. Set the flag for external cleanup.
                self.is_abandoned = True
                break

            if event is None:
                queue.task_done()
                break

            if self._emitter:
                try:
                    await self._emitter(event)
                except asyncio.CancelledError:
                    log.warning("Open WebUI event callback was cancelled; dropping the event.")
                except Exception:
                    log.error("Error in EventEmitter background worker")

            queue.task_done()

    def _enqueue(self, event: "Event", is_toast: bool = False) -> None:
        """Pushes a new event into the appropriate queue without blocking."""
        if self._emitter is None:
            return

        target_queue = self._toast_queue if is_toast else self._queue
        target_queue.put_nowait(event)

    async def flush(self) -> None:
        """Blocks until all currently queued events across all queues have been processed."""
        self._drain_queue_if_worker_stopped(self._queue, self._worker_task)
        self._drain_queue_if_worker_stopped(self._toast_queue, self._toast_worker_task)
        await asyncio.gather(self._queue.join(), self._toast_queue.join())

    async def shutdown(self) -> None:
        """Sends the poison pill to all active workers and waits for them to finish."""
        tasks_to_await = []

        if self._worker_task and not self._worker_task.done():
            self._queue.put_nowait(None)
            tasks_to_await.append(self._worker_task)

        if self._toast_worker_task and not self._toast_worker_task.done():
            self._toast_queue.put_nowait(None)
            tasks_to_await.append(self._toast_worker_task)
        else:
            self._drain_queue_if_worker_stopped(self._toast_queue, self._toast_worker_task)

        if self._worker_task is None or self._worker_task.done():
            self._drain_queue_if_worker_stopped(self._queue, self._worker_task)

        if tasks_to_await:
            await asyncio.gather(*tasks_to_await)

    @staticmethod
    def _drain_queue_if_worker_stopped(
        queue: asyncio.Queue["Event | None"], worker: asyncio.Task | None
    ) -> None:
        if worker is not None and not worker.done():
            return
        while not queue.empty():
            queue.get_nowait()
            queue.task_done()

    def emit_toast(
        self,
        msg: str,
        type: Literal["info", "success", "warning", "error"] = "info",
    ) -> None:
        event: NotificationEvent = {
            "type": "notification",
            "data": {"type": type, "content": msg},
        }
        self._enqueue(event, is_toast=True)

    def emit_status(
        self,
        description: str,
        done: bool = False,
        hidden: bool = False,
        *,
        is_successful_finish: bool = False,
        is_thought: bool = False,
        indent_level: int = 0,
    ) -> None:
        if self.status_mode == "disable":
            return
        if self.status_mode == "hidden_compact" and is_thought:
            return

        if "visible_timed" in self.status_mode:
            elapsed = time.monotonic() - self.start_time
            description = f"{description} (+{elapsed:.2f}s)"

        final_hidden = hidden or (
            self.status_mode in ("hidden_compact", "hidden_detailed") and is_successful_finish
        )

        if not final_hidden and indent_level > 0:
            description = f"{'- ' * indent_level}{description}"

        event: StatusEvent = {
            "type": "status",
            "data": {"description": description, "done": done, "hidden": final_hidden},
        }
        self._enqueue(event)

    def emit_completion(
        self,
        content: str | None = None,
        done: bool = False,
        error: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        data: dict[str, Any] = {"done": done}
        if content is not None:
            data["content"] = content
        if error is not None:
            data["error"] = {"detail": error}
        if usage is not None:
            data["usage"] = usage

        event: ChatCompletionEvent = {
            "type": "chat:completion",
            "data": cast(Any, data),
        }
        self._enqueue(event)

    def emit_sources(self, source_data: "Source") -> None:
        event: CitationEvent = {
            "type": "source",
            "data": {
                "source": source_data["source"],
                "document": source_data["document"],
                "metadata": source_data["metadata"],
            },
        }
        self._enqueue(event)

    def emit_error(self, error_msg: str, exception: bool = True) -> None:
        log.opt(depth=1, exception=exception).error(error_msg)
        self.emit_completion(error=f"\n{error_msg}", done=True)

    def emit_grounding_queries(self, queries: list[str]) -> None:
        if not queries:
            return
        event: StatusEvent = {
            "type": "status",
            "data": {
                "action": "web_search_queries_generated",
                "queries": queries,
                "done": False,
            },
        }
        self._enqueue(event)


class Filter:
    class Valves(BaseModel):
        BYPASS_BACKEND_RAG: bool = Field(
            default=True,
            description="""Decide if you want ot bypass Open WebUI's RAG and send your documents directly to Google API.
            Default value is True.""",
        )
        MODEL_CONFIG_PATH: str = Field(
            default=DEFAULT_MODEL_CONFIG_PATH,
            description=f"""URL to the YAML file containing model definitions.
            Must be a publicly accessible URL (http:// or https://).
            Default value is '{DEFAULT_MODEL_CONFIG_PATH}'.""",
        )
        URL_RESOLVE_TIMEOUT: int = Field(
            default=10,
            description="Timeout in seconds for resolving a single source URL. Default is 10.",
        )
        URL_RESOLVE_MAX_RETRIES: int = Field(
            default=3,
            description="Maximum number of attempts to resolve a URL before giving up. Default is 3.",
        )
        URL_RESOLVE_BASE_DELAY: float = Field(
            default=0.5,
            description="Initial delay in seconds between retries, using exponential backoff. Default is 0.5.",
        )
        STATUS_EMISSION_BEHAVIOR: Literal[
            "disable",
            "hidden_compact",
            "hidden_detailed",
            "visible",
            "visible_timed",
        ] = Field(
            default="hidden_detailed",
            description="""Control status display. (Default: hidden_detailed) • Options • disable: No status.
            • hidden_compact: Final success hidden, no thoughts. • hidden_detailed: Final success hidden, with thoughts.
            • visible: All status visible. • visible_timed: Visible with timestamps.""",
        )
        LOG_LEVEL: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = (
            Field(
                default="INFO",
                description="Select logging level. Use `docker logs -f open-webui` to view logs.",
            )
        )

    # TODO: Support user settting through UserValves.

    def __init__(self):
        # Initialize valves with defaults; the framework injects DB values before each request.
        self.valves = self.Valves()
        self.log_level = self.valves.LOG_LEVEL
        self._add_log_handler()
        log.success("Function has been initialized.")
        log.trace("Companion initialized; valve values omitted from logs.")

    def inlet(
        self,
        body: "Body",
        __request__: Request,
        __metadata__: "Metadata",
        __event_emitter__: Callable[["Event"], Awaitable[None]],
    ) -> "Body":
        """Modifies the incoming request payload before it's sent to the LLM. Operates on the `form_data` dictionary."""

        app_state: State = __request__.app.state

        # Load and store model configuration in app state
        log.debug("Loading model configuration...")
        catalog_envelope = self._load_model_config(self.valves.MODEL_CONFIG_PATH)
        if date.today() > catalog_envelope.payload.freshness.expires_after:
            raise ModelCatalogError("Gemini model catalog evidence is expired")
        app_state._state["gemini_model_catalog"] = catalog_envelope.model_dump(
            mode="json", exclude_none=False
        )
        log.debug(
            "Stored the atomic model-catalog envelope in app state with "
            f"{len(catalog_envelope.payload.provider_claims)} model(s)."
        )

        # Detect log level change inside self.valves
        if self.log_level != self.valves.LOG_LEVEL:
            log.info(
                f"Detected log level change: {self.log_level=} and {self.valves.LOG_LEVEL=}. "
                "Running the logging setup again."
            )
            self._add_log_handler()

        log.debug(f"inlet method has been called. Gemini Manifold Companion version is {VERSION}")

        _, is_manifold = self._get_model_name(body)

        # Exit early if we are filtering an unsupported model.
        if not is_manifold:
            log.debug(
                "Returning the original body object because conditions for proceeding are not fulfilled."
            )
            return body

        features = body.get("features", {})
        log.debug(f"Received {len(features)} request feature flag(s).")

        # Ensure features field exists
        metadata = body.get("metadata")
        metadata_features = metadata.get("features")
        if metadata_features is None:
            metadata_features = cast("Features", {})
            metadata["features"] = metadata_features

        metadata["chat_control_params"] = self._extract_chat_control_params(body)

        # Add the companion version to the payload for the pipe to consume.
        metadata_features["gemini_manifold_companion_version"] = VERSION

        web_search_enabled = (
            features.get("web_search", False) if isinstance(features, dict) else False
        )
        if web_search_enabled:
            log.info(
                "Search feature was requested; the pipe will authorize it against the selected service policy."
            )
            features["web_search"] = False
            metadata_features["google_search_tool"] = True
        code_execution_enabled = (
            features.get("code_interpreter", False) if isinstance(features, dict) else False
        )
        if code_execution_enabled:
            log.info(
                "Code execution was requested; the pipe will authorize it against the selected service policy."
            )
            features["code_interpreter"] = False
            metadata_features["google_code_execution"] = True
        if self.valves.BYPASS_BACKEND_RAG:
            if __metadata__["chat_id"] == "local":
                # TODO toast notification
                log.warning(
                    "Bypassing Open WebUI's RAG is not possible for temporary chats. "
                    "The Manifold pipe requires a database entry to access uploaded files, "
                    "which temporary chats do not have. Falling back to Open WebUI's RAG."
                )
                metadata_features["upload_documents"] = False
            else:
                log.info(
                    "BYPASS_BACKEND_RAG is enabled, bypassing Open WebUI RAG to let the Manifold pipe handle documents."
                )
                if files := body.get("files"):
                    log.info(f"Removing {len(files)} files from the Open WebUI RAG pipeline.")
                    body["files"] = []
                metadata_features["upload_documents"] = True
        else:
            log.info("BYPASS_BACKEND_RAG is disabled. Open WebUI's RAG will be used if applicable.")
            metadata_features["upload_documents"] = False

        # TODO: Filter out the citation markers here.

        log.debug("inlet method has finished.")
        return body

    def stream(self, event: dict) -> dict:
        """Modifies the streaming response from the LLM in real-time. Operates on individual chunks of data."""
        return event

    async def outlet(
        self,
        body: "Body",
        __request__: Request,
        __metadata__: dict[str, Any],
        __event_emitter__: Callable[["Event"], Awaitable[None]],
    ) -> "Body":
        """Apply the durable, SDK-neutral grounding envelope exactly once."""
        del __request__
        emitter = EventEmitter(__event_emitter__, status_mode=self.valves.STATUS_EMISSION_BEHAVIOR)
        try:
            envelope = await self._load_grounding_envelope(body, __metadata__)
            if envelope is None:
                return body
            text, setter = self._assistant_text_accessor(body)
            if text is None or setter is None:
                emitter.emit_status(
                    "Grounding metadata could not be applied to this response.", done=True
                )
                return body
            current_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if current_digest != envelope.visible_content_sha256:
                emitter.emit_status(
                    "Grounding citations were skipped because the assistant response was edited.",
                    done=True,
                )
                return body
            cited_text, warning_count = self._insert_citation_markers(envelope, text)
            setter(cited_text)
            if envelope.queries:
                emitter.emit_grounding_queries(envelope.queries)
            await self._emit_grounding_sources(envelope, emitter)
            for error in envelope.tool_errors:
                emitter.emit_toast(error, "warning")
            if warning_count or envelope.diagnostics:
                emitter.emit_toast(
                    "Some grounding annotations could not be displayed safely.", "warning"
                )
            if envelope.sources:
                emitter.emit_status("This response was grounded with a Google tool", done=True)
            return body
        except ValidationError:
            log.error("Invalid Gemini grounding envelope.")
            emitter.emit_toast("Stored grounding metadata is invalid and was ignored.", "warning")
            return body
        finally:
            await emitter.flush()
            await emitter.shutdown()

    # region 1. Helper methods inside the Filter class

    # region 1.1 Add citations

    @staticmethod
    async def _load_grounding_envelope(
        body: "Body", metadata: dict[str, Any]
    ) -> GroundingEnvelope | None:
        messages = body.get("messages") or []
        if messages and isinstance(messages[-1], dict):
            interaction = messages[-1].get("gemini_interaction")
            if isinstance(interaction, dict) and isinstance(interaction.get("grounding"), dict):
                return GroundingEnvelope.model_validate(interaction["grounding"])
        chat_id = metadata.get("chat_id")
        message_id = metadata.get("message_id")
        user_id = metadata.get("user_id")
        if not all(isinstance(value, str) and value for value in (chat_id, message_id, user_id)):
            return None
        chat = await Chats.get_chat_by_id_and_user_id(id=chat_id, user_id=user_id)
        chat_data = getattr(chat, "chat", None)
        if not isinstance(chat_data, dict):
            return None
        history = chat_data.get("history")
        db_messages = history.get("messages") if isinstance(history, dict) else None
        message = db_messages.get(message_id) if isinstance(db_messages, dict) else None
        interaction = message.get("gemini_interaction") if isinstance(message, dict) else None
        grounding = interaction.get("grounding") if isinstance(interaction, dict) else None
        return GroundingEnvelope.model_validate(grounding) if isinstance(grounding, dict) else None

    @staticmethod
    def _assistant_text_accessor(
        body: "Body",
    ) -> tuple[str | None, Callable[[str], None] | None]:
        messages = body.get("messages") or []
        if not messages or not isinstance(messages[-1], dict):
            return None, None
        message = messages[-1]
        content = message.get("content")
        if isinstance(content, str):
            message_mapping = cast(dict[str, object], message)

            def set_message_content(value: str) -> None:
                message_mapping["content"] = value

            return content, set_message_content
        if isinstance(content, list):
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                ):
                    item_mapping = cast(dict[str, object], item)
                    text = cast(str, item_mapping["text"])

                    def set_item_content(
                        value: str, target: dict[str, object] = item_mapping
                    ) -> None:
                        target["text"] = value

                    return text, set_item_content
        return None, None

    @staticmethod
    def _insert_citation_markers(envelope: GroundingEnvelope, visible_text: str) -> tuple[str, int]:
        source_numbers = {source.id: index + 1 for index, source in enumerate(envelope.sources)}
        citations_by_block: dict[int, list[GroundingCitation]] = {}
        for citation in envelope.citations:
            citations_by_block.setdefault(citation.block_index, []).append(citation)
        output = visible_text
        search_from = 0
        replacements: list[tuple[int, int, str]] = []
        warnings = 0
        for block_index, block in enumerate(envelope.text_blocks):
            start_at = output.find(block.text, search_from)
            if start_at < 0:
                warnings += len(citations_by_block.get(block_index, []))
                continue
            search_from = start_at + len(block.text)
            grouped: dict[int, set[int]] = {}
            for citation in citations_by_block.get(block_index, []):
                number = source_numbers.get(citation.source_id)
                position = Filter._citation_end_character(citation, block.text)
                if number is None or position is None:
                    warnings += 1
                    continue
                grouped.setdefault(position, set()).add(number)
            for position, numbers in grouped.items():
                marker = "".join(f"[{number}]" for number in sorted(numbers))
                replacements.append((start_at + position, start_at + position, marker))
        for start, end, marker in sorted(replacements, reverse=True):
            output = output[:start] + marker + output[end:]
        return output, warnings

    @staticmethod
    def _citation_end_character(citation: GroundingCitation, text: str) -> int | None:
        if citation.start < 0 or citation.end < citation.start:
            return None
        if citation.index_unit == "unicode_codepoints":
            return citation.end if citation.end <= len(text) else None
        if citation.index_unit == "provider" and not text.isascii():
            return None
        encoded = text.encode("utf-8")
        if citation.end > len(encoded):
            return None
        try:
            encoded[: citation.start].decode("utf-8")
            return len(encoded[: citation.end].decode("utf-8"))
        except UnicodeDecodeError:
            return None

    async def _resolve_url(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> tuple[str, bool]:
        """
        Resolves a given URL using values from Valves.
        Returns the final URL and a boolean indicating success.
        """
        if not url:
            return "", False

        timeout = aiohttp.ClientTimeout(total=self.valves.URL_RESOLVE_TIMEOUT)
        max_retries = self.valves.URL_RESOLVE_MAX_RETRIES
        base_delay = self.valves.URL_RESOLVE_BASE_DELAY

        for attempt in range(max_retries + 1):
            try:
                async with session.get(
                    url,
                    allow_redirects=True,
                    timeout=timeout,
                ) as response:
                    final_url = str(response.url)
                    log.debug(f"Resolved a grounding URL after {attempt} retries.")
                    return final_url, True
            except (TimeoutError, aiohttp.ClientError):
                if attempt == max_retries:
                    log.error(
                        f"Failed to resolve a grounding URL after {max_retries + 1} attempts."
                    )
                    return url, False
                else:
                    delay = min(base_delay * (2**attempt), 10.0)
                    log.warning(
                        f"Grounding URL retry {attempt + 1}/{max_retries + 1}; waiting {delay:.1f}s."
                    )
                    await asyncio.sleep(delay)
            except Exception:
                log.error("Unexpected grounding URL resolution failure.")
                return url, False
        return url, False

    async def _emit_grounding_sources(
        self, envelope: GroundingEnvelope, emitter: EventEmitter
    ) -> None:
        grouped: dict[str, list[GroundingSource]] = {}
        for source in envelope.sources:
            name = (
                "google_maps"
                if source.kind == "place"
                else "file_search"
                if source.kind == "file"
                else "google_search"
            )
            grouped.setdefault(name, []).append(source)
        for name, sources in grouped.items():
            documents: list[str] = []
            metadata: list[SourceMetadata] = []
            for source in sources:
                original_uri = source.uri
                resolved_uri = original_uri
                if original_uri and original_uri.startswith(
                    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"
                ):
                    async with aiohttp.ClientSession() as session:
                        resolved_uri, _ = await self._resolve_url(session, original_uri)
                details = [
                    value for value in (source.title, source.file_name, source.place_id) if value
                ]
                documents.append("\n".join(details))
                metadata.append(
                    {
                        "source": resolved_uri,
                        "original_url": original_uri,
                        "supports": [],
                    }
                )
            emitter.emit_sources(
                {"source": {"name": name}, "document": documents, "metadata": metadata}
            )

    # endregion 1.1 Add citations

    # region 1.2 Remove citation markers
    # TODO: Remove citation markers from model input.
    # endregion 1.2 Remove citation markers

    # region 1.3 Configuration loading

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _load_model_config(config_path: str) -> CatalogAppStateEnvelope:
        """Loads the model configuration from a URL.

        Uses LRU cache to avoid reloading the same configuration repeatedly.
        Cache is tied to the config_path argument.
        """
        if not config_path:
            raise ModelCatalogError("MODEL_CONFIG_PATH must not be empty")

        try:
            if not (config_path.startswith("http://") or config_path.startswith("https://")):
                raise ModelCatalogError("MODEL_CONFIG_PATH must be an HTTP(S) URL.")

            log.debug("Loading the configured Gemini model catalog.")
            with urllib.request.urlopen(config_path, timeout=10) as response:
                raw_config = yaml.load(response.read(), Loader=_UniqueKeyLoader)
            catalog = ModelCatalog.model_validate(raw_config)
            envelope = CatalogAppStateEnvelope.from_catalog(catalog)
            log.success(
                f"Loaded Gemini model catalog protocol {catalog.schema_version} "
                f"with {len(catalog.provider_claims)} model(s)."
            )
            return envelope
        except ModelCatalogError:
            raise
        except Exception:
            raise ModelCatalogError("Gemini model catalog is unavailable or invalid.") from None

    # endregion 1.3 Configuration loading

    # region 1.5 Model capability checks

    @staticmethod
    def _check_model_capability(
        model_id: str,
        config: dict,
        capability: str,
        *,
        service: Literal["developer", "enterprise"] = "developer",
    ) -> bool:
        """Check if a model supports a specific capability based on YAML config.

        Args:
            model_id: The canonical model id (without prefixes)
            config: The loaded YAML configuration dict
            capability: The capability to check (e.g., "search_grounding", "code_execution")

        Returns:
            True if the model supports the capability, False otherwise
        """
        if model_id not in config:
            log.debug(
                f"Model '{model_id}' not found in config, capability '{capability}' check returns False."
            )
            return False

        model_config = config[model_id]
        service_policy = model_config.get("services", {}).get(service, {})
        if service_policy.get("availability") != "supported":
            return False
        capability_paths = {
            "search_grounding": ("tools", "google_search"),
            "code_execution": ("tools", "code_execution"),
            "url_context": ("tools", "url_context"),
            "grounding_google_maps": ("tools", "google_maps"),
            "file_search": ("tools", "file_search"),
            "function_calling": ("custom_function_calling",),
            "thinking": ("thinking", "supported"),
            "structured_outputs": ("response_format",),
        }
        path = capability_paths.get(capability)
        if path is None:
            log.warning("An unknown catalog capability was denied.")
            return False
        value: object = service_policy.get("interactions", {})
        for key in path:
            if not isinstance(value, dict):
                return False
            value = value.get(key, False)
        result = value is True

        log.debug("Completed a catalog capability check.")
        return result

    # endregion 1.5 Model capability checks

    # region 1.6 Utility helpers

    def _extract_chat_control_params(self, body: "Body") -> dict[str, Any]:
        """
        Extracts custom parameters set at the chat level.
        By storing these in metadata, we protect them from being overwritten
        by model-level defaults during OWUI's pre-pipe merge phase. The pipe
        can then prioritize these chat-specific settings over model-wide defaults.
        """
        chat_control_params: dict[str, Any] = {}
        # Standard OWUI body keys. Any others are treated as custom chat parameters.
        known_body_keys = {
            "stream",
            "model",
            "messages",
            "files",
            "features",
            "metadata",
            "options",
            "stream_options",
        }

        custom_param_keys = [key for key in body if key not in known_body_keys]
        for key in custom_param_keys:
            chat_control_params[key] = body[key]

        if custom_param_keys:
            log.debug("Found and preserved custom chat control parameters.")

        return chat_control_params

    @staticmethod
    def _get_model_name(body: "Body") -> tuple[str, bool]:
        """
        Extracts the effective and canonical model name from the request body.

        Handles standard model names and custom workspace models by prioritizing
        the base_model_id found in metadata.

        Args:
            body: The request body dictionary.

        Returns:
            A tuple containing:
            - The canonical model name (prefix removed).
            - A boolean indicating if the effective model name contained the
              'gemini_manifold_google_genai.' prefix.
        """
        # 1. Get the initially requested model name from the top level
        effective_model_name: str = body.get("model", "")
        initial_model_name = effective_model_name
        base_model_name = None

        # 2. Check for a base model ID in the metadata for custom models
        # If metadata exists, attempt to extract the base_model_id
        if metadata := body.get("metadata"):
            # Safely navigate the nested structure: metadata -> model -> info -> base_model_id
            base_model_name = metadata.get("model", {}).get("info", {}).get("base_model_id", None)
            # If a base model ID is found, it overrides the initially requested name
            if base_model_name:
                effective_model_name = base_model_name

        # 3. Determine if the effective model name contains the manifold prefix.
        # This flag indicates if the model (after considering base_model_id)
        # appears to be one defined or routed via the manifold pipe function.
        is_manifold_model = "gemini_manifold_google_genai." in effective_model_name

        # 4. Create the canonical model name by removing the manifold prefix
        # from the effective model name.
        canonical_model_name = effective_model_name.replace("gemini_manifold_google_genai.", "")

        # 5. Log the relevant names for debugging purposes
        log.debug(
            f"Model Name Extraction: initial='{initial_model_name}', "
            f"base='{base_model_name}', effective='{effective_model_name}', "
            f"canonical='{canonical_model_name}', is_manifold={is_manifold_model}"
        )

        # 6. Return the canonical name and the manifold flag
        return canonical_model_name, is_manifold_model

    def _is_flat_dict(self, data: Any) -> bool:
        """
        Checks if a dictionary contains only non-dict/non-list values (is one level deep).
        """
        if not isinstance(data, dict):
            return False
        return not any(isinstance(value, (dict, list)) for value in data.values())

    @classmethod
    def _redact_log_data(cls, data: object) -> object:
        """Remove credentials, signed state, and secret-bearing locations recursively."""
        if isinstance(data, dict):
            redacted: dict[object, object] = {}
            for key, value in data.items():
                normalized = "_".join(
                    part for part in str(key).lower().replace("-", "_").split("_") if part
                )
                sensitive = bool(
                    normalized == "gemini_interaction"
                    or "signature" in normalized
                    or normalized
                    in {
                        "api_key",
                        "authorization",
                        "credential",
                        "credential_fingerprint",
                        "password",
                        "secret",
                        "access_token",
                        "refresh_token",
                        "uri",
                        "url",
                    }
                    or normalized.endswith("_api_key")
                    or normalized.endswith("_credential")
                    or normalized.endswith("_secret")
                )
                redacted[key] = "[REDACTED]" if sensitive else cls._redact_log_data(value)
            return redacted
        if isinstance(data, list):
            return [cls._redact_log_data(item) for item in data]
        if isinstance(data, tuple):
            return tuple(cls._redact_log_data(item) for item in data)
        return data

    def _truncate_long_strings(
        self, data: Any, max_len: int, truncation_marker: str, truncation_enabled: bool
    ) -> Any:
        """
        Recursively traverses a data structure (dicts, lists) and truncates
        long string values. Creates copies to avoid modifying original data.

        Args:
            data: The data structure (dict, list, str, int, float, bool, None) to process.
            max_len: The maximum allowed length for string values.
            truncation_marker: The string to append to truncated values.
            truncation_enabled: Whether truncation is enabled.

        Returns:
            A potentially new data structure with long strings truncated.
        """
        if not truncation_enabled or max_len <= len(truncation_marker):
            # If truncation is disabled or max_len is too small, return original
            # Make a copy only if it's a mutable type we might otherwise modify
            if isinstance(data, (dict, list)):
                return copy.deepcopy(data)  # Ensure deep copy for nested structures
            return data  # Primitives are immutable

        if isinstance(data, str):
            if len(data) > max_len:
                return data[: max_len - len(truncation_marker)] + truncation_marker
            return data  # Return original string if not truncated
        elif isinstance(data, dict):
            # Process dictionary items, creating a new dict
            return {
                k: self._truncate_long_strings(v, max_len, truncation_marker, truncation_enabled)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            # Process list items, creating a new list
            return [
                self._truncate_long_strings(item, max_len, truncation_marker, truncation_enabled)
                for item in data
            ]
        else:
            # Return non-string, non-container types as is (they are immutable)
            return data

    def plugin_stdout_format(self, record: "Record") -> str:
        """
        Custom format function for the plugin's logs.
        Serializes and truncates data passed under the 'payload' key in extra.
        """

        # Configuration Keys
        LOG_OPTIONS_PREFIX = "_log_"
        TRUNCATION_ENABLED_KEY = f"{LOG_OPTIONS_PREFIX}truncation_enabled"
        MAX_LENGTH_KEY = f"{LOG_OPTIONS_PREFIX}max_length"
        TRUNCATION_MARKER_KEY = f"{LOG_OPTIONS_PREFIX}truncation_marker"
        DATA_KEY = "payload"

        original_extra = record["extra"]
        # Extract the data intended for serialization using the chosen key
        data_to_process = original_extra.get(DATA_KEY)

        serialized_data_json = ""
        if data_to_process is not None:
            try:
                serializable_data = pydantic_core.to_jsonable_python(
                    data_to_process, serialize_unknown=True
                )
                serializable_data = self._redact_log_data(serializable_data)

                # Determine truncation settings
                truncation_enabled = original_extra.get(TRUNCATION_ENABLED_KEY, True)
                max_length = original_extra.get(MAX_LENGTH_KEY, 256)
                truncation_marker = original_extra.get(TRUNCATION_MARKER_KEY, "[...]")

                # If max_length was explicitly provided, force truncation enabled
                if MAX_LENGTH_KEY in original_extra:
                    truncation_enabled = True

                # Truncate long strings
                truncated_data = self._truncate_long_strings(
                    serializable_data,
                    max_length,
                    truncation_marker,
                    truncation_enabled,
                )

                # Serialize the (potentially truncated) data
                if self._is_flat_dict(truncated_data) and not isinstance(truncated_data, list):
                    json_string = json.dumps(truncated_data, separators=(",", ":"), default=str)
                    # Add a simple prefix if it's compact
                    serialized_data_json = " - " + json_string
                else:
                    json_string = json.dumps(truncated_data, indent=2, default=str)
                    # Prepend with newline for readability
                    serialized_data_json = "\n" + json_string

            except (TypeError, ValueError):
                serialized_data_json = " - {Serialization Error}"
            except Exception:
                serialized_data_json = " - {Processing Error}"

        # Add the final JSON string (or error message) back into the record
        record["extra"]["_plugin_serialized_data"] = serialized_data_json

        # Base template
        base_template = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        # Append the serialized data
        base_template += "{extra[_plugin_serialized_data]}"
        # Append the exception part
        base_template += "\n{exception}"
        # Return the format string template
        return base_template.rstrip()

    def _add_log_handler(self):
        """
        Adds or updates the loguru handler specifically for this plugin.
        Includes logic for serializing and truncating extra data.
        """

        def plugin_filter(record: "Record"):
            """Filter function to only allow logs from this plugin (based on module name)."""
            return record["name"] == __name__

        # Get the desired level name and number
        desired_level_name = self.valves.LOG_LEVEL
        try:
            # Use the public API to get level details
            desired_level_info = log.level(desired_level_name)
            desired_level_no = desired_level_info.no
        except ValueError:
            log.error(
                f"Invalid LOG_LEVEL '{desired_level_name}' configured for plugin {__name__}. Cannot add/update handler."
            )
            return  # Stop processing if the level is invalid

        # Access the internal state of the log
        handlers: dict[int, Handler] = log._core.handlers  # type: ignore
        handler_id_to_remove = None
        found_correct_handler = False

        for handler_id, handler in handlers.items():
            existing_filter = handler._filter  # Access internal attribute

            # Check if the filter matches our plugin_filter
            # Comparing function objects directly can be fragile if they are recreated.
            # Comparing by name and module is more robust for functions defined at module level.
            is_our_filter = (
                existing_filter is not None  # Make sure a filter is set
                and hasattr(existing_filter, "__name__")
                and existing_filter.__name__ == plugin_filter.__name__
                and hasattr(existing_filter, "__module__")
                and existing_filter.__module__ == plugin_filter.__module__
            )

            if is_our_filter:
                existing_level_no = handler.levelno
                log.trace(
                    f"Found existing handler {handler_id} for {__name__} with level number {existing_level_no}."
                )

                # Check if the level matches the desired level
                if existing_level_no == desired_level_no:
                    log.debug(
                        f"Handler {handler_id} for {__name__} already exists with the correct level '{desired_level_name}'."
                    )
                    found_correct_handler = True
                    break  # Found the correct handler, no action needed
                else:
                    # Found our handler, but the level is wrong. Mark for removal.
                    log.info(
                        f"Handler {handler_id} for {__name__} found, but log level differs "
                        f"(existing: {existing_level_no}, desired: {desired_level_no}). "
                        f"Removing it to update."
                    )
                    handler_id_to_remove = handler_id
                    break  # Found the handler to replace, stop searching

        # Remove the old handler if marked for removal
        if handler_id_to_remove is not None:
            try:
                log.remove(handler_id_to_remove)
                log.debug(f"Removed handler {handler_id_to_remove} for {__name__}.")
            except ValueError:
                # This might happen if the handler was somehow removed between the check and now
                log.warning(
                    f"Could not remove handler {handler_id_to_remove} for {__name__}. It might have already been removed."
                )
                # If removal failed but we intended to remove, we should still proceed to add
                # unless found_correct_handler is somehow True (which it shouldn't be if handler_id_to_remove was set).

        # Add a new handler if no correct one was found OR if we just removed an incorrect one
        if not found_correct_handler:
            self.log_level = desired_level_name
            log.add(
                sys.stdout,
                level=desired_level_name,
                format=self.plugin_stdout_format,
                filter=plugin_filter,
            )
            log.debug(
                f"Added new handler to loguru for {__name__} with level {desired_level_name}."
            )

    # endregion 1.4 Utility helpers

    # endregion 1. Helper methods inside the Filter class
