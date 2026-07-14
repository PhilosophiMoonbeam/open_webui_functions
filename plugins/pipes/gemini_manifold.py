"""
title: Gemini Manifold google_genai
id: gemini_manifold_google_genai
description: Manifold function for Gemini Developer API and Gemini Enterprise. Uses the newer google-genai SDK. Aims to support as many features from it as possible.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 3.0.0
requirements: google-genai==2.11.0, pikepdf
"""

# I change these only when I make a release to avoid PR merge conflicts.
# If you are making a PR then please do not change these values.
VERSION = "3.0.0"
# This is the recommended version for the companion filter.
# Older versions might still work, but backward compatibility is not guaranteed
# during the development of this personal use plugin.
RECOMMENDED_COMPANION_VERSION = "3.0.0"
MODEL_CATALOG_SCHEMA_VERSION = 1
GROUNDING_ENVELOPE_PROTOCOL_VERSION = 1
CATALOG_MODEL_IDS = frozenset(
    {
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-flash-image-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
        "gemini-3-pro-image-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-image",
        "gemini-2.5-flash-lite",
    }
)


# Keys `title`, `id` and `description` in the frontmatter above are used for my own development purposes.
# They don't have any effect on the plugin's functionality.


# This helper provides a manifold for the Gemini Developer API and Gemini Enterprise.
# Be sure to check out my GitHub repository for more information! Contributions, questions and suggestions are very welcome.

import asyncio
import base64
import copy
import difflib
import fnmatch
import hashlib
import io
import json
import mimetypes
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    Unpack,
    cast,
    overload,
)
from urllib.parse import parse_qs, urlparse

import aiofiles
import httpx
import pydantic_core
import xxhash
from aiocache import cached
from aiocache.backends.memory import SimpleMemoryCache
from aiocache.base import BaseCache
from aiocache.serializers import NullSerializer
from fastapi import FastAPI, Request
from fastapi.datastructures import State
from google import genai
from google.genai import errors as genai_errors
from google.genai import interactions as interaction_types
from google.genai import types
from loguru import logger
from open_webui.models.chats import Chats
from open_webui.models.files import FileForm, Files
from open_webui.models.functions import Functions
from open_webui.storage.provider import Storage
from open_webui.utils.misc import pop_system_message
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, field_validator

# This block is skipped at runtime.
if TYPE_CHECKING:
    from loguru import Record
    from loguru._handler import Handler  # type: ignore

    # Imports custom type definitions (TypedDicts) for static analysis purposes (mypy/pylance).
    from utils.manifold_types import *

# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main log.
log = logger.bind(auditable=False)


class EventEmitter(Protocol):
    start_time: float

    def emit_toast(
        self, msg: str, type: Literal["info", "success", "warning", "error"] = "info"
    ) -> None: ...

    def emit_status(
        self,
        description: str,
        done: bool = False,
        hidden: bool = False,
        *,
        is_successful_finish: bool = False,
        is_thought: bool = False,
        indent_level: int = 0,
    ) -> None: ...


class PipeEventEmitter:
    """Request-local ordered emitter; never crosses process or request state."""

    def __init__(self, emitter: Callable[["Event"], Awaitable[None]] | None) -> None:
        self._emitter = emitter
        self.start_time = time.monotonic()
        self._queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        self._worker = asyncio.create_task(self._run()) if emitter is not None else None

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                if event is None:
                    return
                if self._emitter is not None:
                    try:
                        await self._emitter(cast("Event", event))
                    except asyncio.CancelledError:
                        log.warning("Open WebUI event callback was cancelled; dropping the event.")
                    except Exception:
                        log.exception("Open WebUI event callback failed; dropping the event.")
            finally:
                self._queue.task_done()

    def _enqueue(self, event: dict[str, object]) -> None:
        if self._worker is not None:
            self._queue.put_nowait(event)

    def emit_toast(
        self, msg: str, type: Literal["info", "success", "warning", "error"] = "info"
    ) -> None:
        self._enqueue({"type": "notification", "data": {"type": type, "content": msg}})

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
        del is_successful_finish, is_thought
        if indent_level:
            description = f"{'- ' * indent_level}{description}"
        self._enqueue(
            {
                "type": "status",
                "data": {"description": description, "done": done, "hidden": hidden},
            }
        )

    async def flush(self) -> None:
        if self._worker is not None and self._worker.done():
            while not self._queue.empty():
                self._queue.get_nowait()
                self._queue.task_done()
        await self._queue.join()

    async def shutdown(self) -> None:
        if self._worker is None:
            return
        if self._worker.done():
            while not self._queue.empty():
                self._queue.get_nowait()
                self._queue.task_done()
            return
        self._queue.put_nowait(None)
        await self._worker


# These tags will be "disabled" in the response, meaning that they will not be parsed by the backend.
SPECIAL_TAGS_TO_DISABLE = [
    "details",
    "think",
    "thinking",
    "reason",
    "reasoning",
    "thought",
    "Thought",
    "|begin_of_thought|",
    "code_interpreter",
    "|begin_of_solution|",
]
ZWS = "\u200b"
GEMINI_PDF_MAX_BYTES: Final = 50 * 1024 * 1024
GEMINI_PDF_SAFE_TARGET_BYTES: Final = 48 * 1024 * 1024
GEMINI_PDF_MAX_PAGES: Final = 1000
GEMINI_PDF_MITIGATION_CACHE_TTL_SECONDS: Final = 6 * 60 * 60
GEMINI_PDF_MITIGATION_CACHE_DIR_NAME: Final = "open_webui_gemini_pdf_mitigation"
GEMINI_PDF_PROCESSING_CONCURRENCY: Final = 1
GEMINI_INTERACTION_MAX_REQUEST_BYTES: Final = 100 * 1024 * 1024
GEMINI_TOOL_MAX_ROUNDS: Final = 8
GEMINI_TOOL_MAX_CALLS_PER_ROUND: Final = 16
GEMINI_TOOL_MAX_TOTAL_CALLS: Final = 32
GEMINI_TOOL_TIMEOUT_SECONDS: Final = 30.0
GEMINI_TOOL_MAX_RESULT_BYTES: Final = 1024 * 1024
GEMINI_TOOL_NAME_PATTERN: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
MESSAGE_COMPLETION_TTL_SECONDS: Final = 15 * 60.0
MESSAGE_COMPLETION_MAX_ENTRIES: Final = 4096
LOG_REDACTED: Final = "[REDACTED]"


@dataclass(frozen=True)
class PreparedPDFPart:
    path: str
    size: int
    start_page: int
    end_page: int


@dataclass(frozen=True)
class PreparedPDFResult:
    parts: list[PreparedPDFPart]
    page_count: int
    was_mitigated: bool


@dataclass(frozen=True)
class PDFMitigationOutcome:
    original_hash: str
    result: PreparedPDFResult


@dataclass(frozen=True)
class LocalFileSource:
    file_bytes: bytes | None
    file_path: str | None
    mime_type: str
    is_temp: bool = False


@dataclass
class MessageGate:
    """Reference-counted lock that remains reachable while holders or waiters exist."""

    lock: asyncio.Lock
    users: int = 0


InteractionEvent = interaction_types.InteractionSSEEvent
InteractionStep = interaction_types.Step
InteractionContent = interaction_types.Content

InteractionEventT = TypeVar("InteractionEventT", covariant=True)


class AsyncInteractionStream(Protocol, Generic[InteractionEventT]):
    """Public structural view of the SDK stream, whose concrete class is private."""

    def __aiter__(self) -> AsyncIterator[InteractionEventT]: ...

    async def __anext__(self) -> InteractionEventT: ...

    async def close(self) -> None: ...

    async def __aenter__(self) -> "AsyncInteractionStream[InteractionEventT]": ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None: ...


class AsyncInteractionsBoundary(Protocol):
    """Typed facade over the SDK's runtime-erased ``create`` signature."""

    @overload
    async def create(
        self, **request: Unpack[interaction_types.CreateModelInteractionParamsNonStreaming]
    ) -> interaction_types.Interaction: ...

    @overload
    async def create(
        self, **request: Unpack[interaction_types.CreateModelInteractionParamsStreaming]
    ) -> AsyncInteractionStream[InteractionEvent]: ...

    async def get(
        self,
        id: str,
        *,
        stream: Literal[True],
        last_event_id: str,
    ) -> AsyncInteractionStream[InteractionEvent]: ...

    async def cancel(self, id: str) -> interaction_types.Interaction: ...


class AsyncOpenWebUIToolCallable(Protocol):
    """Callable already authorized and context-bound by Open WebUI."""

    def __call__(self, **kwargs: object) -> Awaitable[object]: ...


@dataclass(frozen=True)
class OpenWebUITool:
    """Validated executable boundary paired with its Interactions declaration."""

    name: str
    parameters: dict[str, object]
    function: AsyncOpenWebUIToolCallable
    declaration: interaction_types.Function


@dataclass(frozen=True)
class ToolCallRecord:
    """Per-request idempotency record for a completed custom function call."""

    fingerprint: str
    result: interaction_types.FunctionResultStep


class NormalizedInteractionUsage(BaseModel):
    """Stable project-owned usage shape consumed outside the SDK boundary."""

    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    cached_tokens: int = 0
    tool_use_tokens: int = 0
    total_tokens: int = 0
    input_by_modality: list[dict[str, JsonValue]] = Field(default_factory=list)
    output_by_modality: list[dict[str, JsonValue]] = Field(default_factory=list)
    cached_by_modality: list[dict[str, JsonValue]] = Field(default_factory=list)
    tool_use_by_modality: list[dict[str, JsonValue]] = Field(default_factory=list)
    grounding_tool_count: list[dict[str, JsonValue]] = Field(default_factory=list)


class NormalizedInteractionEvent(BaseModel):
    """Lossless event envelope; reducers must explicitly handle unknown variants."""

    event_type: str
    event_id: str | None = None
    payload: dict[str, JsonValue]
    is_unknown: bool = False


class InteractionsSDKBoundary:
    """The sole conversion point between SDK models and project-owned records."""

    @staticmethod
    def normalize_usage(
        usage: interaction_types.Usage | None,
    ) -> NormalizedInteractionUsage:
        if usage is None:
            return NormalizedInteractionUsage()
        return NormalizedInteractionUsage(
            input_tokens=usage.total_input_tokens or 0,
            output_tokens=usage.total_output_tokens or 0,
            thought_tokens=usage.total_thought_tokens or 0,
            cached_tokens=usage.total_cached_tokens or 0,
            tool_use_tokens=usage.total_tool_use_tokens or 0,
            total_tokens=usage.total_tokens or 0,
            input_by_modality=[
                cast(dict[str, JsonValue], item.model_dump(mode="json", exclude_none=True))
                for item in (usage.input_tokens_by_modality or [])
            ],
            output_by_modality=[
                cast(dict[str, JsonValue], item.model_dump(mode="json", exclude_none=True))
                for item in (usage.output_tokens_by_modality or [])
            ],
            cached_by_modality=[
                cast(dict[str, JsonValue], item.model_dump(mode="json", exclude_none=True))
                for item in (usage.cached_tokens_by_modality or [])
            ],
            tool_use_by_modality=[
                cast(dict[str, JsonValue], item.model_dump(mode="json", exclude_none=True))
                for item in (usage.tool_use_tokens_by_modality or [])
            ],
            grounding_tool_count=[
                cast(dict[str, JsonValue], item.model_dump(mode="json", exclude_none=True))
                for item in (usage.grounding_tool_count or [])
            ],
        )

    @staticmethod
    def normalize_event(event: InteractionEvent) -> NormalizedInteractionEvent:
        payload = cast(
            dict[str, JsonValue],
            event.model_dump(mode="json", exclude_none=True),
        )
        event_type = str(payload.get("event_type", "UNKNOWN"))
        return NormalizedInteractionEvent(
            event_type=event_type,
            event_id=cast(str | None, payload.get("event_id")),
            payload=payload,
            is_unknown=getattr(event, "is_unknown", False),
        )


class ReducerEmission(BaseModel):
    """A transport-neutral semantic output consumed by the OWUI presenter."""

    kind: Literal["content", "reasoning", "media", "source", "tool", "status"]
    text: str | None = None
    payload: dict[str, JsonValue] | None = None


class GroundingTextBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step_index: int
    content_index: int
    text: str


class GroundingReviewSnippet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    review_id: str | None = None
    title: str | None = None
    uri: str | None = None


class GroundingSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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


class GroundingCitation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    block_index: int
    start: int
    end: int
    index_unit: Literal["provider", "utf8_bytes", "unicode_codepoints"] = "provider"


class GroundingToolRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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


class GroundingDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    detail: str
    step_index: int | None = None


class GroundingEnvelope(BaseModel):
    """Versioned SDK-neutral grounding contract shared with the companion."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[1] = 1
    visible_content_sha256: str = ""
    grounded_text_sha256: str = ""
    text_blocks: list[GroundingTextBlock] = Field(default_factory=list)
    sources: list[GroundingSource] = Field(default_factory=list)
    citations: list[GroundingCitation] = Field(default_factory=list)
    tool_records: list[GroundingToolRecord] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    tool_errors: list[str] = Field(default_factory=list)
    diagnostics: list[GroundingDiagnostic] = Field(default_factory=list)


class InteractionReduction(BaseModel):
    """Project-owned, persistable result of reducing an Interaction timeline."""

    interaction_id: str | None = None
    status: str = "in_progress"
    steps: list[dict[str, JsonValue]] = Field(default_factory=list)
    usage: NormalizedInteractionUsage = Field(default_factory=NormalizedInteractionUsage)
    last_event_id: str | None = None
    terminal: bool = False
    original_content: str = ""
    reasoning_content: str = ""
    grounding: GroundingEnvelope = Field(default_factory=GroundingEnvelope)


class InteractionEnvelopeV1(BaseModel):
    """The only durable Gemini message contract; unknown fields fail closed."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    interaction_id: str | None = None
    endpoint_scope: str
    model_id: str
    store: bool
    status: Literal[
        "in_progress",
        "requires_action",
        "completed",
        "failed",
        "cancelled",
        "incomplete",
        "budget_exceeded",
    ]
    steps: list[dict[str, JsonValue]] = Field(default_factory=list)
    visible_content: str
    usage: NormalizedInteractionUsage = Field(default_factory=NormalizedInteractionUsage)
    last_event_id: str | None = None
    grounding: GroundingEnvelope = Field(default_factory=GroundingEnvelope)


@dataclass(frozen=True)
class ContinuationDecision:
    """Request input selected from the active Open WebUI branch."""

    input: list[interaction_types.StepParam]
    previous_interaction_id: str | None
    used_server_state: bool
    reason: str


class InteractionReducer:
    """Finite-state reducer shared by streamed SSE and unary Interaction responses."""

    _TERMINAL = {"completed", "failed", "cancelled", "incomplete", "budget_exceeded"}

    def __init__(self) -> None:
        self.state = InteractionReduction()
        self._events_seen: set[str] = set()
        self._step_payloads: dict[int, dict[str, JsonValue]] = {}
        self._step_types: dict[int, str] = {}
        self._stopped: set[int] = set()
        self._emitted_text: dict[tuple[int, str], str] = {}
        self._step_models: dict[int, InteractionStep] = {}
        self._pending_annotations: dict[int, list[interaction_types.Annotation]] = {}
        self._retrieval_records: list[GroundingToolRecord] = []

    def consume_event(self, event: InteractionEvent) -> list[ReducerEmission]:
        event_id = getattr(event, "event_id", None)
        if event_id and event_id in self._events_seen:
            return []
        if event_id:
            self._events_seen.add(event_id)
            self.state.last_event_id = event_id

        if isinstance(event, interaction_types.UnknownInteractionSSEEvent):
            raise InteractionExecutionError(
                GenerationFailureKind.INTERACTION_ERROR,
                f"Unknown Interaction event: {event.raw}",
            )
        if isinstance(event, interaction_types.InteractionCreatedEvent):
            return self._consume_interaction_header(event.interaction)
        if isinstance(event, interaction_types.InteractionStatusUpdate):
            return self._transition(str(event.status))
        if isinstance(event, interaction_types.ErrorEvent):
            detail = (
                event.error.message if event.error and event.error.message else "Interaction error"
            )
            self.state.status = "failed"
            self.state.terminal = True
            raise InteractionExecutionError(GenerationFailureKind.INTERACTION_ERROR, detail)
        if isinstance(event, interaction_types.StepStart):
            if event.index in self._step_payloads:
                existing = self._step_payloads[event.index]
                incoming = self._dump(event.step)
                if existing != incoming:
                    raise self._protocol_error(f"Conflicting step.start for index {event.index}")
                return []
            payload = self._dump(event.step)
            self._step_payloads[event.index] = payload
            self._step_types[event.index] = str(payload.get("type", "UNKNOWN"))
            self._step_models[event.index] = event.step
            return self._emit_snapshot(event.index, event.step)
        if isinstance(event, interaction_types.StepDelta):
            if event.index not in self._step_payloads or event.index in self._stopped:
                raise self._protocol_error(f"Delta for inactive step index {event.index}")
            return self._consume_delta(event.index, event.delta)
        if isinstance(event, interaction_types.StepStop):
            if event.index not in self._step_payloads or event.index in self._stopped:
                raise self._protocol_error(f"Invalid step.stop for index {event.index}")
            self._stopped.add(event.index)
            usage = event.step_usage or event.usage
            if usage is not None:
                self.state.usage = InteractionsSDKBoundary.normalize_usage(usage)
            return []
        if isinstance(event, interaction_types.InteractionCompletedEvent):
            emissions = self._consume_interaction_header(event.interaction)
            for index, step in enumerate(event.interaction.steps or []):
                emissions.extend(self._reconcile_snapshot(index, step))
            if event.interaction.usage is not None:
                self.state.usage = InteractionsSDKBoundary.normalize_usage(event.interaction.usage)
            emissions.extend(self._transition(str(event.interaction.status)))
            return emissions
        raise self._protocol_error(f"Unhandled Interaction event {type(event).__name__}")

    def consume_interaction(
        self, interaction: interaction_types.Interaction
    ) -> list[ReducerEmission]:
        self.state.interaction_id = interaction.id or None
        emissions: list[ReducerEmission] = []
        for index, step in enumerate(interaction.steps or []):
            emissions.extend(self._reconcile_snapshot(index, step))
        self.state.usage = InteractionsSDKBoundary.normalize_usage(interaction.usage)
        emissions.extend(self._transition(str(interaction.status)))
        return emissions

    def finalize_steps(self) -> None:
        self.state.steps = [self._step_payloads[index] for index in sorted(self._step_payloads)]
        self.state.grounding = self._build_grounding_envelope()

    def _consume_interaction_header(
        self, interaction: interaction_types.InteractionSseEventInteraction
    ) -> list[ReducerEmission]:
        if self.state.interaction_id and self.state.interaction_id != interaction.id:
            raise self._protocol_error("Interaction ID changed within one response")
        self.state.interaction_id = interaction.id
        if interaction.usage is not None:
            self.state.usage = InteractionsSDKBoundary.normalize_usage(interaction.usage)
        return self._transition(str(interaction.status))

    def _transition(self, status: str) -> list[ReducerEmission]:
        if self.state.terminal and status != self.state.status:
            raise self._protocol_error(f"Terminal status {self.state.status} changed to {status}")
        self.state.status = status
        self.state.terminal = status in self._TERMINAL
        if status in {"failed", "cancelled", "incomplete", "budget_exceeded"}:
            raise InteractionExecutionError(
                GenerationFailureKind.INTERACTION_STATUS,
                f"Interaction ended with status {status}",
            )
        return [ReducerEmission(kind="status", text=status)]

    def _reconcile_snapshot(self, index: int, step: InteractionStep) -> list[ReducerEmission]:
        incoming = self._dump(step)
        existing_type = self._step_types.get(index)
        incoming_type = str(incoming.get("type", "UNKNOWN"))
        if existing_type and existing_type != incoming_type:
            raise self._protocol_error(f"Step type changed at index {index}")
        self._step_payloads[index] = incoming
        self._step_types[index] = incoming_type
        self._step_models[index] = step
        self._stopped.add(index)
        return self._emit_snapshot(index, step)

    def _emit_snapshot(self, index: int, step: InteractionStep) -> list[ReducerEmission]:
        if isinstance(step, interaction_types.UnknownStep):
            raise self._protocol_error(f"Unknown Interaction step: {step.raw}")
        emissions: list[ReducerEmission] = []
        if isinstance(step, interaction_types.ModelOutputStep):
            if step.error and step.error.message:
                raise InteractionExecutionError(
                    GenerationFailureKind.MODEL_OUTPUT_STATUS, step.error.message
                )
            for content_index, content in enumerate(step.content or []):
                emissions.extend(self._emit_content(index, content_index, content, "content"))
        elif isinstance(step, interaction_types.ThoughtStep):
            for content_index, content in enumerate(step.summary or []):
                emissions.extend(self._emit_content(index, content_index, content, "reasoning"))
        elif isinstance(
            step,
            (
                interaction_types.FunctionCallStep,
                interaction_types.FunctionResultStep,
                interaction_types.MCPServerToolCallStep,
            ),
        ):
            emissions.append(ReducerEmission(kind="tool", payload=self._dump(step)))
        elif isinstance(
            step,
            (
                interaction_types.URLContextCallStep,
                interaction_types.GoogleSearchCallStep,
                interaction_types.FileSearchCallStep,
                interaction_types.GoogleMapsCallStep,
            ),
        ):
            emissions.append(self._source_emission(step, "call"))
        elif isinstance(step, interaction_types.CodeExecutionCallStep):
            code = step.arguments.code if step.arguments else None
            if code:
                emissions.append(
                    ReducerEmission(kind="content", text=f"```python\n{code.rstrip()}\n```\n\n")
                )
        elif isinstance(step, interaction_types.CodeExecutionResultStep):
            if step.result:
                emissions.append(
                    ReducerEmission(
                        kind="content", text=f"**Output:**\n\n```\n{step.result.rstrip()}\n```\n\n"
                    )
                )
        elif isinstance(
            step,
            (
                interaction_types.URLContextResultStep,
                interaction_types.GoogleSearchResultStep,
                interaction_types.FileSearchResultStep,
                interaction_types.GoogleMapsResultStep,
                interaction_types.MCPServerToolResultStep,
            ),
        ):
            emissions.append(self._source_emission(step, "result"))
        return emissions

    def _emit_content(
        self,
        step_index: int,
        content_index: int,
        content: InteractionContent | interaction_types.ThoughtSummaryContent,
        channel: Literal["content", "reasoning"],
    ) -> list[ReducerEmission]:
        if isinstance(content, interaction_types.UnknownContent):
            raise self._protocol_error(f"Unknown Interaction content: {content.raw}")
        if isinstance(content, interaction_types.TextContent):
            key = (step_index, f"{channel}:{content_index}")
            emitted = self._emitted_text.get(key, "")
            text = content.text
            suffix = text[len(emitted) :] if text.startswith(emitted) else text
            self._emitted_text[key] = text
            if not suffix:
                return []
            if channel == "content":
                self.state.original_content += suffix
            else:
                self.state.reasoning_content += suffix
            return [ReducerEmission(kind=channel, text=suffix)]
        return [ReducerEmission(kind="media", payload=self._dump(content))]

    def _consume_delta(
        self, index: int, delta: interaction_types.StepDeltaData
    ) -> list[ReducerEmission]:
        if isinstance(delta, interaction_types.UnknownStepDeltaData):
            raise self._protocol_error(f"Unknown Interaction delta: {delta.raw}")
        if isinstance(delta, interaction_types.TextDelta):
            channel: Literal["content", "reasoning"] = (
                "reasoning" if self._step_types[index] == "thought" else "content"
            )
            if channel == "content":
                self.state.original_content += delta.text
            else:
                self.state.reasoning_content += delta.text
            key = (index, f"{channel}:0")
            self._emitted_text[key] = self._emitted_text.get(key, "") + delta.text
            return [ReducerEmission(kind=channel, text=delta.text)]
        if isinstance(delta, interaction_types.ThoughtSummaryDelta) and delta.content:
            return self._emit_content(index, 0, delta.content, "reasoning")
        if isinstance(
            delta,
            (
                interaction_types.ImageDelta,
                interaction_types.AudioDelta,
                interaction_types.DocumentDelta,
                interaction_types.VideoDelta,
            ),
        ):
            # Media deltas may contain partial base64. Preserve them in the ledger
            # and wait for the final snapshot before presentation.
            self._step_payloads[index].setdefault("deltas", [])
            media_deltas = self._step_payloads[index]["deltas"]
            if isinstance(media_deltas, list):
                media_deltas.append(self._dump(delta))
            return []
        if isinstance(delta, interaction_types.TextAnnotationDelta):
            self._pending_annotations.setdefault(index, []).extend(delta.annotations or [])
            return []
        if isinstance(delta, interaction_types.RetrievalCallDelta):
            arguments = delta.arguments
            queries = list(arguments.queries or []) if arguments else []
            self._retrieval_records.append(
                GroundingToolRecord(
                    tool="retrieval",
                    phase="call",
                    step_index=index,
                    queries=queries,
                    retrieval_type=str(delta.retrieval_type) if delta.retrieval_type else None,
                )
            )
            return []
        if isinstance(delta, interaction_types.RetrievalResultDelta):
            self._retrieval_records.append(
                GroundingToolRecord(
                    tool="retrieval",
                    phase="result",
                    step_index=index,
                    is_error=delta.is_error,
                )
            )
            return []
        # Signatures, argument fragments, and server-tool call/result deltas are
        # replay-critical ledger data but do not directly render user content.
        self._step_payloads[index].setdefault("deltas", [])
        deltas = self._step_payloads[index]["deltas"]
        if isinstance(deltas, list):
            deltas.append(self._dump(delta))
        return []

    @staticmethod
    def _dump(model: BaseModel) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], model.model_dump(mode="json", exclude_none=True))

    def _source_emission(
        self, step: InteractionStep, phase: Literal["call", "result"]
    ) -> ReducerEmission:
        step_payload = self._dump(step)
        step_type = str(step_payload.get("type", "unknown"))
        kind = step_type.removesuffix("_call").removesuffix("_result")
        call_id = step_payload.get("call_id", step_payload.get("id"))
        payload: dict[str, JsonValue] = {
            "kind": kind,
            "phase": phase,
        }
        if isinstance(call_id, str):
            payload["call_id"] = call_id
        return ReducerEmission(kind="source", payload=payload)

    def _build_grounding_envelope(self) -> GroundingEnvelope:
        envelope = GroundingEnvelope()
        source_indexes: dict[str, int] = {}
        citation_keys: set[tuple[str, int, int, int]] = set()

        for step_index in sorted(self._step_models):
            step = self._step_models[step_index]
            if isinstance(step, interaction_types.ModelOutputStep):
                text_contents = [
                    (content_index, content)
                    for content_index, content in enumerate(step.content or [])
                    if isinstance(content, interaction_types.TextContent)
                ]
                for content_index, content in text_contents:
                    block_index = len(envelope.text_blocks)
                    envelope.text_blocks.append(
                        GroundingTextBlock(
                            step_index=step_index,
                            content_index=content_index,
                            text=content.text,
                        )
                    )
                    annotations = list(content.annotations or [])
                    if not annotations and content_index == text_contents[0][0]:
                        annotations = self._pending_annotations.get(step_index, [])
                    for annotation in annotations:
                        self._normalize_annotation(
                            envelope,
                            annotation,
                            block_index,
                            step_index,
                            source_indexes,
                            citation_keys,
                        )
            record = self._normalize_tool_step(step_index, step)
            if record is not None:
                envelope.tool_records.append(record)
                for query in record.queries:
                    if query not in envelope.queries:
                        envelope.queries.append(query)
                if record.is_error:
                    envelope.tool_errors.append(
                        f"{record.tool} result for {record.call_id or f'step {step_index}'} failed"
                    )
                if record.tool == "url_context":
                    for status in record.statuses:
                        if status not in {"success"}:
                            envelope.tool_errors.append(f"URL context returned status {status}")
                for place in record.places:
                    if place.id not in source_indexes:
                        source_indexes[place.id] = len(envelope.sources)
                        envelope.sources.append(place)

        envelope.tool_records.extend(self._retrieval_records)
        visible_text = "".join(block.text for block in envelope.text_blocks)
        envelope.grounded_text_sha256 = hashlib.sha256(visible_text.encode("utf-8")).hexdigest()
        return envelope

    def _normalize_annotation(
        self,
        envelope: GroundingEnvelope,
        annotation: interaction_types.Annotation,
        block_index: int,
        step_index: int,
        source_indexes: dict[str, int],
        citation_keys: set[tuple[str, int, int, int]],
    ) -> None:
        if isinstance(annotation, interaction_types.UnknownAnnotation):
            envelope.diagnostics.append(
                GroundingDiagnostic(
                    code="unknown_annotation",
                    detail="The provider returned an unsupported annotation variant.",
                    step_index=step_index,
                )
            )
            return
        if isinstance(annotation, interaction_types.URLCitation):
            source = GroundingSource(id="", kind="url", uri=annotation.url, title=annotation.title)
        elif isinstance(annotation, interaction_types.FileCitation):
            source = GroundingSource(
                id="",
                kind="file",
                uri=annotation.document_uri,
                file_name=annotation.file_name,
                media_id=annotation.media_id,
                page_number=annotation.page_number,
                source=annotation.source,
                custom_metadata=cast(dict[str, JsonValue] | None, annotation.custom_metadata),
            )
        elif isinstance(annotation, interaction_types.PlaceCitation):
            source = GroundingSource(
                id="",
                kind="place",
                uri=annotation.url,
                title=annotation.name,
                place_id=annotation.place_id,
                review_snippets=[
                    GroundingReviewSnippet(
                        review_id=snippet.review_id,
                        title=snippet.title,
                        uri=snippet.url,
                    )
                    for snippet in (annotation.review_snippets or [])
                ],
            )
        else:
            return
        stable_payload = source.model_dump_json(exclude={"id"}, exclude_none=True)
        source_id = f"{source.kind}:{hashlib.sha256(stable_payload.encode()).hexdigest()[:20]}"
        source = source.model_copy(update={"id": source_id})
        if source_id not in source_indexes:
            source_indexes[source_id] = len(envelope.sources)
            envelope.sources.append(source)
        if annotation.start_index is None or annotation.end_index is None:
            return
        citation_key = (source_id, block_index, annotation.start_index, annotation.end_index)
        if citation_key in citation_keys:
            return
        citation_keys.add(citation_key)
        envelope.citations.append(
            GroundingCitation(
                source_id=source_id,
                block_index=block_index,
                start=annotation.start_index,
                end=annotation.end_index,
                index_unit="provider",
            )
        )

    @staticmethod
    def _normalize_tool_step(step_index: int, step: InteractionStep) -> GroundingToolRecord | None:
        if isinstance(step, interaction_types.GoogleSearchCallStep):
            arguments = getattr(step, "arguments", None)
            return GroundingToolRecord(
                tool="google_search",
                phase="call",
                step_index=step_index,
                call_id=getattr(step, "id", None),
                queries=list(getattr(arguments, "queries", None) or []),
                search_type=(
                    str(search_type)
                    if (search_type := getattr(step, "search_type", None))
                    else None
                ),
            )
        if isinstance(step, interaction_types.GoogleSearchResultStep):
            results = getattr(step, "result", None) or []
            return GroundingToolRecord(
                tool="google_search",
                phase="result",
                step_index=step_index,
                call_id=getattr(step, "call_id", None),
                search_suggestions=[
                    suggestion
                    for item in results
                    if (suggestion := getattr(item, "search_suggestions", None)) is not None
                ],
                is_error=getattr(step, "is_error", None),
            )
        if isinstance(step, interaction_types.URLContextCallStep):
            arguments = getattr(step, "arguments", None)
            return GroundingToolRecord(
                tool="url_context",
                phase="call",
                step_index=step_index,
                call_id=getattr(step, "id", None),
                urls=list(getattr(arguments, "urls", None) or []),
            )
        if isinstance(step, interaction_types.URLContextResultStep):
            results = getattr(step, "result", None) or []
            return GroundingToolRecord(
                tool="url_context",
                phase="result",
                step_index=step_index,
                call_id=getattr(step, "call_id", None),
                urls=[url for item in results if (url := getattr(item, "url", None)) is not None],
                statuses=[
                    str(status)
                    for item in results
                    if (status := getattr(item, "status", None)) is not None
                ],
                is_error=getattr(step, "is_error", None),
            )
        if isinstance(step, interaction_types.GoogleMapsCallStep):
            arguments = getattr(step, "arguments", None)
            return GroundingToolRecord(
                tool="google_maps",
                phase="call",
                step_index=step_index,
                call_id=getattr(step, "id", None),
                queries=list(getattr(arguments, "queries", None) or []),
            )
        if isinstance(step, interaction_types.GoogleMapsResultStep):
            places: list[GroundingSource] = []
            tokens: list[str] = []
            for result in getattr(step, "result", None) or []:
                if token := getattr(result, "widget_context_token", None):
                    tokens.append(token)
                for place in getattr(result, "places", None) or []:
                    source = GroundingSource(
                        id="",
                        kind="place",
                        uri=place.url,
                        title=place.name,
                        place_id=place.place_id,
                        review_snippets=[
                            GroundingReviewSnippet(
                                review_id=snippet.review_id,
                                title=snippet.title,
                                uri=snippet.url,
                            )
                            for snippet in (getattr(place, "review_snippets", None) or [])
                        ],
                    )
                    stable = source.model_dump_json(exclude={"id"}, exclude_none=True)
                    places.append(
                        source.model_copy(
                            update={
                                "id": f"place:{hashlib.sha256(stable.encode()).hexdigest()[:20]}"
                            }
                        )
                    )
            return GroundingToolRecord(
                tool="google_maps",
                phase="result",
                step_index=step_index,
                call_id=getattr(step, "call_id", None),
                places=places,
                widget_context_tokens=tokens,
            )
        if isinstance(step, interaction_types.FileSearchCallStep):
            return GroundingToolRecord(
                tool="file_search",
                phase="call",
                step_index=step_index,
                call_id=getattr(step, "id", None),
            )
        if isinstance(step, interaction_types.FileSearchResultStep):
            return GroundingToolRecord(
                tool="file_search",
                phase="result",
                step_index=step_index,
                call_id=getattr(step, "call_id", None),
            )
        return None

    @staticmethod
    def _protocol_error(detail: str) -> "InteractionExecutionError":
        return InteractionExecutionError(GenerationFailureKind.INTERACTION_ERROR, detail)


class EndpointIdentity(BaseModel):
    """Credential and endpoint scope for Interaction IDs, files, and continuations."""

    service: Literal["developer", "enterprise"]
    credential_fingerprint: str
    project: str | None = None
    location: str | None = None
    base_url: str | None = None
    api_version: str
    model: str | None = None

    def for_model(self, model: str) -> "EndpointIdentity":
        return self.model_copy(update={"model": model})

    @property
    def scope(self) -> str:
        return xxhash.xxh64(self.model_dump_json(exclude_none=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GenAIClientBinding:
    """Project-owned client wrapper that carries its non-portable endpoint scope."""

    client: genai.Client
    identity: EndpointIdentity

    def close(self) -> None:
        self.client.close()

    async def aclose(self) -> None:
        await self.client.aio.aclose()


class GenerationFailureKind(StrEnum):
    RATE_LIMIT = "rate_limit"
    UNAVAILABLE = "unavailable"
    TRANSPORT = "transport"
    PERMISSION = "permission"
    INVALID_REQUEST = "invalid_request"
    INTERACTION_ERROR = "interaction_error"
    INTERACTION_STATUS = "interaction_status"
    MODEL_OUTPUT_STATUS = "model_output_status"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GenerationFailure:
    kind: GenerationFailureKind
    retryable_across_endpoint: bool
    detail: str


class InteractionExecutionError(Exception):
    """Typed failure emitted by the Interaction event/status reducer."""

    def __init__(self, kind: GenerationFailureKind, detail: str):
        if kind not in {
            GenerationFailureKind.INTERACTION_ERROR,
            GenerationFailureKind.INTERACTION_STATUS,
            GenerationFailureKind.MODEL_OUTPUT_STATUS,
        }:
            raise ValueError(f"Invalid Interaction failure kind: {kind}")
        self.kind = kind
        super().__init__(detail)


class ResolvedGeminiFeatures(BaseModel):
    """Single canonical feature contract consumed by routing and request assembly."""

    reasoning: bool = False
    google_search: bool = False
    code_execution: bool = False
    url_context: bool = False
    google_maps: bool = False
    paid_api: bool = False
    enterprise: bool = False


class InteractionRequestOptions(BaseModel):
    """Validated top-level options for ``interactions.create``."""

    generation_config: interaction_types.GenerationConfig
    system_instruction: str | None = None
    safety_settings: list[interaction_types.SafetySetting] = Field(default_factory=list)
    response_format: interaction_types.CreateModelInteractionResponseFormat | None = None
    tools: list[interaction_types.Tool] = Field(default_factory=list)
    features: ResolvedGeminiFeatures


class GenaiApiError(Exception):
    """Custom exception for errors during Genai API interactions."""

    pass


class FilesAPIError(Exception):
    """Custom exception for errors during Files API operations."""

    pass


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be prepared within Gemini API limits."""

    pass


class ContentBuildError(Exception):
    """Raised when request content cannot be prepared for Gemini."""

    pass


class ForeignEndpointReplayError(ValueError):
    """Raised when persisted provider media belongs to another endpoint identity."""

    pass


class GeminiPDFProcessor:
    """
    Prepares PDFs for Gemini's per-document limits.

    The processor keeps the PDF as a PDF. It removes page thumbnail entries and
    saves with compressed streams/object streams, then splits by page range when
    either the byte or page limit still requires it.
    """

    def __init__(
        self,
        *,
        max_bytes: int = GEMINI_PDF_MAX_BYTES,
        target_bytes: int = GEMINI_PDF_SAFE_TARGET_BYTES,
        max_pages: int = GEMINI_PDF_MAX_PAGES,
    ):
        self.max_bytes = max_bytes
        self.target_bytes = min(target_bytes, max_bytes)
        self.max_pages = max_pages

    def prepare(self, file_bytes: bytes) -> tuple[list[bytes], int, bool]:
        """
        Backward-compatible byte API used by focused tests and raw fallback paths.
        Production PDF mitigation uses prepare_to_directory() to avoid retaining
        optimized/split chunks in memory.
        """
        pikepdf = self._get_pikepdf()
        page_count = self._count_pages(pikepdf, file_bytes)

        if len(file_bytes) <= self.max_bytes and page_count <= self.max_pages:
            return [file_bytes], page_count, False

        optimized_bytes = self._optimize_pdf(pikepdf, file_bytes)
        optimized_page_count = self._count_pages(pikepdf, optimized_bytes)

        if len(optimized_bytes) <= self.max_bytes and optimized_page_count <= self.max_pages:
            return [optimized_bytes], optimized_page_count, True

        chunks = self._split_pdf(pikepdf, optimized_bytes)
        return chunks, optimized_page_count, True

    def prepare_to_directory(
        self,
        source_path: str,
        output_dir: str,
        *,
        source_size: int | None = None,
    ) -> PreparedPDFResult:
        pikepdf = self._get_pikepdf()
        source_size = source_size if source_size is not None else os.path.getsize(source_path)
        page_count = self._count_pages_from_path(pikepdf, source_path)

        if source_size <= self.max_bytes and page_count <= self.max_pages:
            return PreparedPDFResult([], page_count, False)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if source_size <= self.max_bytes and page_count > self.max_pages:
            parts = self._split_pdf_to_directory(
                pikepdf,
                source_path,
                str(output_path),
                source_size=source_size,
                total_pages=page_count,
            )
            return PreparedPDFResult(parts, page_count, True)

        optimized_path = str(output_path / "optimized.pdf")
        self._optimize_pdf_to_path(pikepdf, source_path, optimized_path)
        optimized_size = os.path.getsize(optimized_path)
        optimized_page_count = self._count_pages_from_path(pikepdf, optimized_path)

        if optimized_size <= self.max_bytes and optimized_page_count <= self.max_pages:
            return PreparedPDFResult(
                [
                    PreparedPDFPart(
                        path=optimized_path,
                        size=optimized_size,
                        start_page=1,
                        end_page=optimized_page_count,
                    )
                ],
                optimized_page_count,
                True,
            )

        parts = self._split_pdf_to_directory(
            pikepdf,
            optimized_path,
            str(output_path),
            source_size=optimized_size,
            total_pages=optimized_page_count,
        )
        if os.path.exists(optimized_path):
            os.remove(optimized_path)
        return PreparedPDFResult(parts, optimized_page_count, True)

    @staticmethod
    def _get_pikepdf() -> Any:
        try:
            import pikepdf
        except ImportError as e:
            raise PDFProcessingError(
                "PDF mitigation requires the 'pikepdf' package. "
                "Install the plugin requirements and try again."
            ) from e
        return pikepdf

    @staticmethod
    def _open_pdf(pikepdf: Any, file_bytes: bytes) -> Any:
        try:
            return pikepdf.open(io.BytesIO(file_bytes))
        except Exception as e:
            raise PDFProcessingError(f"Could not open PDF for processing: {e}") from e

    def _count_pages(self, pikepdf: Any, file_bytes: bytes) -> int:
        with self._open_pdf(pikepdf, file_bytes) as pdf:
            return len(pdf.pages)

    @staticmethod
    def _open_pdf_path(pikepdf: Any, file_path: str) -> Any:
        try:
            return pikepdf.open(file_path)
        except Exception as e:
            raise PDFProcessingError(f"Could not open PDF for processing: {e}") from e

    def _count_pages_from_path(self, pikepdf: Any, file_path: str) -> int:
        with self._open_pdf_path(pikepdf, file_path) as pdf:
            return len(pdf.pages)

    def _optimize_pdf(self, pikepdf: Any, file_bytes: bytes) -> bytes:
        with self._open_pdf(pikepdf, file_bytes) as pdf:
            self._remove_page_thumbnails(pikepdf, pdf)
            return self._save_pdf(pikepdf, pdf)

    def _optimize_pdf_to_path(self, pikepdf: Any, source_path: str, destination_path: str) -> None:
        with self._open_pdf_path(pikepdf, source_path) as pdf:
            self._remove_page_thumbnails(pikepdf, pdf)
            self._save_pdf_to_path(pikepdf, pdf, destination_path)

    @staticmethod
    def _remove_page_thumbnails(pikepdf: Any, pdf: Any) -> int:
        removed = 0
        thumb_name = pikepdf.Name("/Thumb")
        for page in pdf.pages:
            if thumb_name in page.obj:
                del page.obj[thumb_name]
                removed += 1
        return removed

    @staticmethod
    def _save_pdf(pikepdf: Any, pdf: Any) -> bytes:
        buffer = io.BytesIO()
        save_kwargs = {
            "compress_streams": True,
            "object_stream_mode": pikepdf.ObjectStreamMode.generate,
            "recompress_flate": True,
        }
        try:
            pdf.save(buffer, **save_kwargs)
        except TypeError:
            save_kwargs.pop("recompress_flate", None)
            buffer = io.BytesIO()
            pdf.save(buffer, **save_kwargs)
        return buffer.getvalue()

    @staticmethod
    def _save_pdf_to_path(pikepdf: Any, pdf: Any, destination_path: str) -> None:
        save_kwargs = {
            "compress_streams": True,
            "object_stream_mode": pikepdf.ObjectStreamMode.generate,
            "recompress_flate": True,
        }
        try:
            pdf.save(destination_path, **save_kwargs)
        except TypeError:
            save_kwargs.pop("recompress_flate", None)
            pdf.save(destination_path, **save_kwargs)

    def _split_pdf(self, pikepdf: Any, file_bytes: bytes) -> list[bytes]:
        with self._open_pdf(pikepdf, file_bytes) as pdf:
            total_pages = len(pdf.pages)
            chunks: list[bytes] = []
            start = 0

            while start < total_pages:
                remaining_pages = total_pages - start
                high = min(self.max_pages, remaining_pages)
                best_chunk: bytes | None = None
                best_page_count = 0
                low = 1

                while low <= high:
                    page_count = (low + high) // 2
                    candidate = self._save_page_range(pikepdf, pdf, start, start + page_count)

                    if len(candidate) <= self.target_bytes:
                        best_chunk = candidate
                        best_page_count = page_count
                        low = page_count + 1
                    else:
                        high = page_count - 1

                if best_chunk is None:
                    single_page = self._save_page_range(pikepdf, pdf, start, start + 1)
                    if len(single_page) > self.max_bytes:
                        raise PDFProcessingError(
                            "A single PDF page remains larger than Gemini's 50 MB "
                            "per-document limit after compression. This PDF cannot "
                            "be sent without lossy page/image downsampling."
                        )
                    best_chunk = single_page
                    best_page_count = 1

                chunks.append(best_chunk)
                start += best_page_count

        return chunks

    def _split_pdf_to_directory(
        self,
        pikepdf: Any,
        source_path: str,
        output_dir: str,
        *,
        source_size: int,
        total_pages: int,
    ) -> list[PreparedPDFPart]:
        with self._open_pdf_path(pikepdf, source_path) as pdf:
            parts: list[PreparedPDFPart] = []
            start = 0
            average_page_bytes = max(1.0, source_size / max(1, total_pages))
            estimated_pages = max(
                1,
                min(
                    self.max_pages,
                    int((self.target_bytes / average_page_bytes) * 0.92),
                ),
            )

            while start < total_pages:
                remaining_pages = total_pages - start
                page_count = min(self.max_pages, estimated_pages, remaining_pages)

                while True:
                    part_number = len(parts) + 1
                    start_page = start + 1
                    end_page = start + page_count
                    candidate_path = os.path.join(output_dir, f"part-{part_number:04d}.tmp.pdf")
                    self._save_page_range_to_path(
                        pikepdf, pdf, start, start + page_count, candidate_path
                    )
                    candidate_size = os.path.getsize(candidate_path)

                    if candidate_size <= self.target_bytes:
                        final_path = os.path.join(
                            output_dir,
                            f"part-{part_number:04d}-pages-{start_page:06d}-{end_page:06d}.pdf",
                        )
                        os.replace(candidate_path, final_path)
                        parts.append(
                            PreparedPDFPart(
                                path=final_path,
                                size=candidate_size,
                                start_page=start_page,
                                end_page=end_page,
                            )
                        )
                        start += page_count
                        average_page_bytes = max(
                            1.0,
                            (
                                (average_page_bytes * max(1, total_pages))
                                + (candidate_size / page_count)
                            )
                            / (max(1, total_pages) + 1),
                        )
                        estimated_pages = max(
                            1,
                            min(
                                self.max_pages,
                                int((self.target_bytes / average_page_bytes) * 0.92),
                            ),
                        )
                        break

                    if page_count == 1:
                        if candidate_size > self.max_bytes:
                            raise PDFProcessingError(
                                "A single PDF page remains larger than Gemini's 50 MB "
                                "per-document limit after compression. This PDF cannot "
                                "be sent without lossy page/image downsampling."
                            )
                        final_path = os.path.join(
                            output_dir,
                            f"part-{part_number:04d}-pages-{start_page:06d}-{end_page:06d}.pdf",
                        )
                        os.replace(candidate_path, final_path)
                        parts.append(
                            PreparedPDFPart(
                                path=final_path,
                                size=candidate_size,
                                start_page=start_page,
                                end_page=end_page,
                            )
                        )
                        start += 1
                        break

                    next_count = int(page_count * (self.target_bytes / candidate_size) * 0.88)
                    page_count = max(1, min(page_count - 1, next_count))

        return parts

    def _save_page_range(self, pikepdf: Any, source_pdf: Any, start: int, stop: int) -> bytes:
        chunk_pdf = pikepdf.Pdf.new()
        chunk_pdf.pages.extend(source_pdf.pages[start:stop])
        self._remove_page_thumbnails(pikepdf, chunk_pdf)
        return self._save_pdf(pikepdf, chunk_pdf)

    def _save_page_range_to_path(
        self,
        pikepdf: Any,
        source_pdf: Any,
        start: int,
        stop: int,
        destination_path: str,
    ) -> None:
        chunk_pdf = pikepdf.Pdf.new()
        chunk_pdf.pages.extend(source_pdf.pages[start:stop])
        self._remove_page_thumbnails(pikepdf, chunk_pdf)
        self._save_pdf_to_path(pikepdf, chunk_pdf, destination_path)


class UploadStatusManager:
    """
    Manages and centralizes status updates for concurrent file uploads.

    This manager is self-configuring. It discovers the number of files that
    require an actual upload at runtime, only showing a status message to the
    user when network activity is necessary.

    The communication protocol uses tuples sent via an asyncio.Queue:
    - ('REGISTER_UPLOAD',): Sent by a worker when it determines an upload is needed.
    - ('COMPLETE_UPLOAD',): Sent by a worker when its upload is finished.
    - ('FINALIZE',): Sent by the orchestrator when all workers are done.
    """

    def __init__(
        self,
        event_emitter: "EventEmitter",
    ):
        self.event_emitter = event_emitter
        self.queue = asyncio.Queue()
        self.total_uploads_expected = 0
        self.uploads_completed = 0
        self.finalize_received = False
        self.is_active = False

    async def run(self) -> None:
        """
        Runs the manager loop, listening for updates and emitting status to the UI.
        This should be started as a background task using asyncio.create_task().
        """
        while not (
            self.finalize_received and self.total_uploads_expected == self.uploads_completed
        ):
            msg = await self.queue.get()
            msg_type = msg[0]

            if msg_type == "REGISTER_UPLOAD":
                self.is_active = True
                self.total_uploads_expected += 1
                await self._emit_progress_update()
            elif msg_type == "COMPLETE_UPLOAD":
                self.uploads_completed += 1
                await self._emit_progress_update()
            elif msg_type == "FINALIZE":
                self.finalize_received = True

            self.queue.task_done()

        log.debug("UploadStatusManager finished its run.")

    async def _emit_progress_update(self) -> None:
        """Emits the current progress to the front-end if uploads are active."""
        if not self.is_active:
            return

        is_done = (
            self.total_uploads_expected > 0
            and self.uploads_completed == self.total_uploads_expected
        )

        if is_done:
            message = f"Upload complete. {self.uploads_completed} file(s) processed."
        else:
            # Show "Uploading 1 of N..."
            message = (
                f"Uploading file {self.uploads_completed + 1} of {self.total_uploads_expected}..."
            )

        self.event_emitter.emit_status(message, done=is_done, indent_level=1)


class FilesAPIManager:
    """
    Manages uploading, caching, and retrieving files using the Google Gemini Files API.

    This class provides a stateless and efficient way to handle files by using a fast,
    non-cryptographic hash (xxHash) of the file's content as the primary identifier.
    This enables content-addressable storage, preventing duplicate uploads of the
    same file. It uses a multi-tiered approach:

    1. Hot Path (In-Memory Caches): For instantly retrieving file objects and hashes
       for recently used files.
    2. Warm Path (Stateless GET): For quickly recovering file state after a server
       restart by using a deterministic name (derived from the content hash) and a
       single `get` API call.
    3. Cold Path (Upload): As a last resort, for uploading new files or re-uploading
       expired ones.
    """

    def __init__(
        self,
        client: genai.Client,
        endpoint_identity: EndpointIdentity,
        file_cache: SimpleMemoryCache,
        id_hash_cache: SimpleMemoryCache,
        event_emitter: "EventEmitter",
    ):
        """
        Initializes the FilesAPIManager.

        Args:
            client: An initialized `google.genai.Client` instance.
            file_cache: An aiocache instance for mapping `content_hash -> types.File`.
                        Must be configured with `aiocache.serializers.NullSerializer`.
            id_hash_cache: An aiocache instance for mapping `owui_file_id -> content_hash`.
                           This is an optimization to avoid re-hashing known files.
            event_emitter: An abstract class for emitting events to the front-end.
        """
        self.client = client
        self.endpoint_identity = endpoint_identity
        self.file_cache = file_cache
        self.id_hash_cache = id_hash_cache
        self.event_emitter = event_emitter
        # A dictionary to manage locks for concurrent uploads.
        # The key is a composite of api_key_hash and content_hash.
        self.upload_locks: dict[str, asyncio.Lock] = {}
        self.api_key_hash = endpoint_identity.scope

    def _get_file_cache_key(self, content_hash: str) -> str:
        """Gets the namespaced key for the file cache."""
        return f"{self.api_key_hash}:{content_hash}"

    def _get_lock_key(self, content_hash: str) -> str:
        """Gets the namespaced key for upload locks."""
        # Although the deterministic_name is content-based, the file's ownership
        # is tied to the API key (project). Locking per API key + content hash
        # allows concurrent uploads of the same file for different users.
        return f"{self.api_key_hash}:{content_hash}"

    async def get_or_upload_file(
        self,
        file_bytes: bytes,
        mime_type: str,
        *,
        owui_file_id: str | None = None,
        status_queue: asyncio.Queue | None = None,
    ) -> types.File:
        """
        The main public method to get a file, using caching, recovery, or uploading.

        This method uses a fast content hash (xxHash) as the primary key for all
        caching and remote API interactions to ensure deduplication and performance.
        It is safe from race conditions during concurrent uploads.

        Args:
            file_bytes: The raw byte content of the file. Required.
            mime_type: The MIME type of the file (e.g., 'image/png'). Required.
            owui_file_id: The unique ID of the file from Open WebUI, if available.
                      RECOMMENDED_COMPANION_VERSION    Used for logging and as a key for the hash cache optimization.
            status_queue: An optional asyncio.Queue to report upload lifecycle events.

        Returns:
            An `ACTIVE` `google.genai.types.File` object.

        Raises:
            FilesAPIError: If the file fails to upload or process.
        """
        # Step 1: Get the fast content hash, using the ID cache as an optimization if possible.
        content_hash = await self._get_content_hash(file_bytes, owui_file_id)

        # Step 2: The Hot Path (Check Local File Cache)
        # A cache hit means the file is valid and we can return immediately.
        file_cache_key = self._get_file_cache_key(content_hash)
        cached_file: types.File | None = await self.file_cache.get(file_cache_key)
        if cached_file:
            log_id = f"OWUI ID: {owui_file_id}" if owui_file_id else "anonymous file"
            log.debug(f"Cache HIT for file hash {content_hash} ({log_id}). Returning immediately.")
            return cached_file

        # On cache miss, acquire a lock specific to this file's content to prevent race conditions.
        # dict.setdefault is atomic, ensuring only one lock is created per hash.
        lock_key = self._get_lock_key(content_hash)
        lock = self.upload_locks.setdefault(lock_key, asyncio.Lock())
        if lock.locked():
            log.debug(
                f"Lock for key {lock_key} is held by another task. "
                f"This call will now wait for the lock to be released."
            )

        async with lock:
            # Step 2.5: Double-Checked Locking
            # After acquiring the lock, check the cache again. Another task might have
            # completed the upload while we were waiting for the lock.
            cached_file = await self.file_cache.get(file_cache_key)
            if cached_file:
                log.debug(
                    f"Cache HIT for file hash {content_hash} after acquiring lock. Returning."
                )
                return cached_file

            # Step 3: The Warm/Cold Path (On Cache Miss)
            # The file ID (name after "files/") must be <= 40 chars.
            # "owui-" (5) + hash (16) + "-" (1) + hash (16) = 38 chars.
            deterministic_name = f"files/owui-{self.api_key_hash}-{content_hash}"
            log.debug(
                f"Cache MISS for hash {content_hash}. Attempting stateless recovery with GET: {deterministic_name}"
            )

            try:
                # Attempt to get the file (Warm Path)
                file = await self.client.aio.files.get(name=deterministic_name)
                if not file.name:
                    raise FilesAPIError(
                        f"Stateless recovery for {deterministic_name} returned a file without a name."
                    )

                log.debug(
                    f"Stateless recovery successful for {deterministic_name}. File exists on server."
                )
                active_file = await self._poll_for_active_state(file.name, owui_file_id)

                ttl_seconds = self._calculate_ttl(active_file.expiration_time)
                await self.file_cache.set(file_cache_key, active_file, ttl=ttl_seconds)

                return active_file
            except genai_errors.ClientError as e:
                # NOTE: The Gemini Files API returns 403 Forbidden when trying to GET
                # a file that either does not exist or belongs to another project.
                # We treat 403 as the "not found" signal for our warm path and
                # include 404 for forward compatibility.
                if e.code == 403 or e.code == 404:
                    log.info(
                        f"File {deterministic_name} not found on server (received {e.code}). Proceeding to upload."
                    )
                    # Proceed to upload (Cold Path)
                    return await self._upload_and_process_file(
                        content_hash,
                        file_bytes,
                        mime_type,
                        deterministic_name,
                        owui_file_id,
                        status_queue,
                    )
                else:
                    log.exception(
                        f"An unhandled client error (code: {e.code}) occurred during stateless recovery for {deterministic_name}."
                    )
                    self.event_emitter.emit_toast(
                        f"API error for file: {e.code}. Please check permissions.",
                        "error",
                    )
                    raise FilesAPIError(
                        f"Failed to check file status for {deterministic_name}: {e}"
                    ) from e
            except Exception as e:
                log.exception(
                    f"An unexpected error occurred during stateless recovery for {deterministic_name}."
                )
                self.event_emitter.emit_toast(
                    "Unexpected error retrieving a file. Please try again.",
                    "error",
                )
                raise FilesAPIError(
                    f"Failed to check file status for {deterministic_name}: {e}"
                ) from e
            finally:
                # Clean up the lock from the dictionary once processing is complete
                # for this hash, preventing memory growth over time.
                # This is safe because any future request for this hash will hit the cache.
                if lock_key in self.upload_locks:
                    del self.upload_locks[lock_key]

    async def _get_content_hash(self, file_bytes: bytes, owui_file_id: str | None) -> str:
        """
        Retrieves the file's content hash, using a cache for known IDs or computing it.

        This acts as a memoization layer for the hashing process, avoiding
        re-computation for files with a known Open WebUI ID. For anonymous files
        (owui_file_id=None), it will always compute the hash.
        """
        if owui_file_id:
            # First, check the ID-to-Hash cache for known files.
            # This cache is NOT namespaced by API key, as the mapping from
            # an OWUI file ID to its content hash is constant.
            cached_hash: str | None = await self.id_hash_cache.get(owui_file_id)
            if cached_hash:
                log.trace(f"Hash cache HIT for OWUI ID {owui_file_id}.")
                return cached_hash

        # If not in cache or if file is anonymous, compute the fast hash.
        log.trace(
            f"Hash cache MISS for OWUI ID {owui_file_id if owui_file_id else 'N/A'}. Computing hash."
        )
        content_hash = xxhash.xxh64(file_bytes).hexdigest()

        # If there was an ID, store the newly computed hash for next time.
        if owui_file_id:
            await self.id_hash_cache.set(owui_file_id, content_hash)

        return content_hash

    async def get_or_upload_file_from_path(
        self,
        file_path: str,
        mime_type: str,
        *,
        owui_file_id: str | None = None,
        status_queue: asyncio.Queue | None = None,
    ) -> types.File:
        content_hash = await self._get_content_hash_from_path(file_path, owui_file_id)
        file_cache_key = self._get_file_cache_key(content_hash)
        cached_file: types.File | None = await self.file_cache.get(file_cache_key)
        if cached_file:
            log_id = f"OWUI ID: {owui_file_id}" if owui_file_id else "anonymous file"
            log.debug(f"Cache HIT for file hash {content_hash} ({log_id}). Returning immediately.")
            return cached_file

        lock_key = self._get_lock_key(content_hash)
        lock = self.upload_locks.setdefault(lock_key, asyncio.Lock())
        if lock.locked():
            log.debug(
                f"Lock for key {lock_key} is held by another task. "
                f"This call will now wait for the lock to be released."
            )

        async with lock:
            cached_file = await self.file_cache.get(file_cache_key)
            if cached_file:
                log.debug(
                    f"Cache HIT for file hash {content_hash} after acquiring lock. Returning."
                )
                return cached_file

            deterministic_name = f"files/owui-{self.api_key_hash}-{content_hash}"
            log.debug(
                f"Cache MISS for hash {content_hash}. Attempting stateless recovery with GET: {deterministic_name}"
            )

            try:
                file = await self.client.aio.files.get(name=deterministic_name)
                if not file.name:
                    raise FilesAPIError(
                        f"Stateless recovery for {deterministic_name} returned a file without a name."
                    )

                log.debug(
                    f"Stateless recovery successful for {deterministic_name}. File exists on server."
                )
                active_file = await self._poll_for_active_state(file.name, owui_file_id)

                ttl_seconds = self._calculate_ttl(active_file.expiration_time)
                await self.file_cache.set(file_cache_key, active_file, ttl=ttl_seconds)

                return active_file
            except genai_errors.ClientError as e:
                if e.code == 403 or e.code == 404:
                    log.info(
                        f"File {deterministic_name} not found on server (received {e.code}). Proceeding to upload."
                    )
                    return await self._upload_and_process_file_from_path(
                        content_hash,
                        file_path,
                        mime_type,
                        deterministic_name,
                        owui_file_id,
                        status_queue,
                    )
                else:
                    log.exception(
                        f"An unhandled client error (code: {e.code}) occurred during stateless recovery for {deterministic_name}."
                    )
                    self.event_emitter.emit_toast(
                        f"API error for file: {e.code}. Please check permissions.",
                        "error",
                    )
                    raise FilesAPIError(
                        f"Failed to check file status for {deterministic_name}: {e}"
                    ) from e
            except Exception as e:
                log.exception(
                    f"An unexpected error occurred during stateless recovery for {deterministic_name}."
                )
                self.event_emitter.emit_toast(
                    "Unexpected error retrieving a file. Please try again.",
                    "error",
                )
                raise FilesAPIError(
                    f"Failed to check file status for {deterministic_name}: {e}"
                ) from e
            finally:
                if lock_key in self.upload_locks:
                    del self.upload_locks[lock_key]

    async def _get_content_hash_from_path(self, file_path: str, owui_file_id: str | None) -> str:
        if owui_file_id:
            cached_hash: str | None = await self.id_hash_cache.get(owui_file_id)
            if cached_hash:
                log.trace(f"Hash cache HIT for OWUI ID {owui_file_id}.")
                return cached_hash

        log.trace(
            f"Hash cache MISS for OWUI ID {owui_file_id if owui_file_id else 'N/A'}. Computing hash from path."
        )
        content_hash = await asyncio.to_thread(self._hash_file_path, file_path)

        if owui_file_id:
            await self.id_hash_cache.set(owui_file_id, content_hash)

        return content_hash

    @staticmethod
    def _hash_file_path(file_path: str) -> str:
        digest = xxhash.xxh64()
        with open(file_path, "rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _calculate_ttl(self, expiration_time: datetime | None) -> float | None:
        """Calculates the TTL in seconds from an expiration datetime."""
        if not expiration_time:
            return None

        now_utc = datetime.now(UTC)
        if expiration_time <= now_utc:
            return 0

        return (expiration_time - now_utc).total_seconds()

    @staticmethod
    def _is_already_exists_error(error: Exception) -> bool:
        if not isinstance(error, genai_errors.ClientError):
            return False
        return error.code == 409 or getattr(error, "status", "") == "ALREADY_EXISTS"

    async def _cache_active_file(self, content_hash: str, active_file: types.File) -> None:
        ttl_seconds = self._calculate_ttl(active_file.expiration_time)
        file_cache_key = self._get_file_cache_key(content_hash)
        await self.file_cache.set(file_cache_key, active_file, ttl=ttl_seconds)
        log.debug(f"Cached file object for hash {content_hash} with TTL: {ttl_seconds}s.")

    async def _recover_after_upload_conflict(
        self,
        content_hash: str,
        deterministic_name: str,
        owui_file_id: str | None,
        *,
        attempts: int = 5,
        retry_delay: float = 0.5,
    ) -> types.File:
        """
        Recover when deterministic create reports ALREADY_EXISTS.

        The Files API can reject create with 409 for a deterministic name even
        after the preceding stateless GET did not return the file. Treat the
        conflict as an idempotent success path by fetching the existing object,
        allowing a short retry window for service-side consistency.
        """
        log.info(f"Upload conflict for {deterministic_name}; attempting to reuse existing file.")
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                file = await self.client.aio.files.get(name=deterministic_name)
                if not file.name:
                    raise FilesAPIError(
                        f"Conflict recovery for {deterministic_name} returned a file without a name."
                    )

                if file.state == types.FileState.ACTIVE:
                    active_file = file
                else:
                    active_file = await self._poll_for_active_state(file.name, owui_file_id)

                await self._cache_active_file(content_hash, active_file)
                return active_file
            except FilesAPIError:
                raise
            except Exception as e:
                last_error = e
                if attempt == attempts:
                    break
                log.debug(
                    f"Conflict recovery GET for {deterministic_name} failed "
                    f"on attempt {attempt}/{attempts}; retrying."
                )
                await asyncio.sleep(retry_delay)

        raise FilesAPIError(
            f"Upload conflict recovery failed for {deterministic_name}: {last_error}"
        ) from last_error

    async def _upload_and_process_file(
        self,
        content_hash: str,
        file_bytes: bytes,
        mime_type: str,
        deterministic_name: str,
        owui_file_id: str | None,
        status_queue: asyncio.Queue | None = None,
    ) -> types.File:
        """Handles the full upload and post-upload processing workflow."""

        # Register with the manager that an actual upload is starting.
        if status_queue:
            await status_queue.put(("REGISTER_UPLOAD",))

        log.info(f"Starting upload for {deterministic_name}...")

        try:
            file_io = io.BytesIO(file_bytes)
            upload_config = types.UploadFileConfig(name=deterministic_name, mime_type=mime_type)
            uploaded_file = await self.client.aio.files.upload(file=file_io, config=upload_config)
            if not uploaded_file.name:
                raise FilesAPIError(
                    f"File upload for {deterministic_name} did not return a file name."
                )

            log.debug(f"{uploaded_file.name} uploaded.")
            log.trace("Uploaded file details omitted from logs.")

            # Check if the file is already active. If so, we can skip polling.
            if uploaded_file.state == types.FileState.ACTIVE:
                log.debug(f"File {uploaded_file.name} is already ACTIVE. Skipping poll.")
                active_file = uploaded_file
            else:
                # If not active, proceed with the original polling logic.
                log.debug(
                    f"{uploaded_file.name} uploaded with state {uploaded_file.state}. Polling for ACTIVE state."
                )
                active_file = await self._poll_for_active_state(uploaded_file.name, owui_file_id)
                log.debug(f"File {active_file.name} is now ACTIVE.")

            await self._cache_active_file(content_hash, active_file)

            return active_file
        except genai_errors.ClientError as e:
            if self._is_already_exists_error(e):
                return await self._recover_after_upload_conflict(
                    content_hash, deterministic_name, owui_file_id
                )
            log.exception(f"File upload or processing failed for {deterministic_name}.")
            self.event_emitter.emit_toast(
                "Upload failed for a file. Please check connection and try again.",
                "error",
            )
            raise FilesAPIError(f"Upload failed for {deterministic_name}: {e}") from e
        except Exception as e:
            log.exception(f"File upload or processing failed for {deterministic_name}.")
            self.event_emitter.emit_toast(
                "Upload failed for a file. Please check connection and try again.",
                "error",
            )
            raise FilesAPIError(f"Upload failed for {deterministic_name}: {e}") from e
        finally:
            # Report completion (success or failure) to the status manager.
            # This ensures the progress counter always advances.
            if status_queue:
                await status_queue.put(("COMPLETE_UPLOAD",))

    async def _upload_and_process_file_from_path(
        self,
        content_hash: str,
        file_path: str,
        mime_type: str,
        deterministic_name: str,
        owui_file_id: str | None,
        status_queue: asyncio.Queue | None = None,
    ) -> types.File:
        """Uploads a local file without first loading the full payload into memory."""

        if status_queue:
            await status_queue.put(("REGISTER_UPLOAD",))

        log.info(f"Starting upload for {deterministic_name} from local path...")

        try:
            upload_config = types.UploadFileConfig(name=deterministic_name, mime_type=mime_type)
            with open(file_path, "rb") as file_io:
                uploaded_file = await self.client.aio.files.upload(
                    file=file_io, config=upload_config
                )
            if not uploaded_file.name:
                raise FilesAPIError(
                    f"File upload for {deterministic_name} did not return a file name."
                )

            log.debug(f"{uploaded_file.name} uploaded.")
            log.trace("Uploaded file details omitted from logs.")

            if uploaded_file.state == types.FileState.ACTIVE:
                log.debug(f"File {uploaded_file.name} is already ACTIVE. Skipping poll.")
                active_file = uploaded_file
            else:
                log.debug(
                    f"{uploaded_file.name} uploaded with state {uploaded_file.state}. Polling for ACTIVE state."
                )
                active_file = await self._poll_for_active_state(uploaded_file.name, owui_file_id)
                log.debug(f"File {active_file.name} is now ACTIVE.")

            await self._cache_active_file(content_hash, active_file)

            return active_file
        except genai_errors.ClientError as e:
            if self._is_already_exists_error(e):
                return await self._recover_after_upload_conflict(
                    content_hash, deterministic_name, owui_file_id
                )
            log.exception(f"File upload or processing failed for {deterministic_name}.")
            self.event_emitter.emit_toast(
                "Upload failed for a file. Please check connection and try again.",
                "error",
            )
            raise FilesAPIError(f"Upload failed for {deterministic_name}: {e}") from e
        except Exception as e:
            log.exception(f"File upload or processing failed for {deterministic_name}.")
            self.event_emitter.emit_toast(
                "Upload failed for a file. Please check connection and try again.",
                "error",
            )
            raise FilesAPIError(f"Upload failed for {deterministic_name}: {e}") from e
        finally:
            if status_queue:
                await status_queue.put(("COMPLETE_UPLOAD",))

    async def _poll_for_active_state(
        self,
        file_name: str,
        owui_file_id: str | None,
        timeout: int = 60,
        poll_interval: int = 1,
    ) -> types.File:
        """Polls the file's status until it is ACTIVE or fails."""
        end_time = time.monotonic() + timeout
        while time.monotonic() < end_time:
            try:
                file = await self.client.aio.files.get(name=file_name)
            except Exception as e:
                raise FilesAPIError(
                    f"Polling failed: Could not get status for {file_name}. Reason: {e}"
                ) from e

            if file.state == types.FileState.ACTIVE:
                return file
            if file.state == types.FileState.FAILED:
                log_id = f"'{owui_file_id}'" if owui_file_id else "an uploaded file"
                error_message = f"File processing failed on server for {file_name}."
                toast_message = f"Google could not process {log_id}."
                if file.error:
                    reason = f"Reason: {file.error.message} (Code: {file.error.code})"
                    error_message += f" {reason}"
                    toast_message += f" Reason: {file.error.message}"

                self.event_emitter.emit_toast(toast_message, "error")
                raise FilesAPIError(error_message)

            state_name = file.state.name if file.state else "UNKNOWN"
            log.trace(f"File {file_name} is still {state_name}. Waiting {poll_interval}s...")
            await asyncio.sleep(poll_interval)

        raise FilesAPIError(f"File {file_name} did not become ACTIVE within {timeout} seconds.")


class PDFMitigationManager:
    """Coordinates PDF source files, cache entries, and serialized processing."""

    def __init__(
        self,
        *,
        cache: SimpleMemoryCache | None = None,
        processing_semaphore: asyncio.Semaphore | None = None,
    ):
        self.cache = cache or SimpleMemoryCache(serializer=NullSerializer())
        self.locks: dict[str, asyncio.Lock] = {}
        self.processing_semaphore = processing_semaphore or asyncio.Semaphore(
            GEMINI_PDF_PROCESSING_CONCURRENCY
        )

    async def prepare(
        self,
        *,
        file_bytes: bytes | None,
        file_path: str | None,
    ) -> PDFMitigationOutcome | None:
        if file_path:
            original_hash = await asyncio.to_thread(FilesAPIManager._hash_file_path, file_path)
            return PDFMitigationOutcome(
                original_hash=original_hash,
                result=await self._get_or_prepare(
                    file_path,
                    original_hash,
                    source_size=os.path.getsize(file_path),
                ),
            )

        if not file_bytes:
            return None

        original_hash = xxhash.xxh64(file_bytes).hexdigest()
        cached_result = await self._get_cached_result(original_hash)
        if cached_result:
            return PDFMitigationOutcome(
                original_hash=original_hash,
                result=cached_result,
            )

        source_path = self._write_temp_source(original_hash, file_bytes)
        try:
            result = await self._get_or_prepare(
                source_path,
                original_hash,
                source_size=len(file_bytes),
            )
        finally:
            self._remove_temp_source(source_path)

        return PDFMitigationOutcome(original_hash=original_hash, result=result)

    async def _get_or_prepare(
        self,
        source_path: str,
        original_hash: str,
        *,
        source_size: int,
    ) -> PreparedPDFResult:
        cached_result = await self._get_cached_result(original_hash)
        if cached_result:
            return cached_result

        log.debug(f"PDF mitigation cache MISS for source hash {original_hash}.")
        lock = self.locks.setdefault(original_hash, asyncio.Lock())
        async with lock:
            cached_result = await self._get_cached_result(original_hash)
            if cached_result:
                log.debug(f"PDF mitigation cache HIT for source hash {original_hash} after lock.")
                return cached_result

            cache_key = self._cache_key(original_hash)
            cache_dir = self._cache_dir(original_hash)
            self._cleanup_stale_cache_dirs()
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            processor = GeminiPDFProcessor()
            async with self.processing_semaphore:
                result = await asyncio.to_thread(
                    processor.prepare_to_directory,
                    source_path,
                    str(cache_dir),
                    source_size=source_size,
                )

            if result.was_mitigated:
                await self.cache.set(
                    cache_key,
                    result,
                    ttl=GEMINI_PDF_MITIGATION_CACHE_TTL_SECONDS,
                )
                log.debug(
                    f"Cached PDF mitigation result for source hash {original_hash} "
                    f"with {len(result.parts)} processed PDF(s)."
                )
            elif cache_dir.exists():
                shutil.rmtree(cache_dir)

            return result

    async def _get_cached_result(self, original_hash: str) -> PreparedPDFResult | None:
        cached_result = await self.cache.get(self._cache_key(original_hash))
        if self._cached_result_is_valid(cached_result):
            log.debug(f"PDF mitigation cache HIT for source hash {original_hash}.")
            return cached_result
        return None

    @staticmethod
    def _cache_key(original_hash: str) -> str:
        return f"pdf_mitigation:{original_hash}"

    @staticmethod
    def _cached_result_is_valid(value: Any) -> bool:
        if not isinstance(value, PreparedPDFResult):
            return False
        if not value.was_mitigated:
            return True
        return bool(value.parts) and all(os.path.exists(part.path) for part in value.parts)

    @staticmethod
    def _cache_root() -> Path:
        return Path(tempfile.gettempdir()) / GEMINI_PDF_MITIGATION_CACHE_DIR_NAME

    def _cache_dir(self, original_hash: str) -> Path:
        return self._cache_root() / original_hash

    def _cleanup_stale_cache_dirs(self) -> None:
        cache_root = self._cache_root()
        if not cache_root.exists():
            return
        cutoff = time.time() - GEMINI_PDF_MITIGATION_CACHE_TTL_SECONDS
        for path in cache_root.iterdir():
            try:
                if path.is_dir() and path.stat().st_mtime < cutoff:
                    shutil.rmtree(path)
                elif (
                    path.is_file()
                    and path.name.startswith("source-")
                    and path.stat().st_mtime < cutoff
                ):
                    path.unlink()
            except Exception:
                log.exception(f"Could not clean stale PDF mitigation cache path {path}.")

    @staticmethod
    def _write_temp_source(original_hash: str, file_bytes: bytes) -> str:
        temp_dir = PDFMitigationManager._cache_root()
        temp_dir.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=f"source-{original_hash}-",
            suffix=".pdf",
            dir=temp_dir,
        )
        with os.fdopen(fd, "wb") as file:
            file.write(file_bytes)
        return temp_path

    @staticmethod
    def _remove_temp_source(source_path: str) -> None:
        try:
            os.remove(source_path)
        except FileNotFoundError:
            pass
        except Exception:
            log.exception(f"Could not remove temporary PDF source {source_path}.")


class GeminiContentBuilder:
    """Builds a canonical Interactions step ledger from an Open WebUI request."""

    def __init__(
        self,
        messages_body: list["Message"],
        metadata_body: "Metadata",
        user_data: "UserData",
        event_emitter: "EventEmitter",
        valves: "Pipe.Valves",
        files_api_manager: "FilesAPIManager",
        pdf_mitigation_manager: PDFMitigationManager,
    ):
        self.messages_body = messages_body
        self.upload_documents = (metadata_body.get("features", {}) or {}).get(
            "upload_documents", False
        )
        # Identify if this is a background task (title/tags/etc) to optimize context.
        self.is_task = bool(metadata_body.get("task"))
        self.event_emitter = event_emitter
        self.valves = valves
        self.files_api_manager = files_api_manager
        self.pdf_mitigation_manager = pdf_mitigation_manager
        # FIXME: chat id could be `None`, leading to an iteration error.
        self.is_temp_chat = "local" in metadata_body.get("chat_id", "")
        self.is_enterprise = self.files_api_manager.endpoint_identity.service == "enterprise"

        self.system_prompt, self.messages_body = self._extract_system_prompt(self.messages_body)
        self.metadata_body = metadata_body
        self.user_data = user_data
        self.messages_db = None

    async def build_contents(self) -> list[interaction_types.StepParam]:
        """
        The main public method to generate the contents list by processing all
        message turns concurrently and using a self-configuring status manager.
        """
        # Fetch chat history and cumulative usage from the DB (async APIs).
        self.messages_db = await self._fetch_and_validate_chat_history(
            self.metadata_body, self.user_data
        )
        log.trace("Database history loaded; signed message state omitted from logs.")
        # Retrieve cumulative usage from the DB history and inject it into metadata.
        # This will be picked up later when constructing the final usage payload.
        c_tokens, c_cost = self._retrieve_previous_usage_data()
        self.metadata_body["cumulative_tokens"] = c_tokens
        self.metadata_body["cumulative_cost"] = c_cost

        if not self.messages_db:
            warn_msg = (
                "Database history not ready or lengths mismatched. "
                "Falling back to active memory payload."
            )
            log.warning(warn_msg)
            self.event_emitter.emit_toast(warn_msg, "warning")

        # 1. Set up and launch the status manager. It will activate itself if needed.
        status_manager = UploadStatusManager(self.event_emitter)
        manager_task = asyncio.create_task(status_manager.run())

        # 2. Create and run concurrent processing tasks for each message turn.
        tasks = [
            self._process_message_turn(i, message, status_manager.queue)
            for i, message in enumerate(self.messages_body)
        ]
        log.debug(f"Starting concurrent processing of {len(tasks)} message turns.")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. Signal to the manager that no more uploads will be registered.
        await status_manager.queue.put(("FINALIZE",))

        # 4. Wait for the manager to finish processing all reported uploads.
        await manager_task

        # 5. Flatten message results without changing turn or attachment order.
        steps: list[interaction_types.StepParam] = []
        content_errors: list[Exception] = []
        for i, res in enumerate(results):
            if isinstance(res, list):
                steps.extend(res)
            elif isinstance(res, Exception):
                content_errors.append(res)
                log.error(f"An error occurred while processing message {i} concurrently.")
        if content_errors:
            raise ContentBuildError(str(content_errors[0])) from content_errors[0]
        self._validate_request_size(steps)
        return steps

    @staticmethod
    def _parse_interaction_envelope(message: "ChatMessageTD") -> InteractionEnvelopeV1 | None:
        payload = message.get("gemini_interaction")
        if payload is None:
            return None
        try:
            return InteractionEnvelopeV1.model_validate(payload)
        except pydantic_core.ValidationError as exc:
            raise ContentBuildError(
                "Stored gemini_interaction envelope is malformed or uses an unsupported version."
            ) from exc

    def _canonical_visible_content(self, content: str, sources: list["Source"] | None) -> str:
        content, _thoughts = self._pop_thoughts(content)
        if sources:
            content = self._remove_citation_markers(content, sources)
        return content.strip()

    def select_continuation(
        self,
        full_input: list[interaction_types.StepParam],
        *,
        store: bool,
        endpoint_scope: str,
        model_id: str,
    ) -> ContinuationDecision:
        """Select server continuation only from the immediate active branch parent."""
        if not store:
            return ContinuationDecision(full_input, None, False, "storage_disabled")
        if not self.messages_db or len(self.messages_db) < 2 or len(self.messages_body) < 2:
            return ContinuationDecision(full_input, None, False, "no_parent")
        parent_db = self.messages_db[-2]
        parent_body = self.messages_body[-2]
        if parent_db.get("role") != "assistant" or parent_body.get("role") != "assistant":
            return ContinuationDecision(full_input, None, False, "parent_not_assistant")
        envelope = self._parse_interaction_envelope(parent_db)
        if envelope is None:
            return ContinuationDecision(full_input, None, False, "non_gemini_parent")
        sources = parent_db.get("sources")
        current_content = self._canonical_visible_content(
            cast("AssistantMessage", parent_body).get("content", ""), sources
        )
        eligible = bool(
            envelope.store
            and envelope.status == "completed"
            and envelope.interaction_id
            and envelope.endpoint_scope == endpoint_scope
            and envelope.model_id == model_id
            and current_content == envelope.visible_content.strip()
        )
        if not eligible:
            return ContinuationDecision(full_input, None, False, "parent_not_eligible")
        if not full_input or full_input[-1].get("type") != "user_input":
            raise ContentBuildError("Active branch does not end in the current user input.")
        return ContinuationDecision(
            [full_input[-1]], envelope.interaction_id, True, "same_scope_completed_parent"
        )

    def _validate_request_size(self, steps: list[interaction_types.StepParam]) -> None:
        payload = {
            "input": steps,
            "system_instruction": self.system_prompt,
        }
        payload_size = len(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        )
        if payload_size > GEMINI_INTERACTION_MAX_REQUEST_BYTES:
            raise ContentBuildError(
                f"Interaction request payload is {payload_size} bytes, exceeding the 100 MiB limit."
            )

    def _retrieve_previous_usage_data(self) -> tuple[int | None, float | None]:
        """
        Retrieves the cumulative token count and cost from the last assistant message in the database.

        Returns:
            - (0, 0.0) if it's the start of a conversation (no previous assistant message).
            - (tokens, cost) if the previous assistant message has valid cumulative data.
            - (None, None) if the chain is broken (previous message exists but lacks data)
              or if DB history is unavailable (e.g., temp chat).
        """
        if not self.messages_db:
            return None, None

        for msg in reversed(self.messages_db):
            if msg.get("role") == "assistant":
                usage = msg.get("usage", {})
                # These keys must be populated by the plugin in previous turns
                c_tokens = usage.get("cumulative_token_count")
                c_cost = usage.get("cumulative_total_cost")

                if c_tokens is not None and c_cost is not None:
                    return c_tokens, c_cost
                else:
                    # Previous assistant message exists but lacks cumulative data.
                    # This indicates a broken chain (old message or different plugin).
                    return None, None

        # No assistant message found in history, implying this is the first turn.
        return 0, 0.0

    @staticmethod
    def _extract_system_prompt(
        messages: list["Message"],
    ) -> tuple[str | None, list["Message"]]:
        """Extracts the system prompt and returns it along with the modified message list."""
        system_message, remaining_messages = pop_system_message(messages)  # type: ignore
        system_prompt: str | None = (system_message or {}).get("content")
        return system_prompt, remaining_messages  # type: ignore

    async def _fetch_and_validate_chat_history(
        self, metadata_body: "Metadata", user_data: "UserData"
    ) -> list["ChatMessageTD"] | None:
        """
        Reconstructs the active chat branch from history. Removes the trailing
        assistant placeholder and strictly validates that the DB history length
        matches the request body.
        """
        chat_id = metadata_body.get("chat_id", "")
        if not chat_id or "local" in chat_id:
            return None

        chat = await Chats.get_chat_by_id_and_user_id(id=chat_id, user_id=user_data["id"])
        if not chat:
            return None

        chat_content: ChatObjectDataTD = chat.chat  # type: ignore
        history_data = chat_content.get("history", {})
        messages_dict = history_data.get("messages", {})
        current_id = history_data.get("currentId")

        if not messages_dict or not current_id:
            return None

        # 1. Walk up the parentId chain to reconstruct the linear conversation branch.
        messages_db: list[ChatMessageTD] = []
        curr_id = current_id

        while curr_id and curr_id in messages_dict:
            msg = messages_dict[curr_id]
            messages_db.insert(0, msg)
            curr_id = msg.get("parentId")

        # 2. Handle the trailing assistant placeholder.
        # OWUI often inserts an empty assistant message entry for the turn currently
        # being processed. We remove it to align with the 'messages_body' which
        # only contains previous turns plus the current user message.
        if messages_db and messages_db[-1].get("role") == "assistant":
            messages_db.pop()

        # 3. Strict validation.
        # If the reconstructed history (minus placeholder) doesn't exactly match the
        # length of the request body (minus system prompt), we bail out.
        # This prevents misaligned metadata mapping.
        if len(messages_db) != len(self.messages_body):
            log.debug(
                f"Strict length mismatch: DB={len(messages_db)}, Body={len(self.messages_body)}. Validation failed."
            )
            return None

        return messages_db

    async def _process_message_turn(
        self, i: int, message: "Message", status_queue: asyncio.Queue
    ) -> list[interaction_types.StepParam] | None:
        """Convert one OWUI message into one or more canonical Interaction steps."""
        role = message.get("role")

        if role == "user":
            message = cast("UserMessage", message)
            contents = await self._process_user_message(i, message, status_queue)
            if not contents:
                log.info(
                    f"User message at index {i} is completely empty. "
                    "Injecting a prompt to ask for clarification."
                )
                toast_msg = (
                    f"Your message #{i + 1} was empty. The assistant will ask for clarification."
                )
                self.event_emitter.emit_toast(toast_msg, "info")
                clarification_prompt = (
                    "The user sent an empty message. Please ask the user for "
                    "clarification on what they would like to ask or discuss."
                )
                contents = await self._interaction_contents_from_text(
                    clarification_prompt, status_queue
                )
            elif self.is_enterprise and not any(
                content.get("type") == "text" for content in contents
            ):
                default_prompt_text = (
                    "The user did not send any text message with the additional context. "
                    "Answer by summarizing the newly added context."
                )
                contents.extend(
                    await self._interaction_contents_from_text(default_prompt_text, status_queue)
                )
                self.event_emitter.emit_toast(
                    f"For your message #{i + 1}, a default prompt was added as text is required "
                    "for requests with attachments when using Gemini Enterprise.",
                    "info",
                )
            return [interaction_types.UserInputStepParam(type="user_input", content=contents)]
        elif role == "assistant":
            message = cast("AssistantMessage", message)
            message_db = self.messages_db[i] if self.messages_db else None
            sources = message_db.get("sources") if message_db else None
            return await self._process_assistant_message(
                i, message, message_db, sources, status_queue
            )

        warn_msg = f"Message {i} has an invalid role: {role}. Skipping to the next message."
        log.warning(warn_msg)
        self.event_emitter.emit_toast(warn_msg, "warning")
        return None

    async def _process_user_message(
        self,
        i: int,
        message: "UserMessage",
        status_queue: asyncio.Queue,
    ) -> list[interaction_types.ContentParam]:
        user_contents: list[interaction_types.ContentParam] = []
        db_files_processed = False

        # PATH 1: Database is available (Normal Chat).
        if self.messages_db:
            message_db = self.messages_db[i]
            files: list[FileAttachmentTD] = message_db.get("files", [])

            if files:
                db_files_processed = True
                upload_tasks = []

                for file in files:
                    content_type = file.get("content_type", "")
                    # MIME types for images always start with 'image/' (e.g., image/png, image/jpeg)
                    is_image = content_type.startswith("image/")

                    # Optimization: Task models (titles, tags, etc.) skip heavy documents
                    # but keep images as they provide high context value for low token cost.
                    should_include = is_image or (self.upload_documents and not self.is_task)

                    if not should_include:
                        log.debug(
                            f"Skipping {content_type} '{file.get('id')}' "
                            f"(is_task={self.is_task}, upload_documents={self.upload_documents})"
                        )
                        continue

                    # We always use the internal API endpoint to fetch file content from the DB.
                    # Even for images, the 'url' field in the attachment object is often
                    # just the UUID, which isn't fetchable on its own.
                    if file_id := file.get("id"):
                        uri = f"/api/v1/files/{file_id}/content"
                        upload_tasks.append(
                            self._interaction_contents_from_uri(
                                uri,
                                status_queue,
                                source_name=file.get("name"),
                            )
                        )
                    else:
                        log.warning("Could not determine ID for a database file entry.")

                if upload_tasks:
                    log.info(f"Processing {len(upload_tasks)} file(s) from database.")
                    results = await asyncio.gather(*upload_tasks)
                    for result in results:
                        user_contents.extend(result)

        # Now, process the content from the message payload.
        user_content = message.get("content")
        if isinstance(user_content, str):
            user_content_list: list[Content] = [{"type": "text", "text": user_content}]
        elif isinstance(user_content, list):
            user_content_list = user_content
        else:
            warn_msg = "User message content is not a string or list, skipping."
            log.warning(warn_msg)
            self.event_emitter.emit_toast(warn_msg, "warning")
            return user_contents

        for c in user_content_list:
            c_type = c.get("type")
            if c_type == "text":
                c = cast("TextContent", c)
                if c_text := c.get("text"):
                    user_contents.extend(
                        await self._interaction_contents_from_text(c_text, status_queue)
                    )

            # PATH 2: Temporary Chat Image Handling.
            # FIXME: this puts images to the end of the message, see if it matters where they are.
            elif c_type == "image_url" and not db_files_processed:
                log.info("Processing image from payload (temporary chat mode).")
                c = cast("ImageContent", c)
                if uri := c.get("image_url", {}).get("url"):
                    user_contents.extend(
                        await self._interaction_contents_from_uri(uri, status_queue)
                    )

        return user_contents

    async def _rehydrate_assistant_steps(
        self,
        stored_steps: list[dict[str, object]],
        stored_endpoint_scope: str | None,
    ) -> list[interaction_types.StepParam]:
        """Validate an exact persisted ledger and restore local generated media."""
        validated = TypeAdapter(list[interaction_types.Step]).validate_python(stored_steps)
        payloads: list[dict[str, object]] = [
            cast(dict[str, object], step.model_dump(mode="json", exclude_none=True))
            for step in validated
        ]
        if self._contains_unknown_variant(payloads):
            raise ValueError("Stored Interaction history contains an unknown step or content type.")

        has_scoped_uri = self._contains_nonlocal_uri(payloads)
        current_scope = self.files_api_manager.endpoint_identity.scope
        if has_scoped_uri and stored_endpoint_scope != current_scope:
            raise ForeignEndpointReplayError(
                "Stored Interaction history contains a URI from a different endpoint identity."
            )

        rehydrated = await self._rehydrate_local_media(payloads)
        return cast(list[interaction_types.StepParam], rehydrated)

    @classmethod
    def _contains_unknown_variant(cls, value: object) -> bool:
        if isinstance(value, dict):
            if value.get("type") == "UNKNOWN" or value.get("is_unknown") is True:
                return True
            return any(cls._contains_unknown_variant(item) for item in value.values())
        if isinstance(value, list):
            return any(cls._contains_unknown_variant(item) for item in value)
        return False

    @classmethod
    def _contains_nonlocal_uri(cls, value: object) -> bool:
        if isinstance(value, dict):
            uri = value.get("uri")
            if isinstance(uri, str) and not uri.startswith("/api/v1/files/"):
                return True
            return any(cls._contains_nonlocal_uri(item) for item in value.values())
        if isinstance(value, list):
            return any(cls._contains_nonlocal_uri(item) for item in value)
        return False

    async def _rehydrate_local_media(self, value: object) -> object:
        if isinstance(value, list):
            return [await self._rehydrate_local_media(item) for item in value]
        if not isinstance(value, dict):
            return value

        restored = dict(value)
        uri = restored.get("uri")
        media_type = restored.get("type")
        if (
            isinstance(uri, str)
            and uri.startswith("/api/v1/files/")
            and media_type in {"image", "audio", "video", "document"}
        ):
            file_id = uri.split("/")[4]
            file_bytes, mime_type = await self._get_file_data(file_id)
            if not file_bytes or not mime_type:
                raise ValueError(
                    f"Could not retrieve content for file_id '{file_id}' from assistant history."
                )
            restored.pop("uri", None)
            restored["data"] = self._encode_data(file_bytes)
            restored["mime_type"] = mime_type

        for key, item in list(restored.items()):
            restored[key] = await self._rehydrate_local_media(item)
        return restored

    def _pop_thoughts(self, content: str) -> tuple[str, list[str]]:
        """
        Identifies and removes thought blocks from the content.

        A thought is defined as text between <think> and </think>\n.
        This method handles multiple thought blocks if they are peppered
        throughout the message.

        :param content: The raw message content from the assistant.
        :return: A tuple containing (cleaned_content, list_of_extracted_thoughts).
        """
        # The pattern looks for the <​think> tag, captures everything inside (non-greedy),
        # and matches the <​/think> tag plus a potential trailing newline.
        # re.DOTALL allows the '.' character to match newlines within the capture group.
        thought_pattern = re.compile(r"<think>(.*?)</think>\n?", re.DOTALL)

        thoughts = thought_pattern.findall(content)
        # Replace all occurrences with an empty string to get the "clean" content.
        cleaned_content = thought_pattern.sub("", content)

        return cleaned_content, thoughts

    async def _process_assistant_message(
        self,
        i: int,
        message_body: "AssistantMessage",
        message_db: "ChatMessageTD | None",
        sources: list["Source"] | None,
        status_queue: asyncio.Queue,
    ) -> list[interaction_types.StepParam]:
        """
        Processes an assistant message, prioritizing the stored Interaction step ledger
        when available and unmodified. Falls back to processing the rendered text if steps
        are missing or if the user has edited the message.
        """
        envelope = self._parse_interaction_envelope(message_db) if message_db else None
        current_content = message_body.get("content", "")

        # 1. Pop thoughts out before any comparison or citation stripping.
        # We store 'thoughts' for future use (e.g., adding to a metadata field).
        current_content, thoughts = self._pop_thoughts(current_content)

        # 2. Strip citations as before.
        if sources:
            current_content = self._remove_citation_markers(current_content, sources)

        # --- PATH 1: Restore the exact canonical ledger when the rendered text is unchanged. ---
        if envelope is not None:
            # Now current_content has no thoughts and no citations,
            # making it directly comparable to original_content.
            if current_content.strip() == envelope.visible_content.strip():
                log.debug(f"Reconstructing assistant message at index {i} from stored parts.")
                try:
                    return await self._rehydrate_assistant_steps(
                        cast(list[dict[str, object]], envelope.steps),
                        envelope.endpoint_scope,
                    )
                except ForeignEndpointReplayError as exc:
                    raise ContentBuildError(str(exc)) from exc
                except (pydantic_core.ValidationError, TypeError, ValueError):
                    log.exception(
                        f"Failed to reconstruct Interaction steps for message {i}. "
                        "Falling back to text processing."
                    )
            else:
                # A meaningful edit was detected after accounting for whitespace.
                diff = difflib.unified_diff(
                    envelope.visible_content.strip().splitlines(keepends=True),
                    current_content.strip().splitlines(keepends=True),
                    fromfile="original_content_stripped",
                    tofile="current_content_stripped",
                )
                diff_str = "".join(diff)

                log.warning(
                    f"An edit was detected in assistant message at index {i}. The message will be "
                    "reconstructed from the current edited text, and the original high-fidelity data "
                    "from the database will be ignored for this turn.\n"
                    f"--- Diff (on stripped content) ---\n{diff_str}"
                )
                self.event_emitter.emit_toast(
                    f"An edit was detected in assistant message #{i + 1}. "
                    "Using the edited text, which may affect model context for this turn.",
                    "warning",
                )
        elif message_db:
            log.warning(
                f"Assistant message at index {i} lacks canonical Interaction history. "
                "This message was likely not generated by this plugin. "
                "Falling back to processing its plain text content."
            )

        # --- PATH 2: Fallback to processing text content ---
        # This path is used for non-Gemini messages, edited messages, or on reconstruction failure.
        log.debug(f"Processing assistant message {i} content as plain text.")
        contents = await self._interaction_contents_from_text(current_content, status_queue)
        return [interaction_types.ModelOutputStepParam(type="model_output", content=contents)]

    async def _interaction_contents_from_text(
        self, text: str, status_queue: asyncio.Queue
    ) -> list[interaction_types.ContentParam]:
        if not text:
            return []

        text = self._enable_special_tags(text)
        contents: list[interaction_types.ContentParam] = []
        last_pos = 0

        # Conditionally build a regex to find media links.
        # If YouTube parsing is disabled, the regex will only find markdown image links,
        # leaving YouTube URLs to be treated as plain text.
        markdown_part = r"!\[.*?\]\(([^)]+)\)"  # Group 1: Markdown URI
        youtube_part = r"(https?://(?:(?:www|music)\.)?youtube\.com/(?:watch\?v=|shorts/|live/)[^\s)]+|https?://youtu\.be/[^\s)]+)"  # Group 2: YouTube URL
        if self.valves.PARSE_YOUTUBE_URLS:
            pattern = re.compile(f"{markdown_part}|{youtube_part}")
            process_youtube = True
        else:
            pattern = re.compile(markdown_part)
            process_youtube = False
            log.info("YouTube URL parsing is disabled. URLs will be treated as plain text.")

        for match in pattern.finditer(text):
            # Add the text segment that precedes the media link
            if text_segment := text[last_pos : match.start()].strip():
                contents.append(self._text_content(text_segment))

            # The URI is in group 1 for markdown, or group 2 for YouTube.
            uri = match.group(1) or match.group(2) if process_youtube else match.group(1)

            if not uri:
                log.warning("Found an unsupported URI format in text; skipping it.")
                continue

            # Delegate all URI processing to the unified helper
            media_contents = await self._interaction_contents_from_uri(uri, status_queue)
            contents.extend(media_contents)

            last_pos = match.end()

        # Add any remaining text after the last media link
        if remaining_text := text[last_pos:].strip():
            contents.append(self._text_content(remaining_text))

        # If no media links were found, the whole text is a single part
        if not contents and text.strip():
            contents.append(self._text_content(text.strip()))

        return contents

    @staticmethod
    def _text_content(text: str) -> interaction_types.TextContentParam:
        return interaction_types.TextContentParam(type="text", text=text)

    async def _interaction_contents_from_uri(
        self,
        uri: str,
        status_queue: asyncio.Queue,
        source_name: str | None = None,
    ) -> list[interaction_types.ContentParam]:
        """Resolve an OWUI/data/YouTube URI into ordered Interaction content."""
        if not uri:
            log.warning("Received an empty URI, skipping.")
            return []

        try:
            file_bytes: bytes | None = None
            file_path: str | None = None
            is_temp_source = False
            mime_type: str | None = None
            owui_file_id: str | None = None

            # Step 1: Extract bytes and mime_type from the URI if applicable
            if uri.startswith("data:"):
                match = re.match(r"data:([^;,]+);base64,(.+)", uri, re.DOTALL)
                if not match:
                    raise ValueError("Invalid base64 data URI.")
                mime_type, base64_data = match.group(1), match.group(2)
                file_bytes = base64.b64decode(base64_data, validate=True)
            elif uri.startswith("/api/v1/files/"):
                log.info("Processing a local API file URI.")
                file_id = uri.split("/")[4]
                owui_file_id = file_id
                source = await self._get_file_source(file_id)
                if source:
                    file_bytes = source.file_bytes
                    file_path = source.file_path
                    mime_type = source.mime_type
                    is_temp_source = source.is_temp
            elif "youtube.com/" in uri or "youtu.be/" in uri:
                log.info("Found a YouTube URL.")
                content = self._interaction_content_from_youtube_uri(uri)
                return [content] if content else []
            else:
                warn_msg = f"Unsupported URI: '{uri[:64]}...' Links must be to YouTube or a supported file type."
                log.warning(warn_msg)
                self.event_emitter.emit_toast(warn_msg, "warning")
                return []

            # Step 2: If we have bytes, create the Part using the modularized helper
            if mime_type and (file_bytes or file_path):
                try:
                    return await self._create_interaction_contents_from_file_source(
                        file_bytes=file_bytes,
                        file_path=file_path,
                        mime_type=mime_type,
                        owui_file_id=owui_file_id,
                        status_queue=status_queue,
                        source_name=source_name,
                    )
                finally:
                    if is_temp_source and file_path:
                        try:
                            os.remove(file_path)
                        except FileNotFoundError:
                            pass
                        except Exception:
                            log.exception(f"Could not remove temporary file {file_path}.")

            return []  # Return empty if bytes/mime_type could not be determined

        except FilesAPIError as e:
            error_msg = f"Files API failed for URI '{uri[:64]}...': {e}"
            log.error(error_msg)
            self.event_emitter.emit_toast(error_msg, "error")
            return []
        except PDFProcessingError as e:
            error_msg = f"PDF processing failed for URI '{uri[:64]}...': {e}"
            log.error(error_msg)
            self.event_emitter.emit_toast(error_msg, "error")
            raise ContentBuildError(error_msg) from e
        except Exception:
            log.exception("Error processing a URI; value omitted from logs.")
            return []

    async def _create_interaction_contents_from_file_data(
        self,
        file_bytes: bytes,
        mime_type: str,
        owui_file_id: str | None,
        status_queue: asyncio.Queue,
        force_raw: bool = False,
        source_name: str | None = None,
    ) -> list[interaction_types.ContentParam]:
        return await self._create_interaction_contents_from_file_source(
            file_bytes=file_bytes,
            file_path=None,
            mime_type=mime_type,
            owui_file_id=owui_file_id,
            status_queue=status_queue,
            force_raw=force_raw,
            source_name=source_name,
        )

    async def _create_interaction_contents_from_file_source(
        self,
        *,
        file_bytes: bytes | None,
        file_path: str | None,
        mime_type: str,
        owui_file_id: str | None,
        status_queue: asyncio.Queue,
        force_raw: bool = False,
        source_name: str | None = None,
    ) -> list[interaction_types.ContentParam]:
        if mime_type == "application/pdf" and self.valves.PDF_LIMIT_MITIGATION and not force_raw:
            outcome = await self.pdf_mitigation_manager.prepare(
                file_bytes=file_bytes,
                file_path=file_path,
            )
            if outcome is None:
                return []

            original_hash = outcome.original_hash
            result = outcome.result
            page_count = result.page_count
            if result.was_mitigated:
                pdf_label = source_name or owui_file_id or "attached PDF"
                contents: list[interaction_types.ContentParam] = []

                if len(result.parts) > 1:
                    contents.append(
                        self._text_content(
                            self._pdf_split_instruction_text(pdf_label, result.parts, page_count)
                        )
                    )
                    self.event_emitter.emit_status(
                        f"Optimized and split PDF into {len(result.parts)} parts.",
                        done=True,
                        indent_level=1,
                    )
                else:
                    self.event_emitter.emit_status(
                        "Optimized PDF to fit Gemini API limits.",
                        done=True,
                        indent_level=1,
                    )

                for i, pdf_part in enumerate(result.parts):
                    synthetic_id = self._synthetic_pdf_part_id(
                        owui_file_id, original_hash, i, len(result.parts)
                    )
                    contents.append(
                        await self._create_interaction_content_from_file_path(
                            file_path=pdf_part.path,
                            mime_type=mime_type,
                            owui_file_id=synthetic_id,
                            status_queue=status_queue,
                            force_raw=force_raw,
                        )
                    )
                return contents

        if file_path:
            return [
                await self._create_interaction_content_from_file_path(
                    file_path=file_path,
                    mime_type=mime_type,
                    owui_file_id=owui_file_id,
                    status_queue=status_queue,
                    force_raw=force_raw,
                )
            ]

        if not file_bytes:
            return []

        return [
            await self._create_interaction_content_from_file_data(
                file_bytes=file_bytes,
                mime_type=mime_type,
                owui_file_id=owui_file_id,
                status_queue=status_queue,
                force_raw=force_raw,
            )
        ]

    @staticmethod
    def _pdf_split_instruction_text(
        pdf_label: str,
        parts: list[PreparedPDFPart],
        page_count: int,
    ) -> str:
        page_map = "\n".join(
            (
                f"- PDF '{pdf_label}', attachment {i}: original document pages "
                f"{part.start_page}-{part.end_page}"
            )
            for i, part in enumerate(parts, start=1)
        )
        return (
            f"PDF '{pdf_label}' was optimized and split into {len(parts)} "
            f"consecutive attachments ({page_count} pages total) to fit Gemini "
            "API limits. Process the PDF parts in order as one original document. "
            "When referencing or citing pages, use the original document page "
            "numbers from this absolute page map; do not restart page numbering "
            "at 1 for each attachment.\n"
            f"{page_map}"
        )

    @staticmethod
    def _synthetic_pdf_part_id(
        owui_file_id: str | None, original_hash: str, index: int, total: int
    ) -> str:
        base_id = owui_file_id or "anonymous"
        suffix = "optimized" if total == 1 else f"part-{index + 1:04d}-of-{total:04d}"
        return f"{base_id}:pdf:{original_hash}:{suffix}"

    async def _create_interaction_content_from_file_data(
        self,
        file_bytes: bytes,
        mime_type: str,
        owui_file_id: str | None,
        status_queue: asyncio.Queue,
        force_raw: bool = False,
    ) -> interaction_types.ContentParam:
        """Create URI-backed or explicitly base64-encoded Interaction content."""
        # TODO: The Files API is strict about MIME types (e.g., text/plain,
        # application/pdf). In the future, inspect the content of files
        # with unsupported text-like MIME types (e.g., 'application/json',
        # 'text/markdown'). If the content is detected as plaintext,
        # override the `mime_type` variable to 'text/plain' to allow the upload.

        # Determine whether to use the Files API based on the specified conditions.
        use_files_api = True
        reason = ""

        if force_raw:
            reason = "raw bytes are forced (e.g. for assistant history reconstruction)"
            use_files_api = False
        elif not self.valves.USE_FILES_API:
            reason = "disabled by user setting (USE_FILES_API=False)"
            use_files_api = False
        elif self.is_enterprise:
            reason = "the active client is configured for Gemini Enterprise, which does not support the Files API"
            use_files_api = False
        elif self.is_temp_chat:
            reason = "temporary chat mode is active"
            use_files_api = False

        if use_files_api:
            log.info("Using Google Files API for resource.")
            gemini_file = await self.files_api_manager.get_or_upload_file(
                file_bytes=file_bytes,
                mime_type=mime_type,
                owui_file_id=owui_file_id,
                status_queue=status_queue,
            )
            if not gemini_file.uri:
                raise FilesAPIError("Uploaded file did not provide a URI.")
            return self._media_content(
                mime_type=gemini_file.mime_type or mime_type,
                uri=gemini_file.uri,
            )
        else:
            log.info(f"Sending raw bytes because {reason}.")
            return self._media_content(
                mime_type=mime_type,
                data=self._encode_data(file_bytes),
            )

    async def _create_interaction_content_from_file_path(
        self,
        file_path: str,
        mime_type: str,
        owui_file_id: str | None,
        status_queue: asyncio.Queue,
        force_raw: bool = False,
    ) -> interaction_types.ContentParam:
        """Create Interaction content from a local path without foreign URI reuse."""
        use_files_api = True
        reason = ""

        if force_raw:
            reason = "raw bytes are forced (e.g. for assistant history reconstruction)"
            use_files_api = False
        elif not self.valves.USE_FILES_API:
            reason = "disabled by user setting (USE_FILES_API=False)"
            use_files_api = False
        elif self.is_enterprise:
            reason = "the active client is configured for Gemini Enterprise, which does not support the Files API"
            use_files_api = False
        elif self.is_temp_chat:
            reason = "temporary chat mode is active"
            use_files_api = False

        if use_files_api:
            log.info("Using Google Files API for local resource.")
            gemini_file = await self.files_api_manager.get_or_upload_file_from_path(
                file_path=file_path,
                mime_type=mime_type,
                owui_file_id=owui_file_id,
                status_queue=status_queue,
            )
            if not gemini_file.uri:
                raise FilesAPIError("Uploaded file did not provide a URI.")
            return self._media_content(
                mime_type=gemini_file.mime_type or mime_type,
                uri=gemini_file.uri,
            )

        log.info(f"Sending raw bytes because {reason}.")
        async with aiofiles.open(file_path, "rb") as file:
            file_bytes = await file.read()
        return self._media_content(
            mime_type=mime_type,
            data=self._encode_data(file_bytes),
        )

    @staticmethod
    def _encode_data(file_bytes: bytes) -> str:
        return base64.b64encode(file_bytes).decode("ascii")

    @staticmethod
    def _media_content(
        *,
        mime_type: str,
        data: str | None = None,
        uri: str | None = None,
    ) -> interaction_types.ContentParam:
        if (data is None) == (uri is None):
            raise ValueError("Interaction media content requires exactly one of data or uri.")
        if mime_type.startswith("image/"):
            image_mime = cast(interaction_types.ImageContentMimeType, mime_type)
            if data is not None:
                return interaction_types.ImageContentParam(
                    type="image", mime_type=image_mime, data=data
                )
            return interaction_types.ImageContentParam(
                type="image", mime_type=image_mime, uri=cast(str, uri)
            )
        if mime_type.startswith("audio/"):
            audio_mime = cast(interaction_types.AudioContentMimeType, mime_type)
            if data is not None:
                return interaction_types.AudioContentParam(
                    type="audio", mime_type=audio_mime, data=data
                )
            return interaction_types.AudioContentParam(
                type="audio", mime_type=audio_mime, uri=cast(str, uri)
            )
        if mime_type.startswith("video/"):
            video_mime = cast(interaction_types.VideoContentMimeType, mime_type)
            if data is not None:
                return interaction_types.VideoContentParam(
                    type="video", mime_type=video_mime, data=data
                )
            return interaction_types.VideoContentParam(
                type="video", mime_type=video_mime, uri=cast(str, uri)
            )
        document_mime = cast(interaction_types.DocumentContentMimeType, mime_type)
        if data is not None:
            return interaction_types.DocumentContentParam(
                type="document", mime_type=document_mime, data=data
            )
        return interaction_types.DocumentContentParam(
            type="document", mime_type=document_mime, uri=cast(str, uri)
        )

    def _interaction_content_from_youtube_uri(
        self, uri: str
    ) -> interaction_types.VideoContentParam | None:
        """Canonicalize a YouTube URL into the fields supported by Interactions."""
        # Convert YouTube Music URLs to standard YouTube URLs for consistent parsing.
        if "music.youtube.com" in uri:
            uri = uri.replace("music.youtube.com", "www.youtube.com")
            log.info("Converted a YouTube Music URL to its standard form.")

        # Regex to capture the 11-character video ID from various YouTube URL formats.
        video_id_pattern = re.compile(
            r"(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|shorts/|live/)|youtu.be/)([a-zA-Z0-9_-]{11})"
        )

        match = video_id_pattern.search(uri)
        if not match:
            log.warning("Could not extract a valid YouTube video ID from the supplied URI.")
            return None

        video_id = match.group(1)
        canonical_uri = f"https://www.youtube.com/watch?v={video_id}"
        parsed = urlparse(uri)
        if parsed.query or parsed.fragment:
            unsupported = {"t", "end", "interval", "fps"}
            query_names = set(parse_qs(parsed.query))
            fragment_names = set(parse_qs(parsed.fragment))
            if unsupported & (query_names | fragment_names):
                self.event_emitter.emit_toast(
                    "YouTube start/end/FPS controls are not supported by the Interactions API; "
                    "the full canonical video URL will be used.",
                    "warning",
                )
        return interaction_types.VideoContentParam(
            type="video", uri=canonical_uri, mime_type="video/mp4"
        )

    @staticmethod
    def _enable_special_tags(text: str) -> str:
        """
        Reverses the action of _disable_special_tags by removing the ZWS
        from special tags. This is used to clean up history messages before
        sending them to the model, so it can understand the context correctly.
        """
        if not text:
            return ""

        # The regex finds '<ZWS' followed by an optional '/' and then one of the special tags.
        # The inner parentheses group the tags, so the optional '/' applies to all of them.
        REVERSE_TAG_REGEX = re.compile(
            r"<"
            + ZWS
            + r"(/?"
            + "("
            + "|".join(re.escape(tag) for tag in SPECIAL_TAGS_TO_DISABLE)
            + ")"
            + r")"
        )
        # The substitution restores the original tag, e.g., '<ZWS/think' becomes '</think'.
        restored_text, count = REVERSE_TAG_REGEX.subn(r"<\1", text)
        if count > 0:
            log.debug(f"Re-enabled {count} special tag(s) for model context.")

        return restored_text

    @staticmethod
    async def _get_file_source(file_id: str) -> LocalFileSource | None:
        """
        Retrieves file metadata and resolves it through Open WebUI storage.

        Open WebUI stores provider-specific paths in the database. Use
        Storage.get_file() just like the Files API content route, so local, S3,
        GCS, and Azure storage all resolve to a readable local path.
        """
        if not file_id:
            log.warning("file_id is empty. Cannot continue.")
            return None

        try:
            file_model = await Files.get_file_by_id(file_id)
        except Exception as e:
            log.exception(
                f"An unexpected error occurred during database call for file_id {file_id}: {e}"
            )
            return None

        if file_model is None:
            log.warning(f"File {file_id} not found in the backend's database.")
            return None

        if not (file_path := file_model.path):
            log.warning(
                f"File {file_id} was found in the database but it lacks `path` field. Cannot Continue."
            )
            return None
        if file_model.meta is None:
            log.warning(
                f"File {file_path} was found in the database but it lacks `meta` field. Cannot continue."
            )
            return None
        if not (content_type := file_model.meta.get("content_type")):
            log.warning(
                f"File {file_path} was found in the database but it lacks `meta.content_type` field. Cannot continue."
            )
            return None

        resolved_path = await GeminiContentBuilder._resolve_owui_storage_path(file_path)
        if resolved_path:
            return LocalFileSource(
                file_bytes=None,
                file_path=resolved_path,
                mime_type=content_type,
            )

        log.warning(f"File {file_path} not found on disk.")
        return LocalFileSource(file_bytes=None, file_path=None, mime_type=content_type)

    @staticmethod
    async def _resolve_owui_storage_path(file_path: str) -> str | None:
        if os.path.exists(file_path):
            return file_path

        try:
            resolved_path = await asyncio.to_thread(Storage.get_file, file_path)
        except Exception:
            log.exception(f"Open WebUI storage failed to resolve file path: {file_path}")
            return None

        if resolved_path and os.path.exists(resolved_path):
            return resolved_path
        return None

    @staticmethod
    async def _get_file_data(file_id: str) -> tuple[bytes | None, str | None]:
        """
        Asynchronously retrieves file metadata from the database and its content.
        """
        # TODO: Emit toasts on unexpected conditions.
        if not file_id:
            log.warning("file_id is empty. Cannot continue.")
            return None, None

        # Await the async database call directly.
        try:
            file_model = await Files.get_file_by_id(file_id)
        except Exception as e:
            log.exception(
                f"An unexpected error occurred during database call for file_id {file_id}: {e}"
            )
            return None, None

        if file_model is None:
            # The get_file_by_id method already handles and logs the specific exception,
            # so we just need to handle the None return value.
            log.warning(f"File {file_id} not found in the backend's database.")
            return None, None

        if not (file_path := file_model.path):
            log.warning(
                f"File {file_id} was found in the database but it lacks `path` field. Cannot Continue."
            )
            return None, None
        if file_model.meta is None:
            log.warning(
                f"File {file_path} was found in the database but it lacks `meta` field. Cannot continue."
            )
            return None, None
        if not (content_type := file_model.meta.get("content_type")):
            log.warning(
                f"File {file_path} was found in the database but it lacks `meta.content_type` field. Cannot continue."
            )
            return None, None

        resolved_path = await GeminiContentBuilder._resolve_owui_storage_path(file_path)
        if not resolved_path:
            log.warning(f"File {file_path} could not be resolved through Open WebUI storage.")
            return None, content_type

        try:
            async with aiofiles.open(resolved_path, "rb") as file:
                file_data = await file.read()
            return file_data, content_type
        except FileNotFoundError:
            log.exception(f"File {resolved_path} not found on disk.")
            return None, content_type
        except Exception:
            log.exception(f"Error processing file {resolved_path}")
            return None, content_type

    @staticmethod
    def _remove_citation_markers(text: str, sources: list["Source"]) -> str:
        # FIXME: this should be moved to `Filter.inlet`
        # FIXME: `text` still contains ZWS here, they need to be removed.
        original_text = text
        processed: set[str] = set()
        for source in sources:
            supports = [
                metadata["supports"]
                for metadata in source.get("metadata", [])
                if "supports" in metadata
            ]
            supports = [item for sublist in supports for item in sublist]
            for support in supports:
                if not isinstance(support, dict):
                    continue
                indices = support.get("grounding_chunk_indices")
                segment = support.get("segment")
                if not isinstance(indices, list) or not isinstance(segment, dict):
                    continue
                segment_text = segment.get("text")
                if not isinstance(segment_text, str) or not segment_text:
                    continue
                # Using a shortened version because user could edit the assistant message in the front-end.
                # If citation segment get's edited, then the markers would not be removed. Shortening reduces the
                # chances of this happening.
                segment_end = segment_text[-32:]
                if segment_end in processed:
                    continue
                processed.add(segment_end)
                citation_markers = "".join(
                    f"[{index + 1}]" for index in indices if isinstance(index, int)
                )
                # Find the position of the citation markers in the text
                pos = text.find(segment_text + citation_markers)
                if pos != -1:
                    # Remove the citation markers
                    text = (
                        text[: pos + len(segment_text)]
                        + text[pos + len(segment_text) + len(citation_markers) :]
                    )
        trim = len(original_text) - len(text)
        log.debug(
            f"Citation removal finished. Returning text str that is {trim} character shorter than the original input."
        )
        return text


class Pipe:
    _cached_client_bindings: list[GenAIClientBinding] = []

    @staticmethod
    def _validate_coordinates_format(v: str | None) -> str | None:
        """Reusable validator for 'latitude,longitude' format."""
        if v is not None and v != "":
            try:
                parts = v.split(",")
                if len(parts) != 2:
                    raise ValueError("Must contain exactly two parts separated by a comma.")

                lat_str, lon_str = parts
                lat = float(lat_str.strip())
                lon = float(lon_str.strip())

                if not (-90 <= lat <= 90):
                    raise ValueError("Latitude must be between -90 and 90.")
                if not (-180 <= lon <= 180):
                    raise ValueError("Longitude must be between -180 and 180.")
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid format for MAPS_GROUNDING_COORDINATES: '{v}'. "
                    f"Expected 'latitude,longitude' (e.g., '40.7128,-74.0060'). Original error: {e}"
                ) from e
        return v

    class Valves(BaseModel):
        # FIXME: docstrings don't get markdown rendered in the admin UI currently. rewrite docstrings accordingly.
        GEMINI_FREE_API_KEY: str | None = Field(
            default=None, description="Free Gemini Developer API key."
        )
        GEMINI_PAID_API_KEY: str | None = Field(
            default=None, description="Paid Gemini Developer API key."
        )
        USER_MUST_PROVIDE_AUTH_CONFIG: bool = Field(
            default=False,
            description="""Whether to require users (including admins) to provide their own authentication configuration.
            User can provide these through UserValves. Setting this to True will disallow users from using Gemini Enterprise.
            Default value is False.""",
        )
        AUTH_WHITELIST: str | None = Field(
            default=None,
            description="""Comma separated list of user emails that are allowed to bypass USER_MUST_PROVIDE_AUTH_CONFIG and use the default authentication configuration.
            Default value is None (no users are whitelisted).""",
        )
        GEMINI_API_BASE_URL: str | None = Field(
            default=None,
            description="""The base URL for calling the Gemini API.
            Default value is None.""",
        )
        # FIXME: assume the user wants Enterprise if they set ENTERPRISE_PROJECT, removing the need for this valve.
        USE_ENTERPRISE: bool = Field(
            default=False,
            description="""Whether to use Google Cloud Gemini Enterprise instead of the standard Gemini API.
            If ENTERPRISE_PROJECT is not set then the plugin will use the Gemini Developer API.
            Default value is False.
            Users can opt out of this by setting USE_ENTERPRISE to False in their UserValves.""",
        )
        ENTERPRISE_PROJECT: str | None = Field(
            default=None,
            description="""The Google Cloud project ID to use with Gemini Enterprise.
            Default value is None.""",
        )
        ENTERPRISE_LOCATION: str = Field(
            default="global",
            description="""The Google Cloud region to use with Gemini Enterprise.
            Default value is 'global'.""",
        )
        ENABLE_FREE_TIER_FALLBACK: bool = Field(
            default=False,
            description="""Automatically switch to the Paid API if a Free API request fails due to quota limits (429) or model overload (503).
            Requires both Free and Paid API keys to be configured. 
            Default value is False.""",
        )
        TASK_MODEL_ROUTING: Literal[
            "only_free",
            "free_fallback",
            "only_paid",
            "match_main",
        ] = Field(
            default="match_main",
            description="""Determines how task models (like title generation) are routed between Free and Paid APIs.
            • only_free: Use only the Free API.
            • free_fallback: Use Free API first, fallback to Paid on failure.
            • only_paid: Bypass Free API and use Paid API directly (or Enterprise if enabled).
            • match_main: Follow the same logic as the main chat generation.
            Default is match_main.""",
        )
        MODEL_WHITELIST: str = Field(
            default="*",
            description="""Comma-separated list of allowed model names.
            Supports `fnmatch` patterns: *, ?, [seq], [!seq].
            Default value is * (all models allowed).""",
        )
        MODEL_BLACKLIST: str | None = Field(
            default=None,
            description="""Comma-separated list of blacklisted model names.
            Supports `fnmatch` patterns: *, ?, [seq], [!seq].
            Default value is None (no blacklist).""",
        )
        CACHE_MODELS: bool = Field(
            default=True,
            description="""Whether to request models only on first load and when white- or blacklist changes.
            Default value is True.""",
        )
        THINKING_LEVEL: Literal["minimal", "low", "medium", "high"] = Field(
            default="high",
            description="Interactions thinking level used when reasoning is enabled.",
        )
        THINKING_SUMMARIES: Literal["auto", "none"] = Field(
            default="auto",
            description="Whether Interactions should return automatic thinking summaries.",
        )
        STORE_INTERACTIONS: bool = Field(
            default=True,
            description="Allow persisted chats to store requests and responses with Google for server-side continuation. Temporary chats and tasks always disable storage. Users may opt out, but cannot override an administrator opt-out.",
        )
        USE_FILES_API: bool = Field(
            default=True,
            description="""Whether to use the Google Files API for uploading files.
            This provides caching and performance benefits, but can be disabled for privacy, cost, or compatibility reasons.
            If disabled, files are sent as raw bytes in the request.
            Default value is True.""",
        )
        PDF_LIMIT_MITIGATION: bool = Field(
            default=True,
            description="""Whether to automatically compress and split PDFs that exceed Gemini's PDF limits.
            Gemini accepts PDFs up to 50 MiB or 1000 pages per document. When enabled, oversized PDFs are optimized
            and split into ordered sub-documents before being sent to Gemini.
            Default value is True.""",
        )
        PARSE_YOUTUBE_URLS: bool = Field(
            default=True,
            description="""Whether to parse YouTube URLs from user messages and provide them as context to the model.
            If disabled, YouTube links are treated as plain text.
            This is only applicable for models that support video.
            Default value is True.""",
        )
        MAPS_GROUNDING_COORDINATES: str | None = Field(
            default=None,
            description="""Optional latitude and longitude coordinates for location-aware results with Google Maps grounding.
            Expected format: 'latitude,longitude' (e.g., '40.7128,-74.0060').
            Default value is None.""",
        )
        LOG_LEVEL: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = (
            Field(
                default="INFO",
                description="""Select logging level. Use `docker logs -f open-webui` to view logs.
            Default value is INFO.""",
            )
        )
        IMAGE_RESOLUTION: Literal["1K", "2K", "4K"] = Field(
            default="1K",
            description="""Resolution for image generation (Gemini 3 Pro Image only).
            Default value is 1K.""",
        )
        IMAGE_ASPECT_RATIO: Literal[
            "1:1",
            "2:3",
            "3:2",
            "3:4",
            "4:3",
            "4:5",
            "5:4",
            "9:16",
            "16:9",
            "21:9",
        ] = Field(
            default="16:9",
            description="""Aspect ratio for image generation (Gemini 3 Pro Image and 2.5 Flash Image).
            Default value is 16:9.""",
        )

        @field_validator("MAPS_GROUNDING_COORDINATES", mode="after")
        @classmethod
        def validate_coordinates_format(cls, v: str | None):
            return Pipe._validate_coordinates_format(v)

    class UserValves(BaseModel):
        """Defines user-specific settings that can override the default `Valves`.

        The `UserValves` class provides a mechanism for individual users to customize
        their Gemini API settings for each request. This system is designed as a
        practical workaround for backend/frontend limitations, enabling per-user
        configurations.

        Think of the main `Valves` as the global, admin-configured template for the
        plugin. `UserValves` acts as a user-provided "overlay" or "patch" that
        is applied on top of that template at runtime.

        How it works:
        1.  **Default Behavior:** At the start of a request, the system merges the
            user's `UserValves` with the admin's `Valves`. If a field in
            `UserValves` has a value (i.e., is not `None` or an empty string `""`),
            it overrides the corresponding value from the main `Valves`. If a
            field is `None` or `""`, the admin's default is used.

        2.  **Special Authentication Logic:** A critical exception exists to enforce
            security and usage policies. If the admin sets `USER_MUST_PROVIDE_AUTH_CONFIG`
            to `True` in the main `Valves`, the merging logic changes for any user
            not on the `AUTH_WHITELIST`:
            - The user's `GEMINI_API_KEY` is taken directly from their `UserValves`,
              bypassing the admin's key entirely.
            - The ability to use the admin-configured Gemini Enterprise is disabled
              (`USE_ENTERPRISE` is forced to `False`).
            This ensures that when required, users must use their own credentials
            and cannot fall back on the shared, system-level authentication.

        This two-tiered configuration allows administrators to set sensible defaults
        and enforce policies, while still giving users the flexibility to tailor
        certain parameters, like their API key or model settings, for their own use.
        """

        # FIXME: `Literal[""]` might not be necessary anymore
        GEMINI_FREE_API_KEY: str | None = Field(
            default=None,
            description="""Free Gemini Developer API key. If not provided, the admin's key may be used if permitted.""",
        )
        GEMINI_PAID_API_KEY: str | None = Field(
            default=None,
            description="""Paid Gemini Developer API key. If not provided, the admin's key may be used if permitted.""",
        )
        GEMINI_API_BASE_URL: str | None = Field(
            default=None,
            description="""The base URL for calling the Gemini API
            Default value is None.""",
        )
        USE_ENTERPRISE: bool | None | Literal[""] = Field(
            default=None,
            description="""Whether to use Google Cloud Gemini Enterprise instead of the standard Gemini API.
            Default value is None.""",
        )
        ENTERPRISE_PROJECT: str | None = Field(
            default=None,
            description="""The Google Cloud project ID to use with Gemini Enterprise.
            Default value is None.""",
        )
        ENTERPRISE_LOCATION: str | None = Field(
            default=None,
            description="""The Google Cloud region to use with Gemini Enterprise.
            Default value is None.""",
        )
        ENABLE_FREE_TIER_FALLBACK: bool | None | Literal[""] = Field(
            default=None,
            description="""Override the default setting for Free API fallback.
            Set to True to enable automatic fallback to the Paid API, False to disable.
            Default is None (use the admin's setting).""",
        )
        TASK_MODEL_ROUTING: (
            Literal["only_free", "free_fallback", "only_paid", "match_main", ""] | None
        ) = Field(
            default=None,
            description="""Override the default routing strategy for task models. 
            Possible values: only_free | free_fallback | only_paid | match_main.""",
        )
        THINKING_LEVEL: Literal["minimal", "low", "medium", "high", ""] | None = Field(
            default=None,
            description="Override the Interactions thinking level.",
        )
        THINKING_SUMMARIES: Literal["auto", "none", ""] | None = Field(
            default=None,
            description="Override automatic thinking summaries.",
        )
        STORE_INTERACTIONS: bool | None | Literal[""] = Field(
            default=None,
            description="Opt out of Google server-side Interaction storage. False forces local full-ledger replay; True is effective only when the administrator also permits storage.",
        )
        USE_FILES_API: bool | None | Literal[""] = Field(
            default=None,
            description="""Override the default setting for using the Google Files API.
            Set to True to force use, False to disable.
            Default is None (use the admin's setting).""",
        )
        PDF_LIMIT_MITIGATION: bool | None | Literal[""] = Field(
            default=None,
            description="""Override automatic PDF compression and splitting for oversized PDFs.
            Default is None (use the admin's setting).""",
        )
        PARSE_YOUTUBE_URLS: bool | None | Literal[""] = Field(
            default=None,
            description="""Override the default setting for parsing YouTube URLs.
            Set to True to enable, False to disable.
            Default is None (use the admin's setting).""",
        )
        MAPS_GROUNDING_COORDINATES: str | None | Literal[""] = Field(
            default=None,
            description="""Optional latitude and longitude coordinates for location-aware results with Google Maps grounding.
            Overrides the admin setting. Expected format: 'latitude,longitude' (e.g., '40.7128,-74.0060').
            Default value is None.""",
        )
        IMAGE_RESOLUTION: Literal["1K", "2K", "4K"] | None | Literal[""] = Field(
            default=None,
            description="""Resolution for image generation (Gemini 3 Pro Image only).
            Default value is None (use the admin's setting). Possible values: 1K, 2K, 4K""",
        )
        IMAGE_ASPECT_RATIO: (
            Literal[
                "1:1",
                "2:3",
                "3:2",
                "3:4",
                "4:3",
                "4:5",
                "5:4",
                "9:16",
                "16:9",
                "21:9",
            ]
            | None
            | Literal[""]
        ) = Field(
            default=None,
            description="""Aspect ratio for image generation (Gemini 3 Pro Image and 2.5 Flash Image).
            Default value is None (use the admin's setting). Possible values: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9""",
        )

        @field_validator("MAPS_GROUNDING_COORDINATES", mode="after")
        @classmethod
        def validate_coordinates_format(cls, v: str | None):
            return Pipe._validate_coordinates_format(v)

    def __init__(self):
        self.valves = self.Valves()
        self.file_content_cache = SimpleMemoryCache(serializer=NullSerializer())
        self.file_id_to_hash_cache = SimpleMemoryCache(serializer=NullSerializer())
        self.pdf_mitigation_manager = PDFMitigationManager()
        self._message_locks: dict[tuple[str, str], MessageGate] = {}
        self._persisted_messages: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._generation_inflight: set[tuple[str, str]] = set()
        self._shutdown_lock = asyncio.Lock()
        self._shutdown_complete = False
        log.success("Function has been initialized.")

    async def pipes(self) -> list["ModelData"]:
        """Register all available Google models."""
        self._add_log_handler(self.valves.LOG_LEVEL)
        log.debug("pipes method has been called.")

        # Clear cache if caching is disabled
        if not self.valves.CACHE_MODELS:
            log.debug("CACHE_MODELS is False, clearing model cache.")
            cache_instance = getattr(self._get_genai_models, "cache", None)
            if isinstance(cache_instance, BaseCache):
                await cache_instance.clear()

        log.info("Fetching and filtering models from Google API.")
        # Get and filter models (potentially cached based on API key, base URL, white- and blacklist)
        try:
            client_args = self._prepare_client_args(self.valves)
            client_args += [self.valves.MODEL_WHITELIST, self.valves.MODEL_BLACKLIST]
            filtered_models = await self._get_genai_models(*client_args)
        except GenaiApiError:
            error_msg = "Error getting the models from Google API, check the logs."
            return [self._return_error_model(error_msg, exception=True)]

        log.info(f"Returning {len(filtered_models)} models to Open WebUI.")
        log.debug("Model list details omitted from logs.")
        log.debug("pipes method has finished.")

        return filtered_models

    async def pipe(
        self,
        body: "Body",
        __user__: "UserData",
        __request__: Request,
        __event_emitter__: Callable[["Event"], Awaitable[None]] | None,
        __metadata__: "Metadata",
        __tools__: dict[str, dict[str, object]] | None = None,
    ) -> AsyncGenerator[dict | str, None] | dict:

        self._add_log_handler(self.valves.LOG_LEVEL)

        log.debug(f"pipe method has been called. Gemini Manifold google_genai version is {VERSION}")
        log.trace("Request metadata received; values omitted from logs.")
        features = __metadata__.get("features", {}) or cast("Features", {})

        # Check the version of the companion filter
        self._check_companion_filter_version(features)

        # Retrieve model configuration from app state
        app_state: State = __request__.app.state
        model_config: dict[str, Any] | None = app_state._state.get("gemini_model_config")
        if model_config is None:
            error_msg = (
                "FATAL: Gemini model configuration not found in app state. "
                "Please ensure the Gemini Manifold Companion filter is installed and enabled."
            )
            raise ValueError(error_msg)
        catalog_version = app_state._state.get("gemini_model_catalog_schema_version")
        if catalog_version != MODEL_CATALOG_SCHEMA_VERSION:
            raise ValueError(
                "FATAL: incompatible Gemini model catalog protocol: "
                f"expected {MODEL_CATALOG_SCHEMA_VERSION}, received {catalog_version!r}. "
                "Update and enable the matching Gemini Manifold Companion filter."
            )

        merged_custom_params = self._resolve_custom_params(body, __metadata__)
        __metadata__["merged_custom_params"] = merged_custom_params

        model_id = self._get_model_name(body)
        __metadata__["canonical_model_id"] = model_id

        catalog_model = model_config.get(model_id)
        if catalog_model is None:
            raise ValueError(
                f"Model '{model_id}' is not present in the audited Gemini Interactions "
                "catalog; use is denied until the catalog is updated."
            )

        # 1. Capture the raw state of keys before any overrides
        valves: Pipe.Valves = self._get_merged_valves(
            self.valves, __user__.get("valves"), __user__.get("email")
        )
        raw_user_valves = __user__.get("valves")
        user_store_preference = (
            raw_user_valves.get("STORE_INTERACTIONS")
            if isinstance(raw_user_valves, dict)
            else getattr(raw_user_valves, "STORE_INTERACTIONS", None)
        )
        __metadata__["gemini_store_interactions"] = self._resolve_store_policy(
            admin_allows=self.valves.STORE_INTERACTIONS,
            user_preference=user_store_preference,
            is_temp=False,
            is_task=bool(__metadata__.get("task")),
        )

        if task_type := __metadata__.get("task"):
            log.info(
                f"{task_type=}, disabling event emissions, YouTube URL parsing and document processing."
            )
            # We disable YouTube parsing for task models to minimize latency and token costs,
            # as simple tasks like title or tag generation do not require video context.
            valves.PARSE_YOUTUBE_URLS = False
            # TODO: disable tools. for now I assume that model will know to not use them even if enabled, but if that's not the case then this needs to be addressed.
            # TODO: use the structured outputs feature to ensure a valid json at all times?

        # 3. Determine Execution Order
        execution_order = await self._determine_execution_order(
            valves=valves,
            __metadata__=__metadata__,
            model_config=model_config,
            features=features,
        )

        log.debug(
            f"Chat ID: {__metadata__.get('chat_id')}, Message ID: {__metadata__.get('message_id')}"
        )

        event_emitter = PipeEventEmitter(__event_emitter__)

        # --- Execution Loop ---
        for attempt_idx, tier in enumerate(execution_order):
            is_last_attempt = attempt_idx == len(execution_order) - 1

            # Create a "Tier-Specific" valves object for this attempt.
            # We copy to ensure we don't pollute the original valves or metadata.
            current_valves = copy.copy(valves)

            if tier == "free":
                current_valves.GEMINI_PAID_API_KEY = None
                current_valves.USE_ENTERPRISE = False
                __metadata__["is_paid_api"] = False
            elif tier == "paid":
                current_valves.GEMINI_FREE_API_KEY = None
                current_valves.USE_ENTERPRISE = False
                __metadata__["is_paid_api"] = True
            elif tier == "enterprise":
                current_valves.USE_ENTERPRISE = True
                __metadata__["is_paid_api"] = True

            service = "enterprise" if current_valves.USE_ENTERPRISE else "developer"
            availability = catalog_model.get("services", {}).get(service, "unsupported")
            if availability != "supported":
                message = (
                    f"Model '{model_id}' is {availability!r} for the Gemini {service} "
                    "Interactions service; this attempt is denied by catalog policy."
                )
                if is_last_attempt:
                    await event_emitter.shutdown()
                    raise ValueError(message)
                log.warning(message)
                continue

            try:
                log.info(f"Starting generation attempt on tier: {tier}")

                # Execute the attempt. This encapsulates client creation,
                # file uploads (scoped to key), and the API call.
                message_key = (
                    str(__metadata__.get("chat_id", "")),
                    str(__metadata__.get("message_id", "")),
                )

                async def execute_attempt(
                    attempt_tier: str = tier,
                    attempt_valves: Pipe.Valves = current_valves,
                ):
                    return await self._execute_generation_attempt(
                        tier=attempt_tier,
                        valves=attempt_valves,
                        body=body,
                        __user__=__user__,
                        __metadata__=__metadata__,
                        __request__=__request__,
                        event_emitter=event_emitter,
                        model_config=model_config,
                        __tools__=__tools__,
                    )

                result = await self._execute_message_once(
                    message_key,
                    execute_attempt,
                )
                return await self._attach_emitter_lifecycle(result, event_emitter)

            except Exception as e:
                failure = self._classify_generation_failure(e)

                should_retry = (
                    not is_last_attempt and tier == "free" and failure.retryable_across_endpoint
                )

                if should_retry:
                    reason = failure.kind.value.replace("_", " ")
                    log.warning(f"Free Tier {reason} (Error: {e}). Switching to Paid API...")
                    event_emitter.emit_status(
                        f"Free Tier {reason}, switching to Paid API...", done=False
                    )
                    continue

                # If we can't retry, re-raise the error to stop execution
                error_msg = f"Gemini request failed before a model response could be generated: {e}"
                log.exception(f"Error during request execution (Tier: {tier}): {e}")
                event_emitter.emit_status("Request failed", done=True)
                event_emitter.emit_toast(error_msg, "error")
                if body.get("stream", False):
                    return await self._attach_emitter_lifecycle(
                        self._error_completion_stream(error_msg), event_emitter
                    )
                return await self._attach_emitter_lifecycle(
                    self._error_completion_response(error_msg), event_emitter
                )

        error_msg = "Exhausted execution options without result."
        if body.get("stream", False):
            return await self._attach_emitter_lifecycle(
                self._error_completion_stream(error_msg), event_emitter
            )
        return await self._attach_emitter_lifecycle(
            self._error_completion_response(error_msg), event_emitter
        )

    # region 2. Helper methods inside the Pipe class

    @asynccontextmanager
    async def _message_guard(self, key: tuple[str, str]) -> AsyncIterator[None]:
        """Serialize one message key and reclaim its gate after all waiters leave."""
        gate = self._message_locks.setdefault(key, MessageGate(lock=asyncio.Lock()))
        gate.users += 1
        try:
            async with gate.lock:
                yield
        finally:
            gate.users -= 1
            if gate.users == 0 and self._message_locks.get(key) is gate:
                self._message_locks.pop(key, None)

    def _prune_completed_messages(self, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        while self._persisted_messages:
            _key, expiry = next(iter(self._persisted_messages.items()))
            if expiry > current:
                break
            self._persisted_messages.popitem(last=False)
        while len(self._persisted_messages) > MESSAGE_COMPLETION_MAX_ENTRIES:
            self._persisted_messages.popitem(last=False)

    def _is_message_completed(self, key: tuple[str, str]) -> bool:
        self._prune_completed_messages()
        expiry = self._persisted_messages.get(key)
        if expiry is None:
            return False
        self._persisted_messages.move_to_end(key)
        return True

    def _remember_completed_message(self, key: tuple[str, str]) -> None:
        self._persisted_messages[key] = time.monotonic() + MESSAGE_COMPLETION_TTL_SECONDS
        self._persisted_messages.move_to_end(key)
        self._prune_completed_messages()

    @staticmethod
    async def _attach_emitter_lifecycle(
        result: AsyncGenerator[dict | str, None] | dict,
        emitter: PipeEventEmitter,
    ) -> AsyncGenerator[dict | str, None] | dict:
        if isinstance(result, dict):
            await emitter.flush()
            await emitter.shutdown()
            return result

        async def guarded() -> AsyncGenerator[dict | str, None]:
            try:
                async for item in result:
                    yield item
            finally:
                await result.aclose()
                await emitter.flush()
                await emitter.shutdown()

        return guarded()

    # region 2.1 Client initialization
    @staticmethod
    @cache
    def _get_or_create_genai_client(
        free_api_key: str | None = None,
        paid_api_key: str | None = None,
        base_url: str | None = None,
        use_enterprise: bool | None = None,
        enterprise_project: str | None = None,
        enterprise_location: str | None = None,
    ) -> GenAIClientBinding:
        """
        Creates a genai.Client instance or retrieves it from cache.
        Raises GenaiApiError on failure.
        """

        # Prioritize the free key, then fall back to the paid key.
        api_key = free_api_key or paid_api_key

        if not enterprise_project and not api_key:
            # FIXME: More detailed reason in the exception (tell user to set the API key).
            msg = "Neither ENTERPRISE_PROJECT nor a Gemini API key (free or paid) is set."
            raise GenaiApiError(msg)

        normalized_base_url = base_url.rstrip("/") if base_url else None
        if use_enterprise and enterprise_project:
            location = enterprise_location or "global"
            kwargs = {
                "enterprise": True,
                "project": enterprise_project,
                "location": location,
                # 2.11's proven Enterprise route remains v1beta1.
                "http_options": types.HttpOptions(
                    api_version="v1beta1", base_url=normalized_base_url
                ),
            }
            api = "Gemini Enterprise"
            identity = EndpointIdentity(
                service="enterprise",
                credential_fingerprint=xxhash.xxh64(
                    f"{enterprise_project}:{location}".encode()
                ).hexdigest(),
                project=enterprise_project,
                location=location,
                base_url=normalized_base_url,
                api_version="v1beta1",
            )
        else:  # Covers (use_enterprise and not enterprise_project) OR (not use_enterprise)
            if use_enterprise and not enterprise_project:
                log.warning(
                    "Gemini Enterprise is enabled but no project is set. Using Gemini Developer API."
                )
            # This also implicitly covers the case where api_key might be None,
            # which is handled by the initial check or the SDK.
            kwargs = {
                "api_key": api_key,
                "http_options": types.HttpOptions(api_version="v1", base_url=normalized_base_url),
            }
            api = "Gemini Developer API"
            assert api_key is not None
            identity = EndpointIdentity(
                service="developer",
                credential_fingerprint=xxhash.xxh64(api_key.encode()).hexdigest(),
                base_url=normalized_base_url,
                api_version="v1",
            )

        try:
            client = genai.Client(**kwargs)
            log.success(f"{api} Genai client successfully initialized.")
            binding = GenAIClientBinding(client=client, identity=identity)
            Pipe._cached_client_bindings.append(binding)
            return binding
        except Exception as e:
            raise GenaiApiError(f"{api} Genai client initialization failed: {e}") from e

    def _get_user_client(self, valves: "Pipe.Valves", user_email: str) -> GenAIClientBinding:
        user_whitelist = valves.AUTH_WHITELIST.split(",") if valves.AUTH_WHITELIST else []
        log.debug(
            f"User whitelist: {user_whitelist}, user email: {user_email}, "
            f"USER_MUST_PROVIDE_AUTH_CONFIG: {valves.USER_MUST_PROVIDE_AUTH_CONFIG}"
        )
        if valves.USER_MUST_PROVIDE_AUTH_CONFIG and user_email not in user_whitelist:
            if not valves.GEMINI_FREE_API_KEY and not valves.GEMINI_PAID_API_KEY:
                error_msg = (
                    "User must provide their own authentication configuration. "
                    "Please set GEMINI_FREE_API_KEY or GEMINI_PAID_API_KEY in your UserValves."
                )
                raise ValueError(error_msg)
        try:
            client_args = self._prepare_client_args(valves)
            binding = self._get_or_create_genai_client(*client_args)
        except GenaiApiError as e:
            error_msg = f"Failed to initialize genai client for user {user_email}: {e}"
            # FIXME: include correct traceback.
            raise ValueError(error_msg) from e
        return binding

    @staticmethod
    def _prepare_client_args(
        source_valves: "Pipe.Valves | Pipe.UserValves",
    ) -> list[str | bool | None]:
        """Prepares arguments for _get_or_create_genai_client from source_valves."""
        ATTRS = [
            "GEMINI_FREE_API_KEY",
            "GEMINI_PAID_API_KEY",
            "GEMINI_API_BASE_URL",
            "USE_ENTERPRISE",
            "ENTERPRISE_PROJECT",
            "ENTERPRISE_LOCATION",
        ]
        return [getattr(source_valves, attr, None) for attr in ATTRS]

    @classmethod
    async def aclose_cached_clients(cls) -> None:
        """Close every owned cached SDK client once and clear the constructor cache."""
        bindings, cls._cached_client_bindings = cls._cached_client_bindings, []
        failures: list[Exception] = []
        for binding in bindings:
            try:
                await binding.aclose()
            except Exception as exc:
                failures.append(exc)
        cls._get_or_create_genai_client.cache_clear()
        if failures:
            raise ExceptionGroup("Failed to close one or more Gemini SDK clients", failures)

    async def shutdown(self) -> None:
        """Release Pipe-owned resources after Open WebUI has quiesced requests.

        Open WebUI does not currently provide a portable function shutdown hook.
        Deployments embedding this Pipe must call ``await pipe.shutdown()`` only
        after request handling has stopped. Repeated calls are safe.
        """
        async with self._shutdown_lock:
            if self._shutdown_complete:
                return
            if self._generation_inflight:
                raise RuntimeError("Cannot shut down Gemini Pipe while requests are active.")
            try:
                await type(self).aclose_cached_clients()
            finally:
                await self.file_content_cache.clear()
                await self.file_id_to_hash_cache.clear()
                await self.pdf_mitigation_manager.cache.clear()
                self.pdf_mitigation_manager.locks.clear()
                model_cache = getattr(self._get_genai_models, "cache", None)
                if isinstance(model_cache, BaseCache):
                    await model_cache.clear()
                self._message_locks.clear()
                self._generation_inflight.clear()
                self._persisted_messages.clear()
                self._shutdown_complete = True

    # endregion 2.1 Client initialization

    # region 2.2 Model retrival from Google API
    @cached()  # aiocache.cached for async method
    async def _get_genai_models(
        self,
        free_api_key: str | None = None,
        paid_api_key: str | None = None,
        base_url: str | None = None,
        use_enterprise: bool | None = None,
        enterprise_project: str | None = None,
        enterprise_location: str | None = None,
        whitelist_str: str = "*",
        blacklist_str: str | None = None,
    ) -> list["ModelData"]:
        """
        Gets valid Google models from API(s) and filters them.
        If use_enterprise, enterprise_project, and api_key are all provided,
        models are fetched from both Gemini Enterprise and Gemini Developer API and merged.
        """
        all_raw_models: list[types.Model] = []

        # Condition for fetching from both sources
        fetch_both = bool(use_enterprise and enterprise_project and (free_api_key or paid_api_key))

        if fetch_both:
            log.info(
                "Attempting to fetch models from both Gemini Developer API and Gemini Enterprise."
            )
            gemini_models_list: list[types.Model] = []
            enterprise_models_list: list[types.Model] = []

            # TODO: perf, consider parallelizing these two fetches
            # 1. Fetch from Gemini Developer API
            try:
                gemini_binding = self._get_or_create_genai_client(
                    free_api_key=free_api_key,
                    paid_api_key=paid_api_key,
                    base_url=base_url,
                    use_enterprise=False,  # Explicitly target Gemini API
                    enterprise_project=None,
                    enterprise_location=None,
                )
                gemini_models_list = await self._fetch_models_from_client_internal(
                    gemini_binding.client, "Gemini Developer API"
                )
            except GenaiApiError as e:
                log.warning(
                    f"Failed to initialize or retrieve models from Gemini Developer API: {e}"
                )
            except Exception as e:
                log.warning(
                    f"An unexpected error occurred with Gemini Developer API models: {e}",
                    exc_info=True,
                )

            # 2. Fetch from Gemini Enterprise
            try:
                enterprise_binding = self._get_or_create_genai_client(
                    use_enterprise=True,  # Explicitly target Gemini Enterprise
                    enterprise_project=enterprise_project,
                    enterprise_location=enterprise_location,
                    base_url=base_url,  # Pass base_url for potential Enterprise custom endpoints
                )
                enterprise_models_list = await self._fetch_models_from_client_internal(
                    enterprise_binding.client, "Gemini Enterprise"
                )
            except GenaiApiError as e:
                log.warning(f"Failed to initialize or retrieve models from Gemini Enterprise: {e}")
            except Exception as e:
                log.warning(
                    f"An unexpected error occurred with Gemini Enterprise models: {e}",
                    exc_info=True,
                )

            # 3. Combine and de-duplicate
            # Prioritize models from Gemini Developer API in case of ID collision
            combined_models_dict: dict[str, types.Model] = {}

            for model in gemini_models_list:
                if model.name:
                    model_id = self._strip_api_prefix(model.name)
                    if model_id and model_id not in combined_models_dict:
                        combined_models_dict[model_id] = model
                else:
                    log.trace(
                        f"Gemini model without a name encountered: {model.display_name or 'N/A'}"
                    )

            for model in enterprise_models_list:
                if model.name:
                    model_id = self._strip_api_prefix(model.name)
                    if model_id:
                        if model_id not in combined_models_dict:
                            combined_models_dict[model_id] = model
                        else:
                            log.info(
                                f"Duplicate model ID '{model_id}' from Gemini Enterprise already sourced from Gemini API. Keeping Gemini API version."
                            )
                else:
                    log.trace(
                        f"Gemini Enterprise model without a name encountered: {model.display_name or 'N/A'}"
                    )

            all_raw_models = list(combined_models_dict.values())

            log.info(
                f"Fetched {len(gemini_models_list)} models from Gemini API, "
                f"{len(enterprise_models_list)} from Gemini Enterprise. "
                f"Combined to {len(all_raw_models)} unique models."
            )

            if not all_raw_models and (gemini_models_list or enterprise_models_list):
                log.warning(
                    "Models were fetched but resulted in an empty list after de-duplication, possibly due to missing names or empty/duplicate IDs."
                )

            if not all_raw_models and not gemini_models_list and not enterprise_models_list:
                raise GenaiApiError(
                    "Failed to retrieve models: Both Gemini Developer API and Gemini Enterprise attempts yielded no models."
                )

        else:  # Single source logic
            # Determine if we are effectively using Gemini Enterprise or Gemini API
            # This depends on user's config (use_enterprise) and availability of project/key
            client_target_is_enterprise = bool(use_enterprise and enterprise_project)
            client_source_name = (
                "Gemini Enterprise" if client_target_is_enterprise else "Gemini Developer API"
            )
            log.info(f"Attempting to fetch models from a single source: {client_source_name}.")

            try:
                binding = self._get_or_create_genai_client(
                    free_api_key=free_api_key,
                    paid_api_key=paid_api_key,
                    base_url=base_url,
                    use_enterprise=client_target_is_enterprise,  # Pass the determined target
                    enterprise_project=enterprise_project if client_target_is_enterprise else None,
                    enterprise_location=(
                        enterprise_location if client_target_is_enterprise else None
                    ),
                )
                all_raw_models = await self._fetch_models_from_client_internal(
                    binding.client, client_source_name
                )

                if not all_raw_models:
                    raise GenaiApiError(
                        f"No models retrieved from {client_source_name}. This could be due to an API error, network issue, or no models being available."
                    )

            except GenaiApiError as e:
                raise GenaiApiError(f"Failed to get models from {client_source_name}: {e}") from e
            except Exception as e:
                log.error(
                    f"An unexpected error occurred while configuring client or fetching models from {client_source_name}: {e}",
                    exc_info=True,
                )
                raise GenaiApiError(
                    f"An unexpected error occurred while retrieving models from {client_source_name}: {e}"
                ) from e

        # --- Common processing for all_raw_models ---

        if not all_raw_models:
            log.warning("No models available after attempting all configured sources.")
            return []

        log.info(f"Processing {len(all_raw_models)} unique raw models.")

        generative_models: list[types.Model] = []
        for model in all_raw_models:
            if model.name is None:
                log.trace(
                    f"Skipping model with no name during generative filter: {model.display_name or 'N/A'}"
                )
                continue
            # Model.supported_actions describes legacy RPCs and does not establish
            # Interactions support. The validated catalog is the capability authority.
            generative_models.append(model)

        if not generative_models:
            log.warning("No generative models found after filtering all retrieved models.")
            return []

        def match_patterns(name_to_check: str, list_of_patterns_str: str | None) -> bool:
            if not list_of_patterns_str:
                return False
            patterns = [
                pat for pat in list_of_patterns_str.replace(" ", "").split(",") if pat
            ]  # Ensure pat is not empty
            return any(fnmatch.fnmatch(name_to_check, pat) for pat in patterns)

        filtered_models_data: list[ModelData] = []
        for model in generative_models:
            # model.name is guaranteed non-None by generative_models filter logic
            assert model.name is not None
            stripped_name = self._strip_api_prefix(model.name)

            if not stripped_name:
                log.warning(
                    f"Model '{model.name}' (display: {model.display_name}) resulted in an empty ID after stripping. Skipping."
                )
                continue

            if stripped_name not in CATALOG_MODEL_IDS:
                log.info(
                    f"Hiding uncatalogued model '{stripped_name}'; Interactions capabilities "
                    "are denied until the versioned catalog is audited and updated."
                )
                continue

            passes_whitelist = not whitelist_str or match_patterns(stripped_name, whitelist_str)
            passes_blacklist = not blacklist_str or not match_patterns(stripped_name, blacklist_str)

            if passes_whitelist and passes_blacklist:
                filtered_models_data.append(
                    {
                        "id": stripped_name,
                        "name": model.display_name or stripped_name,
                        "description": model.description,
                    }
                )
            else:
                log.trace(
                    f"Model ID '{stripped_name}' filtered out by whitelist/blacklist. Whitelist match: {passes_whitelist}, Blacklist pass: {passes_blacklist}"
                )

        log.info(
            f"Filtered {len(generative_models)} generative models down to {len(filtered_models_data)} models based on white/blacklists."
        )
        return filtered_models_data

    # TODO: Use cache for this method too?
    async def _fetch_models_from_client_internal(
        self, client: genai.Client, source_name: str
    ) -> list[types.Model]:
        """Helper to fetch models from a given client and handle common exceptions."""
        try:
            google_models_pager = await client.aio.models.list(
                config={"query_base": True}  # Fetch base models by default
            )
            models = [model async for model in google_models_pager]
            log.info(f"Retrieved {len(models)} models from {source_name}.")
            log.trace(f"Model details returned by {source_name} are omitted from logs.")
            return models
        except Exception as e:
            log.error(f"Retrieving models from {source_name} failed: {e}")
            # Return empty list; caller decides if this is fatal for the whole operation.
            return []

    @staticmethod
    def _return_error_model(
        error_msg: str, warning: bool = False, exception: bool = True
    ) -> "ModelData":
        """Returns a placeholder model for communicating error inside the pipes method to the front-end."""
        if warning:
            log.opt(depth=1, exception=False).warning(error_msg)
        else:
            log.opt(depth=1, exception=exception).error(error_msg)
        return {
            "id": "error",
            "name": "[gemini_manifold] " + error_msg,
            "description": error_msg,
        }

    @staticmethod
    def _error_completion_response(error_msg: str) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": error_msg,
                    }
                }
            ],
            "usage": {},
        }

    @staticmethod
    async def _error_completion_stream(
        error_msg: str,
    ) -> AsyncGenerator[dict | str, None]:
        yield {"choices": [{"delta": {"content": error_msg}}]}
        yield "data: [DONE]"

    @staticmethod
    def _strip_api_prefix(model_name: str) -> str:
        """
        Extract the model identifier by removing API resource prefixes.
        e.g., "models/gemini-1.5-flash-001" -> "gemini-1.5-flash-001"
        e.g., "publishers/google/models/gemini-1.5-pro" -> "gemini-1.5-pro"
        Does NOT handle the manifold pipe prefix (e.g. "gemini_manifold_google_genai.").
        """
        # Remove everything up to the last '/'
        return model_name.split("/")[-1]

    @staticmethod
    def _get_model_name(body: "Body") -> str:
        """
        Extracts the canonical model name from the request body.

        Handles standard model names and custom workspace models by prioritizing
        the base_model_id found in metadata.

        Args:
            body: The request body dictionary.

        Returns:
            The canonical model name (prefix removed).
        """
        # 1. Get the initially requested model name from the top level
        effective_model_name: str = body.get("model", "")
        initial_model_name = effective_model_name
        base_model_name = None

        # 2. Check for a base model ID in the metadata for custom models
        if metadata := body.get("metadata"):
            # Safely navigate the nested structure: metadata -> model -> info -> base_model_id
            base_model_name = metadata.get("model", {}).get("info", {}).get("base_model_id", None)
            # If a base model ID is found, it overrides the initially requested name
            if base_model_name:
                effective_model_name = base_model_name

        # 3. Create the canonical model name by removing the manifold prefix
        canonical_model_name = effective_model_name.replace("gemini_manifold_google_genai.", "")

        # 4. Log the relevant names for debugging purposes
        log.debug(
            f"Model Name Extraction: initial='{initial_model_name}', "
            f"base='{base_model_name}', effective='{effective_model_name}', "
            f"canonical='{canonical_model_name}'"
        )

        # 5. Return only the canonical name
        return canonical_model_name

    @staticmethod
    def _is_image_model(model_id: str, config: dict) -> bool:
        """Return whether a catalogued model may emit images; unknown models deny."""
        outputs = config.get(model_id, {}).get("content", {}).get("outputs", [])
        return "image" in outputs

    # endregion 2.2 Model retrival from Google API

    # region 2.3 Interactions request option assembly

    async def _build_interaction_request_options(
        self,
        body: "Body",
        __metadata__: "Metadata",
        valves: "Valves",
        config: dict,
        custom_functions: list[interaction_types.Function] | None = None,
    ) -> InteractionRequestOptions:
        """Build canonical, capability-checked Interactions create options."""
        model_id = str(__metadata__.get("canonical_model_id", ""))
        model_policy = config.get(model_id)
        if not isinstance(model_policy, dict):
            raise ValueError(f"Model '{model_id}' is absent from the validated catalog.")
        interaction_policy = model_policy.get("interactions", {})
        features = self._resolve_gemini_features(__metadata__)

        if body.get("top_k") is not None:
            raise ValueError("'top_k' is not supported by the Gemini Interactions API.")
        reasoning_effort = (__metadata__.get("merged_custom_params", {}) or {}).get(
            "reasoning_effort"
        )
        if reasoning_effort is not None and not isinstance(reasoning_effort, str):
            raise ValueError("'reasoning_effort' must be minimal, low, medium, or high.")
        thinking_level = reasoning_effort or valves.THINKING_LEVEL
        thinking_policy = interaction_policy.get("thinking", {})
        if features.reasoning:
            if not thinking_policy.get("supported", False):
                raise ValueError(f"Model '{model_id}' does not support thinking.")
            allowed_levels = thinking_policy.get("levels", [])
            if thinking_level not in allowed_levels:
                raise ValueError(
                    f"Thinking level '{thinking_level}' is not supported by model '{model_id}'."
                )

        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        generation_config = interaction_types.GenerationConfig.model_validate(
            {
                "temperature": body.get("temperature"),
                "top_p": body.get("top_p"),
                "max_output_tokens": body.get("max_tokens"),
                "stop_sequences": stop,
                "seed": body.get("seed"),
                "thinking_level": thinking_level if features.reasoning else None,
                "thinking_summaries": (valves.THINKING_SUMMARIES if features.reasoning else "none"),
            }
        )

        safety_settings = __metadata__.get("safety_settings", []) or []
        safety_types = {
            "hate_speech",
            "dangerous_content",
            "harassment",
            "sexually_explicit",
            "civic_integrity",
            "image_hate",
            "image_dangerous_content",
            "image_harassment",
            "image_sexually_explicit",
            "jailbreak",
        }
        safety_thresholds = {
            "block_low_and_above",
            "block_medium_and_above",
            "block_only_high",
            "block_none",
            "off",
        }
        safety_payloads = [
            setting.model_dump(mode="python", exclude_none=True)
            if isinstance(setting, interaction_types.SafetySetting)
            else setting
            for setting in safety_settings
        ]
        for setting in safety_payloads:
            if not isinstance(setting, dict) or setting.get("type") not in safety_types:
                raise ValueError("Safety setting type must use a canonical lowercase literal.")
            if setting.get("threshold") not in safety_thresholds:
                raise ValueError("Safety threshold must use a canonical lowercase literal.")
        canonical_safety = [
            interaction_types.SafetySetting.model_validate(setting) for setting in safety_payloads
        ]

        response_format: interaction_types.CreateModelInteractionResponseFormat | None
        response_format = interaction_types.TextResponseFormat(mime_type="text/plain")
        requested_format = body.get("response_format")
        if requested_format:
            if not interaction_policy.get("response_format", False):
                raise ValueError(f"Model '{model_id}' does not support structured output.")
            if (
                not isinstance(requested_format, dict)
                or requested_format.get("type") != "json_schema"
            ):
                raise ValueError("response_format must use type='json_schema'.")
            json_schema = requested_format.get("json_schema")
            if not isinstance(json_schema, dict) or not isinstance(json_schema.get("schema"), dict):
                raise ValueError("response_format.json_schema.schema must be an object.")
            response_format = interaction_types.TextResponseFormat(
                mime_type="application/json", schema_=json_schema["schema"]
            )
        elif self._is_image_model(model_id, config):
            response_format = interaction_types.ImageResponseFormat(
                image_size=valves.IMAGE_RESOLUTION,
                aspect_ratio=valves.IMAGE_ASPECT_RATIO,
                delivery="inline",
                mime_type="image/jpeg",
            )

        tools: list[interaction_types.Tool] = []
        if not __metadata__.get("task"):
            tool_policy = interaction_policy.get("tools", {})
            requested_tools: list[tuple[str, bool]] = [
                ("google_search", features.google_search),
                ("code_execution", features.code_execution),
                ("url_context", features.url_context),
                ("google_maps", features.google_maps),
            ]
            unsupported = [
                name for name, enabled in requested_tools if enabled and not tool_policy.get(name)
            ]
            if unsupported:
                raise ValueError(
                    f"Model '{model_id}' does not support requested tools: {', '.join(unsupported)}."
                )
            if features.google_search:
                tools.append(interaction_types.GoogleSearch())
            if features.code_execution:
                tools.append(interaction_types.CodeExecution())
            if features.url_context:
                tools.append(interaction_types.URLContext())
            if features.google_maps:
                coordinates: dict[str, float] = {}
                if valves.MAPS_GROUNDING_COORDINATES:
                    latitude, longitude = valves.MAPS_GROUNDING_COORDINATES.split(",")
                    coordinates = {
                        "latitude": float(latitude.strip()),
                        "longitude": float(longitude.strip()),
                    }
                tools.append(
                    interaction_types.GoogleMaps(
                        latitude=coordinates.get("latitude"),
                        longitude=coordinates.get("longitude"),
                    )
                )

        if custom_functions:
            tools.extend(custom_functions)
            generation_config.tool_choice = interaction_types.ToolChoiceConfig(
                allowed_tools=interaction_types.AllowedTools(
                    mode="auto", tools=[tool.name for tool in custom_functions if tool.name]
                )
            )

        return InteractionRequestOptions(
            generation_config=generation_config,
            safety_settings=canonical_safety,
            response_format=response_format,
            tools=tools,
            features=features,
        )

    @staticmethod
    def _resolve_open_webui_tools(
        raw_tools: dict[str, dict[str, object]] | None,
        *,
        model_id: str,
        model_policy: dict[str, object],
        is_task: bool,
    ) -> dict[str, OpenWebUITool]:
        """Validate only executable tools already authorized by Open WebUI."""
        if not raw_tools or is_task:
            return {}
        interactions_policy = model_policy.get("interactions")
        if not isinstance(interactions_policy, dict) or not interactions_policy.get(
            "custom_function_calling", False
        ):
            raise ValueError(f"Model '{model_id}' is not approved for custom function calling.")

        registry: dict[str, OpenWebUITool] = {}
        for exposed_name, raw_tool in raw_tools.items():
            if not isinstance(exposed_name, str) or not GEMINI_TOOL_NAME_PATTERN.fullmatch(
                exposed_name
            ):
                raise ValueError(f"Invalid Open WebUI tool name: {exposed_name!r}.")
            if not isinstance(raw_tool, dict):
                raise ValueError(f"Open WebUI tool '{exposed_name}' has an invalid descriptor.")
            if raw_tool.get("direct") is True:
                raise ValueError(
                    f"Open WebUI direct frontend tool '{exposed_name}' cannot be executed by this pipe."
                )
            function = raw_tool.get("callable")
            if not callable(function):
                raise ValueError(f"Open WebUI tool '{exposed_name}' has no authorized callable.")
            spec = raw_tool.get("spec")
            if not isinstance(spec, dict):
                raise ValueError(f"Open WebUI tool '{exposed_name}' has no valid specification.")
            parameters = spec.get("parameters", {"type": "object", "properties": {}})
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                raise ValueError(
                    f"Open WebUI tool '{exposed_name}' parameters must be an object schema."
                )
            properties = parameters.get("properties", {})
            required = parameters.get("required", [])
            if not isinstance(properties, dict) or not all(
                isinstance(name, str) and isinstance(schema, dict)
                for name, schema in properties.items()
            ):
                raise ValueError(
                    f"Open WebUI tool '{exposed_name}' has invalid parameter properties."
                )
            if not isinstance(required, list) or not all(
                isinstance(name, str) and name in properties for name in required
            ):
                raise ValueError(
                    f"Open WebUI tool '{exposed_name}' has invalid required parameters."
                )
            if any(name.startswith("__") for name in properties):
                raise ValueError(f"Open WebUI tool '{exposed_name}' exposes a reserved parameter.")
            description = spec.get("description")
            declaration = interaction_types.Function(
                name=exposed_name,
                description=description if isinstance(description, str) else exposed_name,
                parameters=copy.deepcopy(parameters),
            )
            registry[exposed_name] = OpenWebUITool(
                name=exposed_name,
                parameters=copy.deepcopy(parameters),
                function=cast(AsyncOpenWebUIToolCallable, function),
                declaration=declaration,
            )
        return registry

    @staticmethod
    def _validate_tool_arguments(tool: OpenWebUITool, arguments: dict[str, object]) -> str | None:
        properties = cast(dict[str, dict[str, object]], tool.parameters.get("properties", {}))
        required = cast(list[str], tool.parameters.get("required", []))
        unexpected = sorted(set(arguments) - set(properties))
        if unexpected:
            return f"Unexpected arguments: {', '.join(unexpected)}."
        missing = sorted(set(required) - set(arguments))
        if missing:
            return f"Missing required arguments: {', '.join(missing)}."
        for name, value in arguments.items():
            schema = properties[name]
            expected = schema.get("type")
            if expected is None:
                valid = True
            elif expected == "string":
                valid = isinstance(value, str)
            elif expected == "boolean":
                valid = isinstance(value, bool)
            elif expected == "integer":
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif expected == "number":
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            elif expected == "array":
                valid = isinstance(value, list)
            elif expected == "object":
                valid = isinstance(value, dict)
            elif expected == "null":
                valid = value is None
            else:
                valid = False
            if not valid:
                return f"Argument '{name}' does not match its declared type '{expected}'."
            enum_values = schema.get("enum")
            if isinstance(enum_values, list) and value not in enum_values:
                return f"Argument '{name}' is not one of its allowed values."
        return None

    @staticmethod
    def _tool_result_text(value: object) -> tuple[str, bool]:
        try:
            text = (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
        except (TypeError, ValueError):
            return "Tool returned a value that is not JSON serializable.", True
        if len(text.encode("utf-8")) > GEMINI_TOOL_MAX_RESULT_BYTES:
            return "Tool result exceeded the 1 MiB response limit.", True
        return text, False

    async def _execute_custom_function_call(
        self,
        call: interaction_types.FunctionCallStep,
        registry: dict[str, OpenWebUITool],
        records: dict[str, ToolCallRecord],
    ) -> interaction_types.FunctionResultStep:
        fingerprint = json.dumps(
            {"name": call.name, "arguments": call.arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prior = records.get(call.id)
        if prior:
            if prior.fingerprint != fingerprint:
                raise InteractionExecutionError(
                    GenerationFailureKind.INTERACTION_ERROR,
                    f"Function call ID '{call.id}' was reused with different arguments.",
                )
            return prior.result

        tool = registry.get(call.name)
        is_error = False
        if tool is None:
            text = f"Function '{call.name}' is not authorized for this request."
            is_error = True
        elif validation_error := self._validate_tool_arguments(
            tool, cast(dict[str, object], call.arguments)
        ):
            text = validation_error
            is_error = True
        else:
            try:
                value = await asyncio.wait_for(
                    tool.function(**cast(dict[str, object], call.arguments)),
                    timeout=GEMINI_TOOL_TIMEOUT_SECONDS,
                )
                text, is_error = self._tool_result_text(value)
            except TimeoutError:
                text = "Tool execution timed out."
                is_error = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                text = f"Tool execution failed ({type(exc).__name__})."
                is_error = True

        result = interaction_types.FunctionResultStep(
            call_id=call.id,
            name=call.name,
            result=text,
            is_error=is_error,
        )
        records[call.id] = ToolCallRecord(fingerprint=fingerprint, result=result)
        return result

    async def _run_custom_function_loop(
        self,
        *,
        interactions: AsyncInteractionsBoundary,
        common_request: dict[str, object],
        registry: dict[str, OpenWebUITool],
        root_replay_input: list[interaction_types.StepParam] | None = None,
    ) -> tuple[list[ReducerEmission], InteractionReduction]:
        """Run sequential custom functions without emitting OpenAI tool-call deltas."""
        if common_request.get("store") is not True:
            raise ValueError("Custom function calling requires store=true continuation.")
        records: dict[str, ToolCallRecord] = {}
        total_calls = 0
        ledger: list[dict[str, JsonValue]] = []
        total_usage = NormalizedInteractionUsage()
        request_payload = dict(common_request)

        for _round_index in range(GEMINI_TOOL_MAX_ROUNDS):
            request = cast(
                interaction_types.CreateModelInteractionParamsNonStreaming,
                {**request_payload, "stream": False},
            )
            try:
                response = await interactions.create(**request)
            except Exception as exc:
                can_replay_root = bool(
                    _round_index == 0
                    and request_payload.get("previous_interaction_id")
                    and root_replay_input
                    and self._is_interaction_not_found(exc)
                )
                if not can_replay_root:
                    raise
                request_payload = {**common_request, "input": root_replay_input}
                request_payload.pop("previous_interaction_id", None)
                request = cast(
                    interaction_types.CreateModelInteractionParamsNonStreaming,
                    {**request_payload, "stream": False},
                )
                response = await interactions.create(**request)
            reducer = InteractionReducer()
            emissions = reducer.consume_interaction(response)
            reducer.finalize_steps()
            ledger.extend(reducer.state.steps)
            for field_name in (
                "input_tokens",
                "output_tokens",
                "thought_tokens",
                "cached_tokens",
                "tool_use_tokens",
                "total_tokens",
            ):
                setattr(
                    total_usage,
                    field_name,
                    getattr(total_usage, field_name) + getattr(reducer.state.usage, field_name),
                )

            calls = [
                step
                for step in (response.steps or [])
                if isinstance(step, interaction_types.FunctionCallStep)
            ]
            if not calls:
                if reducer.state.status != "completed":
                    raise InteractionExecutionError(
                        GenerationFailureKind.INTERACTION_STATUS,
                        f"Interaction ended with status {reducer.state.status} without function calls.",
                    )
                reducer.state.steps = ledger
                reducer.state.usage = total_usage
                return emissions, reducer.state
            if reducer.state.status != "requires_action":
                raise InteractionExecutionError(
                    GenerationFailureKind.INTERACTION_STATUS,
                    f"Interaction returned function calls with status {reducer.state.status}.",
                )
            if not response.id:
                raise InteractionExecutionError(
                    GenerationFailureKind.INTERACTION_ERROR,
                    "Function-calling interaction did not return an interaction ID.",
                )
            if any(not call.id or not call.name for call in calls):
                raise InteractionExecutionError(
                    GenerationFailureKind.INTERACTION_ERROR,
                    "Interaction returned a function call without a non-empty ID and name.",
                )
            if len(calls) > GEMINI_TOOL_MAX_CALLS_PER_ROUND:
                raise InteractionExecutionError(
                    GenerationFailureKind.INTERACTION_ERROR,
                    "Interaction exceeded the per-round function call limit.",
                )
            total_calls += len(calls)
            if total_calls > GEMINI_TOOL_MAX_TOTAL_CALLS:
                raise InteractionExecutionError(
                    GenerationFailureKind.INTERACTION_ERROR,
                    "Interaction exceeded the total function call limit.",
                )
            results = [
                await self._execute_custom_function_call(call, registry, records) for call in calls
            ]
            result_payloads = [
                cast(
                    interaction_types.StepParam,
                    result.model_dump(mode="json", exclude_none=True),
                )
                for result in results
            ]
            ledger.extend(cast(list[dict[str, JsonValue]], copy.deepcopy(result_payloads)))
            request_payload = {
                **common_request,
                "input": result_payloads,
                "previous_interaction_id": response.id,
            }

        raise InteractionExecutionError(
            GenerationFailureKind.INTERACTION_ERROR,
            f"Interaction exceeded the {GEMINI_TOOL_MAX_ROUNDS}-round function loop limit.",
        )

    @staticmethod
    def _is_interaction_not_found(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 404:
            return True
        code = getattr(exc, "code", None)
        return code == 404 or str(code).upper() == "NOT_FOUND"

    @staticmethod
    def _resolve_gemini_features(__metadata__: "Metadata") -> ResolvedGeminiFeatures:
        raw_features = __metadata__.get("features", {}) or {}
        return ResolvedGeminiFeatures(
            reasoning=bool(raw_features.get("reasoning")),
            google_search=bool(raw_features.get("google_search_tool")),
            code_execution=bool(raw_features.get("google_code_execution")),
            url_context=bool(raw_features.get("url_context")),
            google_maps=bool(raw_features.get("google_maps")),
            paid_api=bool(__metadata__.get("is_paid_api")),
            enterprise=bool(__metadata__.get("is_enterprise")),
        )

    # endregion 2.3 Interactions request option assembly

    # region 2.4 Model response processing

    async def _aggregate_to_dict(
        self,
        generator: AsyncGenerator[dict | str, None],
    ) -> dict:
        """
        Consumes the unified response generator and aggregates the chunks into a
        single OpenAI Chat Completion dictionary. This keeps our processing pipeline
        unified while properly satisfying OWUI's non-streaming request expectations.
        """
        content = ""
        reasoning_content = ""
        usage = None

        async for chunk in generator:
            if isinstance(chunk, str):
                # Skip string yields (like "data: [DONE]") used for streaming protocol
                continue

            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta and delta["content"]:
                    content += delta["content"]
                if "reasoning_content" in delta and delta["reasoning_content"]:
                    reasoning_content += delta["reasoning_content"]

            if "usage" in chunk:
                usage = chunk["usage"]

        # Only add the reasoning key if there was actually reasoning content
        message: dict[str, str] = {
            "role": "assistant",
            "content": content,
        }
        if reasoning_content:
            message["reasoning_content"] = reasoning_content

        return {
            "choices": [
                {
                    "message": message,
                }
            ],
            "usage": usage or {},
        }

    async def _execute_message_once(
        self,
        key: tuple[str, str],
        execute: Callable[[], Awaitable[AsyncGenerator[dict | str, None] | dict]],
    ) -> AsyncGenerator[dict | str, None] | dict:
        """Prevent duplicate in-process generation for one OWUI assistant message."""
        if not key[0] or not key[1] or "local" in key[0]:
            return await execute()
        async with self._message_guard(key):
            if key in self._generation_inflight or self._is_message_completed(key):
                raise RuntimeError(
                    "A generation for this chat message is already in progress or completed."
                )
            self._generation_inflight.add(key)
        try:
            result = await execute()
        except BaseException:
            async with self._message_guard(key):
                self._generation_inflight.discard(key)
            raise
        if isinstance(result, dict):
            async with self._message_guard(key):
                self._generation_inflight.discard(key)
            return result

        async def guarded() -> AsyncGenerator[dict | str, None]:
            try:
                async for item in result:
                    yield item
            finally:
                async with self._message_guard(key):
                    self._generation_inflight.discard(key)

        return guarded()

    async def _execute_generation_attempt(
        self,
        tier: str,
        valves: "Pipe.Valves",
        body: "Body",
        __user__: "UserData",
        __metadata__: "Metadata",
        __request__: Request,
        event_emitter: "EventEmitter",
        model_config: dict,
        __tools__: dict[str, dict[str, object]] | None = None,
    ) -> AsyncGenerator[dict | str, None] | dict:
        """
        Executes a single generation attempt with a specific tier configuration.
        Constructs a fresh client and file manager to ensure assets are
        scoped to the correct API key/project.
        """

        # 1. Client Creation
        binding = self._get_user_client(valves, __user__["email"])
        client = binding.client
        model_id = str(__metadata__.get("canonical_model_id", ""))
        endpoint_identity = binding.identity.for_model(model_id)
        metadata_state = cast(dict[str, object], __metadata__)
        metadata_state["is_enterprise"] = endpoint_identity.service == "enterprise"
        metadata_state["gemini_endpoint_scope"] = endpoint_identity.scope
        api_name = (
            "Gemini Enterprise API"
            if endpoint_identity.service == "enterprise"
            else "Gemini Developer API"
        )

        # 2. Files API Manager (Scoped to the current client)
        files_api_manager = FilesAPIManager(
            client=client,
            endpoint_identity=endpoint_identity,
            file_cache=self.file_content_cache,
            id_hash_cache=self.file_id_to_hash_cache,
            event_emitter=event_emitter,
        )

        # 3. Content Builder (Re-uploads files if client changed)
        builder = GeminiContentBuilder(
            messages_body=body.get("messages"),
            metadata_body=__metadata__,
            user_data=__user__,
            event_emitter=event_emitter,
            valves=valves,
            files_api_manager=files_api_manager,
            pdf_mitigation_manager=self.pdf_mitigation_manager,
        )

        event_emitter.emit_status("Preparing request...")
        full_contents = await builder.build_contents()

        # 4. Configuration Building
        model_policy = cast(dict[str, object], model_config.get(model_id, {}))
        tool_registry = self._resolve_open_webui_tools(
            __tools__,
            model_id=model_id,
            model_policy=model_policy,
            is_task=bool(__metadata__.get("task")),
        )
        request_options = await self._build_interaction_request_options(
            body,
            __metadata__,
            valves,
            model_config,
            [tool.declaration for tool in tool_registry.values()],
        )
        request_options.system_instruction = builder.system_prompt

        model_id = __metadata__.get("canonical_model_id", "")
        store_interaction = self._resolve_store_policy(
            admin_allows=bool(__metadata__.get("gemini_store_interactions", True)),
            user_preference=None,
            is_temp=builder.is_temp_chat,
            is_task=bool(__metadata__.get("task")),
        )
        metadata_state["gemini_effective_store"] = store_interaction
        continuation = builder.select_continuation(
            full_contents,
            store=store_interaction,
            endpoint_scope=endpoint_identity.scope,
            model_id=str(model_id),
        )

        # Check for image/system prompt compatibility
        is_image_model = self._is_image_model(model_id, model_config)
        if (is_image_model or "gemma" in model_id) and request_options.system_instruction:
            request_options.system_instruction = None
            log.warning(f"Model '{model_id}' does not support system prompts. Removing.")

        model = cast(interaction_types.Model, model_id)
        common_request = {
            "model": model,
            "input": continuation.input,
            "store": store_interaction,
            "previous_interaction_id": continuation.previous_interaction_id,
            "system_instruction": request_options.system_instruction,
            "generation_config": request_options.generation_config.model_dump(
                mode="json", exclude_none=True
            ),
            "safety_settings": [
                setting.model_dump(mode="json", exclude_none=True)
                for setting in request_options.safety_settings
            ],
            "tools": [
                tool.model_dump(mode="json", exclude_none=True) for tool in request_options.tools
            ],
            "response_format": (
                request_options.response_format.model_dump(mode="json", exclude_none=True)
                if isinstance(request_options.response_format, BaseModel)
                else request_options.response_format
            ),
        }
        common_request = {key: value for key, value in common_request.items() if value is not None}
        log.debug(f"Passing Interaction request to {api_name} (Tier: {tier}).")
        interactions = cast(AsyncInteractionsBoundary, client.aio.interactions)
        is_streaming_request = bool(body.get("stream", True))

        if tool_registry:
            emissions, reduction = await self._run_custom_function_loop(
                interactions=interactions,
                common_request=cast(dict[str, object], common_request),
                registry=tool_registry,
                root_replay_input=full_contents,
            )
            processor = self._present_reduction(
                emissions=emissions,
                reduction=reduction,
                app=__request__.app,
                event_emitter=event_emitter,
                metadata=__metadata__,
            )
            if is_streaming_request:
                return processor
            return await self._aggregate_to_dict(processor)

        if is_streaming_request:
            request = cast(
                interaction_types.CreateModelInteractionParamsStreaming,
                {**common_request, "stream": True},
            )
            try:
                stream = await interactions.create(**request)
            except Exception as exc:
                if not continuation.used_server_state or not self._is_interaction_not_found(exc):
                    raise
                request = cast(
                    interaction_types.CreateModelInteractionParamsStreaming,
                    {
                        **common_request,
                        "input": full_contents,
                        "stream": True,
                    },
                )
                request.pop("previous_interaction_id", None)
                stream = await interactions.create(**request)
            return self._present_interaction_stream(
                stream=stream,
                interactions=interactions,
                app=__request__.app,
                event_emitter=event_emitter,
                metadata=__metadata__,
            )

        request = cast(
            interaction_types.CreateModelInteractionParamsNonStreaming,
            {**common_request, "stream": False},
        )
        try:
            response = await interactions.create(**request)
        except Exception as exc:
            if not continuation.used_server_state or not self._is_interaction_not_found(exc):
                raise
            request = cast(
                interaction_types.CreateModelInteractionParamsNonStreaming,
                {**common_request, "input": full_contents, "stream": False},
            )
            request.pop("previous_interaction_id", None)
            response = await interactions.create(**request)
        reducer = InteractionReducer()
        emissions = reducer.consume_interaction(response)
        reducer.finalize_steps()
        processor = self._present_reduction(
            emissions=emissions,
            reduction=reducer.state,
            app=__request__.app,
            event_emitter=event_emitter,
            metadata=__metadata__,
        )
        return await self._aggregate_to_dict(processor)

    @staticmethod
    def _resolve_store_policy(
        *,
        admin_allows: bool,
        user_preference: object,
        is_temp: bool,
        is_task: bool,
    ) -> bool:
        """Privacy is monotonic: either administrator or user may disable storage."""
        return bool(admin_allows and user_preference is not False and not is_temp and not is_task)

    def _check_free_tier_eligibility(
        self,
        model_id: str,
        model_config: dict,
        features: ResolvedGeminiFeatures,
    ) -> bool:
        """
        Determines if the request is eligible for the Free Tier based on model config
        and requested features (e.g., grounding).
        """
        # 1. Check if model is configured as having a free tier in YAML
        if model_id not in model_config:
            return False

        pricing = model_config[model_id].get("pricing", {})
        if not pricing.get("free_tier", False):
            return False

        # 2. Check for feature exclusions (e.g. Google Search is often Paid only)
        excluded_features = pricing.get("excluded_features", [])

        # Check Search
        is_search_requested = features.google_search
        if is_search_requested and "google_search" in excluded_features:
            log.info(f"Free Tier ineligible: Search requested but excluded for {model_id}.")
            return False

        # Check Maps
        if features.google_maps and "google_maps" in excluded_features:
            log.info(f"Free Tier ineligible: Maps requested but excluded for {model_id}.")
            return False

        return True

    async def _present_interaction_stream(
        self,
        *,
        stream: AsyncInteractionStream[InteractionEvent],
        interactions: AsyncInteractionsBoundary,
        app: FastAPI,
        event_emitter: "EventEmitter",
        metadata: "Metadata",
    ) -> AsyncGenerator[dict | str, None]:
        """Reduce an SSE stream, resuming the same stored Interaction when possible."""
        reducer = InteractionReducer()
        current_stream = stream
        reconnects = 0
        visible_content_parts: list[str] = []
        while True:
            try:
                async with current_stream:
                    async for event in current_stream:
                        for emission in reducer.consume_event(event):
                            chunk = await self._present_emission(emission, app, metadata)
                            if chunk is not None:
                                visible_content_parts.extend(self._content_deltas(chunk))
                                yield chunk
                break
            except asyncio.CancelledError:
                # Closing a foreground SSE stream is local cleanup, not a server
                # cancellation request. Server cancel is reserved for explicitly
                # created background interactions.
                raise
            except InteractionExecutionError:
                raise
            except Exception as exc:
                can_resume = bool(
                    reducer.state.interaction_id and reducer.state.last_event_id and reconnects < 2
                )
                if not can_resume:
                    raise InteractionExecutionError(
                        GenerationFailureKind.INTERACTION_ERROR,
                        f"Interaction stream interrupted and could not resume: {exc}",
                    ) from exc
                reconnects += 1
                log.warning(
                    f"Interaction stream interrupted; resuming attempt {reconnects}/2 "
                    f"from event {reducer.state.last_event_id}."
                )
                current_stream = await interactions.get(
                    cast(str, reducer.state.interaction_id),
                    stream=True,
                    last_event_id=cast(str, reducer.state.last_event_id),
                )

        if not reducer.state.terminal:
            raise InteractionExecutionError(
                GenerationFailureKind.INTERACTION_STATUS,
                "Interaction stream ended without a terminal status.",
            )
        reducer.finalize_steps()
        reducer.state.original_content = "".join(visible_content_parts)
        async for chunk in self._finalize_reduction(
            reducer.state, app.state, event_emitter, metadata
        ):
            yield chunk

    async def _present_reduction(
        self,
        *,
        emissions: list[ReducerEmission],
        reduction: InteractionReduction,
        app: FastAPI,
        event_emitter: "EventEmitter",
        metadata: "Metadata",
    ) -> AsyncGenerator[dict | str, None]:
        visible_content_parts: list[str] = []
        for emission in emissions:
            chunk = await self._present_emission(emission, app, metadata)
            if chunk is not None:
                visible_content_parts.extend(self._content_deltas(chunk))
                yield chunk
        reduction.original_content = "".join(visible_content_parts)
        async for chunk in self._finalize_reduction(reduction, app.state, event_emitter, metadata):
            yield chunk

    @staticmethod
    def _content_deltas(chunk: dict[str, object]) -> list[str]:
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return []
        values: list[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            content = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(content, str):
                values.append(content)
        return values

    async def _present_emission(
        self,
        emission: ReducerEmission,
        app: FastAPI,
        metadata: "Metadata",
    ) -> dict[str, object] | None:
        if emission.kind in {"status", "source", "tool"}:
            return None
        if emission.kind in {"content", "reasoning"}:
            text, _ = self._disable_special_tags(emission.text or "")
            key = "reasoning_content" if emission.kind == "reasoning" else "content"
            return {"choices": [{"delta": {key: text}}]}
        payload = emission.payload or {}
        media_type = str(payload.get("type", ""))
        if media_type != "image":
            uri = payload.get("uri")
            if isinstance(uri, str):
                return {"choices": [{"delta": {"content": f"[{media_type}]({uri})"}}]}
            return None
        mime_type = payload.get("mime_type")
        data = payload.get("data")
        uri = payload.get("uri")
        image_url: str | None = uri if isinstance(uri, str) else None
        if isinstance(data, str) and isinstance(mime_type, str):
            try:
                image_url = await self._upload_image(
                    base64.b64decode(data, validate=True), mime_type, metadata, app
                )
            except ValueError as exc:
                raise InteractionExecutionError(
                    GenerationFailureKind.INTERACTION_ERROR,
                    f"Generated image contained invalid base64 data: {exc}",
                ) from exc
        markdown = (
            f"![Generated Image]({image_url})"
            if image_url
            else "*An error occurred while trying to store this model generated image.*"
        )
        return {"choices": [{"delta": {"content": markdown}}]}

    async def _finalize_reduction(
        self,
        reduction: InteractionReduction,
        app_state: State,
        event_emitter: "EventEmitter",
        metadata: "Metadata",
    ) -> AsyncGenerator[dict | str, None]:
        if reduction.status != "completed":
            raise InteractionExecutionError(
                GenerationFailureKind.INTERACTION_STATUS,
                f"Interaction did not complete successfully: {reduction.status}",
            )
        reduction.grounding = self._prepare_grounding_envelope(
            reduction.grounding, reduction.original_content
        )
        usage = self._get_interaction_usage_data(
            reduction.usage, app_state, metadata, event_emitter.start_time
        )
        yield {"usage": usage}

        chat_id = metadata.get("chat_id")
        message_id = metadata.get("message_id")
        if chat_id and message_id and "local" not in chat_id and not metadata.get("task"):
            key = (chat_id, message_id)
            async with self._message_guard(key):
                if not self._is_message_completed(key):
                    store = bool(metadata.get("gemini_effective_store", False))
                    endpoint_scope = metadata.get("gemini_endpoint_scope")
                    model_id = metadata.get("canonical_model_id")
                    if not isinstance(endpoint_scope, str) or not isinstance(model_id, str):
                        raise InteractionExecutionError(
                            GenerationFailureKind.INTERACTION_ERROR,
                            "Cannot persist Interaction without endpoint scope and model ID.",
                        )
                    envelope = InteractionEnvelopeV1(
                        interaction_id=reduction.interaction_id if store else None,
                        endpoint_scope=endpoint_scope,
                        model_id=model_id,
                        store=store,
                        status=reduction.status,
                        steps=reduction.steps,
                        visible_content=reduction.original_content,
                        usage=reduction.usage,
                        last_event_id=reduction.last_event_id,
                        grounding=reduction.grounding,
                    )
                    try:
                        await Chats.upsert_message_to_chat_by_id_and_message_id(
                            id=chat_id,
                            message_id=message_id,
                            message={
                                "gemini_interaction": envelope.model_dump(
                                    mode="json", exclude_none=False
                                )
                            },
                        )
                        self._remember_completed_message(key)
                    except Exception:
                        log.exception("Failed to persist gemini_interaction envelope.")
        event_emitter.emit_status("Response finished", done=True, is_successful_finish=True)
        yield "data: [DONE]"

    @classmethod
    def _prepare_grounding_envelope(
        cls, envelope: GroundingEnvelope, visible_content: str
    ) -> GroundingEnvelope:
        prepared = envelope.model_copy(deep=True)
        changed_blocks: set[int] = set()
        for block_index, block in enumerate(prepared.text_blocks):
            sanitized, _ = cls._disable_special_tags(block.text)
            if sanitized != block.text:
                changed_blocks.add(block_index)
                prepared.text_blocks[block_index] = block.model_copy(update={"text": sanitized})
        if changed_blocks:
            prepared.citations = [
                citation
                for citation in prepared.citations
                if not (
                    citation.block_index in changed_blocks and citation.index_unit == "provider"
                )
            ]
            prepared.diagnostics.append(
                GroundingDiagnostic(
                    code="sanitized_provider_span",
                    detail="Provider-indexed citations were skipped after visible text sanitization.",
                )
            )
        grounded_text = "".join(block.text for block in prepared.text_blocks)
        prepared.grounded_text_sha256 = hashlib.sha256(grounded_text.encode("utf-8")).hexdigest()
        prepared.visible_content_sha256 = hashlib.sha256(
            visible_content.encode("utf-8")
        ).hexdigest()
        return prepared

    def _get_interaction_usage_data(
        self,
        usage: NormalizedInteractionUsage,
        app_state: State,
        metadata: "Metadata",
        start_time: float,
    ) -> dict[str, object]:
        token_details = usage.model_dump(mode="json")
        cost_details: dict[str, float] = {
            "input_cost": 0.0,
            "cache_cost": 0.0,
            "output_cost": 0.0,
            "image_output_cost": 0.0,
            "total_cost": 0.0,
        }
        if metadata.get("is_paid_api", True):
            model_id = str(metadata.get("canonical_model_id", ""))
            pricing = (
                app_state._state.get("gemini_model_config", {}).get(model_id, {}).get("pricing", {})
            )
            input_tokens = max(usage.input_tokens - usage.cached_tokens, 0)
            image_output_tokens = 0
            for item in usage.output_by_modality:
                modality = item.get("modality")
                tokens = item.get("tokens")
                if (
                    str(modality).lower() == "image"
                    and isinstance(tokens, int)
                    and not isinstance(tokens, bool)
                ):
                    image_output_tokens += tokens
            text_output_tokens = max(usage.output_tokens - image_output_tokens, 0)
            output_tokens = text_output_tokens + usage.thought_tokens
            input_cost = self._calculate_cost(input_tokens, pricing.get("input", []))
            output_cost = self._calculate_cost(output_tokens, pricing.get("output", []))
            image_output_cost = self._calculate_cost(
                image_output_tokens, pricing.get("image_output", [])
            )
            cost_details.update(
                input_cost=round(input_cost, 6),
                output_cost=round(output_cost, 6),
                image_output_cost=round(image_output_cost, 6),
                total_cost=round(input_cost + output_cost + image_output_cost, 6),
            )
        payload: dict[str, object] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens + usage.thought_tokens,
            "token_details": token_details,
            "cost_details": cost_details,
            "completion_time": round(time.monotonic() - start_time, 2),
        }
        previous_tokens = metadata.get("cumulative_tokens")
        previous_cost = metadata.get("cumulative_cost")
        if isinstance(previous_tokens, int) and isinstance(previous_cost, (int, float)):
            payload["cumulative_token_count"] = previous_tokens + usage.total_tokens
            payload["cumulative_total_cost"] = round(
                float(previous_cost) + cost_details["total_cost"], 6
            )
        return payload

    @staticmethod
    def _disable_special_tags(text: str) -> tuple[str, int]:
        if not text:
            return "", 0
        tag_regex = re.compile(
            r"<(/?" + "(" + "|".join(re.escape(tag) for tag in SPECIAL_TAGS_TO_DISABLE) + ")" + r")"
        )
        return tag_regex.subn(rf"<{ZWS}\1", text)

    async def _upload_image(
        self,
        image_data: bytes,
        mime_type: str,
        metadata: "Metadata",
        app: FastAPI,
    ) -> str | None:
        image_format = mimetypes.guess_extension(mime_type) or ".png"
        file_id = str(uuid.uuid4())
        name = f"generated-image{image_format}"
        image_name = f"{file_id}_{name}"
        try:
            contents, image_path = await asyncio.to_thread(
                Storage.upload_file, io.BytesIO(image_data), image_name, tags={}
            )
            file_item = await Files.insert_new_file(
                metadata.get("user_id"),
                FileForm(
                    id=file_id,
                    filename=name,
                    path=image_path,
                    meta={
                        "name": name,
                        "content_type": mime_type,
                        "size": len(contents),
                        "data": {
                            "model": metadata.get("canonical_model_id"),
                            "chat_id": metadata.get("chat_id"),
                            "message_id": metadata.get("message_id"),
                        },
                    },
                ),
            )
        except Exception:
            log.exception("Could not persist a generated Interaction image.")
            return None
        if not file_item:
            return None
        return str(app.url_path_for("get_file_content_by_id", id=file_item.id))

    @staticmethod
    def _calculate_cost(token_count: int, pricing_tiers: list[dict]) -> float:
        """
        Calculates cost based on tiered pricing structure (in USD)
        """
        if not pricing_tiers or token_count <= 0:
            return 0.0

        total_cost = 0.0
        remaining_tokens = token_count

        for tier in pricing_tiers:
            price_per_million = tier.get("price_per_million", 0.0)
            tier_limit = tier.get("up_to_tokens")  # None means unlimited

            if tier_limit is None:
                # Last tier with no limit - use all remaining tokens
                tokens_in_tier = remaining_tokens
            else:
                # Limited tier - use up to the tier limit
                tokens_in_tier = min(remaining_tokens, tier_limit)

            tier_cost = (tokens_in_tier / 1_000_000) * price_per_million
            total_cost += tier_cost
            remaining_tokens -= tokens_in_tier

            if remaining_tokens <= 0:
                break

        return total_cost

    # endregion 2.5 Post-processing

    # region 2.6 Logging
    # TODO: Move to a separate plugin that does not have any Open WebUI funcitonlity and is only imported by this plugin.

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
                normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
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
                redacted[key] = LOG_REDACTED if sensitive else cls._redact_log_data(value)
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
                    data_to_process, serialize_unknown=True, exclude_none=True
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

            except (TypeError, ValueError) as e:  # Catch specific serialization errors
                serialized_data_json = f" - {{Serialization Error: {e}}}"
            except Exception as e:  # Catch any other unexpected errors during processing
                serialized_data_json = f" - {{Processing Error: {e}}}"

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

    @cache  # noqa: B019 -- Pipe has application lifetime; cache prevents duplicate log handlers.
    def _add_log_handler(self, log_level: str):
        """
        Adds or updates the loguru handler specifically for this plugin.
        Includes logic for serializing and truncating extra data.
        The handler is added only if the log_level has changed since the last call.
        """

        def plugin_filter(record: "Record"):
            """Filter function to only allow logs from this plugin (based on module name)."""
            return record["name"] == __name__

        # Get the desired level name and number
        desired_level_name = log_level
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
            log.add(
                sys.stdout,
                level=desired_level_name,
                format=self.plugin_stdout_format,
                filter=plugin_filter,
            )
            log.debug(
                f"Added new handler to loguru for {__name__} with level {desired_level_name}."
            )

    # endregion 2.6 Logging

    # region 2.7 Utility helpers

    @staticmethod
    def _classify_generation_failure(error: Exception) -> GenerationFailure:
        """Classify failures without parsing mutable human-readable messages."""
        if isinstance(error, InteractionExecutionError):
            return GenerationFailure(error.kind, False, str(error))
        if isinstance(error, httpx.TimeoutException | httpx.TransportError):
            return GenerationFailure(GenerationFailureKind.TRANSPORT, True, str(error))
        if isinstance(error, genai_errors.ClientError):
            if error.code == 429:
                return GenerationFailure(GenerationFailureKind.RATE_LIMIT, True, str(error))
            if error.code in {401, 403}:
                return GenerationFailure(GenerationFailureKind.PERMISSION, False, str(error))
            return GenerationFailure(GenerationFailureKind.INVALID_REQUEST, False, str(error))
        if isinstance(error, genai_errors.ServerError):
            return GenerationFailure(
                GenerationFailureKind.UNAVAILABLE,
                error.code in {500, 502, 503, 504},
                str(error),
            )
        return GenerationFailure(GenerationFailureKind.UNKNOWN, False, str(error))

    async def _determine_execution_order(
        self,
        valves: "Pipe.Valves",
        __metadata__: "Metadata",
        model_config: dict[str, Any],
        features: "Features",
    ) -> list[str]:
        """
        Calculates the sequence of execution tiers (free, paid, enterprise).
        Returns an empty list if no valid routing configuration is found.
        """
        model_id = __metadata__.get("canonical_model_id", "")
        has_free_key = bool(valves.GEMINI_FREE_API_KEY)
        has_paid_key = bool(valves.GEMINI_PAID_API_KEY)

        # Retrieve Toggle Statuses
        enterprise_available, enterprise_toggled_on = await self._get_toggleable_feature_status(
            "gemini_enterprise_toggle", __metadata__
        )
        # Enterprise is only viable if toggled on AND we have a project ID
        can_use_enterprise = (
            enterprise_available and enterprise_toggled_on and bool(valves.ENTERPRISE_PROJECT)
        )

        paid_toggle_available, paid_toggled_on = await self._get_toggleable_feature_status(
            "gemini_paid_api", __metadata__
        )

        task_type = __metadata__.get("task")
        routing_strategy = valves.TASK_MODEL_ROUTING if task_type else "match_main"

        is_free_eligible = self._check_free_tier_eligibility(
            model_id,
            model_config,
            self._resolve_gemini_features(__metadata__),
        )

        execution_order: list[str] = []

        match routing_strategy:
            case "only_free":
                log.debug("Task model routing override: only_free")
                if has_free_key:
                    if not is_free_eligible:
                        log.warning(
                            f"Task model '{model_id}' forced to 'only_free' but is ineligible. "
                            "Expect an upstream API error."
                        )
                    execution_order = ["free"]
                else:
                    log.error(
                        "Routing strategy 'only_free' requested, but no Free API Key is configured."
                    )

            case "free_fallback":
                log.debug("Task model routing override: free_fallback")
                # 1. Attempt Free if keys exist and model is eligible
                if has_free_key and is_free_eligible:
                    execution_order.append("free")

                # 2. Add Paid fallbacks
                if can_use_enterprise:
                    execution_order.append("enterprise")
                elif has_paid_key:
                    execution_order.append("paid")

                if not execution_order:
                    log.warning(
                        "Strategy 'free_fallback' could not find any viable tiers (check keys/eligibility)."
                    )

            case "only_paid":
                log.debug("Task model routing override: only_paid")
                if can_use_enterprise:
                    execution_order = ["enterprise"]
                elif has_paid_key:
                    execution_order = ["paid"]
                else:
                    log.error(
                        "Routing strategy 'only_paid' requested, but neither Gemini Enterprise nor Paid API is configured."
                    )

            case _:
                # Default Routing Logic ("match_main")
                if can_use_enterprise:
                    execution_order = ["enterprise"]
                elif paid_toggle_available and paid_toggled_on:
                    if has_paid_key:
                        execution_order = ["paid"]
                    else:
                        log.error("Paid API toggle is ON, but GEMINI_PAID_API_KEY is missing.")
                else:
                    # Logic for standard/un-toggled flow
                    if has_free_key:
                        if is_free_eligible:
                            execution_order = ["free"]
                            if valves.ENABLE_FREE_TIER_FALLBACK:
                                if can_use_enterprise:
                                    execution_order.append("enterprise")
                                elif has_paid_key:
                                    execution_order.append("paid")
                        else:
                            # If model isn't free-eligible, jump straight to paid tiers.
                            # If no paid tier exists, we try free anyway to let the API return the specific error.
                            if can_use_enterprise:
                                execution_order = ["enterprise"]
                            elif has_paid_key:
                                execution_order = ["paid"]
                            else:
                                log.warning(
                                    f"Model '{model_id}' is ineligible for free tier and no paid keys found."
                                )
                    elif can_use_enterprise:
                        execution_order = ["enterprise"]
                    elif has_paid_key:
                        execution_order = ["paid"]

        log.debug(f"Routing strategy for {model_id} ({routing_strategy}): {execution_order}")
        return execution_order

    def _resolve_custom_params(self, body: "Body", __metadata__: "Metadata") -> dict[str, Any]:
        """
        Resolves custom parameters from the model page and chat controls.
        Chat control settings usually take precedence, but we ignore them
        if this is a task model (e.g., generating titles or tags) to ensure
        these independent calls aren't negatively affected by user chat settings.
        """
        known_body_keys = {
            "stream",
            "model",
            "messages",
            "files",
            "options",
            "stream_options",
        }
        merged_params = {key: value for key, value in body.items() if key not in known_body_keys}
        log.debug(f"Extracted {len(merged_params)} model page parameter(s).")

        if __metadata__.get("task"):
            log.debug(
                f"Task model detected (task: {__metadata__.get('task')}). Ignoring chat control parameters."
            )
            return merged_params

        chat_control_params = __metadata__.get("chat_control_params", {})
        if chat_control_params:
            log.debug(f"Received {len(chat_control_params)} chat control parameter(s).")
            merged_params.update(chat_control_params)

        return merged_params

    @staticmethod
    async def _get_toggleable_feature_status(
        filter_id: str,
        __metadata__: "Metadata",
    ) -> tuple[bool, bool]:
        """
        Checks the complete status of a toggleable filter (function).

        This function performs a series of checks to determine if a feature
        is available for use and if the user has activated it.

        1. Checks if the filter is installed.
        2. Checks if the filter's master toggle is active in the Functions dashboard.
        3. Checks if the filter is enabled for the current model (or is global).
        4. Checks if the user has toggled the feature ON for the current request.

        Args:
            filter_id: The ID of the filter to check.
            __metadata__: The metadata object for the current request.

        Returns:
            A tuple (is_available: bool, is_toggled_on: bool).
            - is_available: True if the filter is installed, active, and configured for the model.
            - is_toggled_on: True if the user has the toggle ON in the UI for this request.
        """
        # 1. Check if the filter is installed
        f = await Functions.get_function_by_id(filter_id)
        if not f:
            log.warning(
                f"The '{filter_id}' filter is not installed. "
                "Install it to use the corresponding front-end toggle."
            )
            return (False, False)

        # 2. Check if the master toggle is active
        if not f.is_active:
            log.warning(
                f"The '{filter_id}' filter is installed but is currently disabled in the "
                "Functions dashboard (master toggle is off). Enable it to make it available."
            )
            return (False, False)

        # 3. Check if the filter is enabled for the model or is global
        model_info = __metadata__.get("model", {}).get("info", {})
        model_filter_ids = model_info.get("meta", {}).get("filterIds", [])
        is_enabled_for_model = filter_id in model_filter_ids or f.is_global

        log.debug(
            f"Checking model enablement for '{filter_id}': in_model_filters={filter_id in model_filter_ids}, "
            f"is_global={f.is_global} -> is_enabled={is_enabled_for_model}"
        )

        if not is_enabled_for_model:
            # This is a configuration issue, not a user-facing warning. Debug is appropriate.
            model_id = __metadata__.get("model", {}).get("id", "Unknown")
            log.debug(
                f"Filter '{filter_id}' is not enabled for model '{model_id}' and is not global."
            )
            return (False, False)

        # 4. Check if the user has toggled the feature ON for this request
        user_toggled_ids = __metadata__.get("filter_ids", [])
        is_toggled_on = filter_id in user_toggled_ids

        if is_toggled_on:
            log.info(
                f"Feature '{filter_id}' is available and enabled by the front-end toggle for this request."
            )
        else:
            log.debug(
                f"Feature '{filter_id}' is available but not enabled by the front-end toggle for this request."
            )

        return (True, is_toggled_on)

    @staticmethod
    def _get_merged_valves(
        default_valves: "Pipe.Valves",
        user_valves: "Pipe.UserValves | None",
        user_email: str,
    ) -> "Pipe.Valves":
        """
        Merges UserValves into a base Valves configuration.

        The general rule is that if a field in UserValves is not None or an empty
        string, it overrides the corresponding field in the default_valves.
        Otherwise, the default_valves field value is used.

        Exceptions:
        - If default_valves.USER_MUST_PROVIDE_AUTH_CONFIG is True and the user is
          not on the AUTH_WHITELIST, then GEMINI_FREE_API_KEY and
          GEMINI_PAID_API_KEY in the merged result will be taken directly from
          user_valves (even if they are None), and Gemini Enterprise usage is disabled.

        Args:
            default_valves: The base Valves object with default configurations.
            user_valves: An optional UserValves object with user-specific overrides.
                         If None, a copy of default_valves is returned.

        Returns:
            A new Valves object representing the merged configuration.
        """
        if user_valves is None:
            # If no user-specific valves are provided, return a copy of the default valves.
            return default_valves.model_copy(deep=True)

        # Start with the values from the base `Valves`
        merged_data = default_valves.model_dump()

        # Override with non-None values from `UserValves`
        # Iterate over fields defined in the UserValves model
        for field_name in Pipe.UserValves.model_fields:
            # getattr is safe as field_name comes from model_fields of user_valves' type
            user_value = getattr(user_valves, field_name)
            if user_value is not None and user_value != "":
                # Only update if the field is also part of the main Valves model
                # (keys of merged_data are fields of default_valves)
                if field_name in merged_data:
                    merged_data[field_name] = user_value

        user_whitelist = (
            default_valves.AUTH_WHITELIST.split(",") if default_valves.AUTH_WHITELIST else []
        )

        # Apply special logic based on default_valves.USER_MUST_PROVIDE_AUTH_CONFIG
        if default_valves.USER_MUST_PROVIDE_AUTH_CONFIG and user_email not in user_whitelist:
            log.info(
                f"User '{user_email}' is required to provide their own authentication credentials due to USER_MUST_PROVIDE_AUTH_CONFIG=True."
                " Admin-provided API keys and Gemini Enterprise settings will not be used."
            )
            # If USER_MUST_PROVIDE_AUTH_CONFIG is True and user is not in the whitelist,
            # they must provide their own API keys.
            # They are also disallowed from using the admin's Gemini Enterprise configuration.
            merged_data["GEMINI_FREE_API_KEY"] = user_valves.GEMINI_FREE_API_KEY
            merged_data["GEMINI_PAID_API_KEY"] = user_valves.GEMINI_PAID_API_KEY
            merged_data["ENTERPRISE_PROJECT"] = None
            merged_data["USE_ENTERPRISE"] = False

        # Create a new Valves instance with the merged data.
        # Pydantic will validate the data against the Valves model definition during instantiation.
        return Pipe.Valves(**merged_data)

    def _check_companion_filter_version(self, features: "Features | dict") -> None:
        """Require the companion that implements this breaking protocol pair."""
        companion_version = features.get("gemini_manifold_companion_version")
        if companion_version != RECOMMENDED_COMPANION_VERSION:
            received = companion_version if companion_version is not None else "not detected"
            raise ValueError(
                "Gemini Manifold protocol mismatch: companion "
                f"{RECOMMENDED_COMPANION_VERSION} is required, received {received}."
            )
        log.debug(f"Gemini Manifold Companion protocol pair: {companion_version}")

    # endregion 2.7 Utility helpers

    # endregion 2. Helper methods inside the Pipe class
