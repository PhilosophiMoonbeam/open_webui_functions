"""Sanitized, JSON-first google-genai 2.11 Interactions fixture corpora."""

from collections.abc import Mapping
from typing import TypeVar, cast

from google.genai import interactions
from pydantic import JsonValue, TypeAdapter

DEFAULT_MODEL = cast(interactions.Model, "gemini-2.5-flash")
JsonObject = dict[str, JsonValue]
T = TypeVar("T")

EVENT_ADAPTER = TypeAdapter(interactions.InteractionSSEEvent)
STEP_ADAPTER = TypeAdapter(interactions.Step)
CONTENT_ADAPTER = TypeAdapter(interactions.Content)
DELTA_ADAPTER = TypeAdapter(interactions.StepDeltaData)
TOOL_ADAPTER = TypeAdapter(interactions.Tool)
ANNOTATION_ADAPTER = TypeAdapter(interactions.Annotation)


def validate_corpus(adapter: TypeAdapter[T], corpus: Mapping[str, JsonObject]) -> dict[str, T]:
    """Validate JSON-compatible payloads solely through a public SDK union."""
    return {name: adapter.validate_python(payload) for name, payload in corpus.items()}


ANNOTATION_PAYLOADS: dict[str, JsonObject] = {
    "url_citation": {
        "type": "url_citation",
        "url": "https://example.test/source",
        "title": "Public source",
        "start_index": 0,
        "end_index": 5,
    },
    "file_citation": {
        "type": "file_citation",
        "document_uri": "gs://sanitized-bucket/document.pdf",
        "file_name": "document.pdf",
        "page_number": 2,
        "start_index": 6,
        "end_index": 10,
    },
    "place_citation": {
        "type": "place_citation",
        "name": "Example Cafe",
        "place_id": "place-sanitized",
        "url": "https://maps.example.test/place",
        "start_index": 11,
        "end_index": 15,
    },
    "unknown": {"type": "future_annotation", "payload": "preserved"},
}

CONTENT_PAYLOADS: dict[str, JsonObject] = {
    "text": {
        "type": "text",
        "text": "Hello",
        "annotations": [ANNOTATION_PAYLOADS["url_citation"]],
    },
    "image": {
        "type": "image",
        "data": "aW1hZ2U=",
        "mime_type": "image/png",
        "resolution": "1x1",
    },
    "audio": {
        "type": "audio",
        "data": "YXVkaW8=",
        "mime_type": "audio/wav",
        "channels": 1,
        "sample_rate": 16000,
    },
    "document": {
        "type": "document",
        "uri": "gs://sanitized-bucket/document.pdf",
        "mime_type": "application/pdf",
    },
    "video": {
        "type": "video",
        "uri": "gs://sanitized-bucket/video.mp4",
        "mime_type": "video/mp4",
        "resolution": "720p",
    },
    "unknown": {"type": "future_content", "payload": "preserved"},
}

STEP_PAYLOADS: dict[str, JsonObject] = {
    "user_input": {"type": "user_input", "content": [CONTENT_PAYLOADS["text"]]},
    "model_output": {"type": "model_output", "content": [CONTENT_PAYLOADS["text"]]},
    "thought": {
        "type": "thought",
        "signature": "c2FuaXRpemVkLXNpZ25hdHVyZQ==",
        "summary": [{"type": "text", "text": "Brief rationale"}],
    },
    "function_call": {
        "type": "function_call",
        "id": "call-function",
        "name": "lookup_weather",
        "arguments": {"city": "Montreal"},
    },
    "code_execution_call": {
        "type": "code_execution_call",
        "id": "call-code",
        "arguments": {"language": "python", "code": "print(2 + 2)"},
        "signature": "c2ln",
    },
    "url_context_call": {
        "type": "url_context_call",
        "id": "call-url",
        "arguments": {"urls": ["https://example.test"]},
        "signature": "c2ln",
    },
    "mcp_server_tool_call": {
        "type": "mcp_server_tool_call",
        "id": "call-mcp",
        "name": "lookup",
        "server_name": "sanitized-server",
        "arguments": {"query": "safe"},
    },
    "google_search_call": {
        "type": "google_search_call",
        "id": "call-search",
        "arguments": {"queries": ["public query"]},
        "search_type": "web_search",
        "signature": "c2ln",
    },
    "file_search_call": {
        "type": "file_search_call",
        "id": "call-file",
        "signature": "c2ln",
    },
    "google_maps_call": {
        "type": "google_maps_call",
        "id": "call-maps",
        "arguments": {"queries": ["coffee nearby"]},
        "signature": "c2ln",
    },
    "function_result": {
        "type": "function_result",
        "call_id": "call-function",
        "name": "lookup_weather",
        "result": {"temperature_c": 20},
        "is_error": False,
    },
    "code_execution_result": {
        "type": "code_execution_result",
        "call_id": "call-code",
        "result": "4\n",
        "is_error": False,
        "signature": "c2ln",
    },
    "url_context_result": {
        "type": "url_context_result",
        "call_id": "call-url",
        "result": [{"url": "https://example.test", "status": "success"}],
        "is_error": False,
        "signature": "c2ln",
    },
    "google_search_result": {
        "type": "google_search_result",
        "call_id": "call-search",
        "result": [{"search_suggestions": "refined public query"}],
        "is_error": False,
        "signature": "c2ln",
    },
    "mcp_server_tool_result": {
        "type": "mcp_server_tool_result",
        "call_id": "call-mcp",
        "name": "lookup",
        "server_name": "sanitized-server",
        "result": {"answer": "safe"},
    },
    "file_search_result": {
        "type": "file_search_result",
        "call_id": "call-file",
        "signature": "c2ln",
    },
    "google_maps_result": {
        "type": "google_maps_result",
        "call_id": "call-maps",
        "result": [
            {
                "widget_context_token": "sanitized-widget-token",
                "places": [
                    {
                        "name": "Example Cafe",
                        "place_id": "place-sanitized",
                        "url": "https://maps.example.test/place",
                    }
                ],
            }
        ],
        "signature": "c2ln",
    },
    "unknown": {"type": "future_step", "payload": "preserved"},
}

DELTA_PAYLOADS: dict[str, JsonObject] = {
    "text": {"type": "text", "text": "partial text"},
    "image": {"type": "image", "data": "aW1h", "mime_type": "image/png"},
    "audio": {"type": "audio", "data": "YXVkaW8=", "mime_type": "audio/wav"},
    "document": {"type": "document", "data": "ZG9j", "mime_type": "application/pdf"},
    "video": {"type": "video", "data": "dmlkZW8=", "mime_type": "video/mp4"},
    "thought_summary": {
        "type": "thought_summary",
        "content": {"type": "text", "text": "partial rationale"},
    },
    "thought_signature": {"type": "thought_signature", "signature": "c2ln"},
    "text_annotation_delta": {
        "type": "text_annotation_delta",
        "annotations": [ANNOTATION_PAYLOADS["url_citation"]],
    },
    "arguments_delta": {"type": "arguments_delta", "arguments": '{"city":'},
    "code_execution_call": {
        "type": "code_execution_call",
        "arguments": {"language": "python", "code": "print(1)"},
        "signature": "c2ln",
    },
    "url_context_call": {
        "type": "url_context_call",
        "arguments": {"urls": ["https://example.test"]},
        "signature": "c2ln",
    },
    "google_search_call": {
        "type": "google_search_call",
        "arguments": {"queries": ["public query"]},
        "signature": "c2ln",
    },
    "mcp_server_tool_call": {
        "type": "mcp_server_tool_call",
        "name": "lookup",
        "server_name": "sanitized-server",
        "arguments": {"query": "safe"},
    },
    "file_search_call": {"type": "file_search_call", "signature": "c2ln"},
    "google_maps_call": {
        "type": "google_maps_call",
        "arguments": {"queries": ["coffee nearby"]},
        "signature": "c2ln",
    },
    "retrieval_call": {
        "type": "retrieval_call",
        "arguments": {"queries": ["private corpus query"]},
        "retrieval_type": "vertex_ai_search",
        "signature": "c2ln",
    },
    "code_execution_result": {
        "type": "code_execution_result",
        "result": "1\n",
        "is_error": False,
        "signature": "c2ln",
    },
    "url_context_result": {
        "type": "url_context_result",
        "result": [{"url": "https://example.test", "status": "success"}],
        "is_error": False,
        "signature": "c2ln",
    },
    "google_search_result": {
        "type": "google_search_result",
        "result": [{"search_suggestions": "refined query"}],
        "is_error": False,
        "signature": "c2ln",
    },
    "mcp_server_tool_result": {
        "type": "mcp_server_tool_result",
        "name": "lookup",
        "server_name": "sanitized-server",
        "result": {"answer": "safe"},
    },
    "file_search_result": {
        "type": "file_search_result",
        "result": [{}],
        "signature": "c2ln",
    },
    "google_maps_result": {
        "type": "google_maps_result",
        "result": [{"widget_context_token": "sanitized-widget-token", "places": []}],
        "signature": "c2ln",
    },
    "retrieval_result": {
        "type": "retrieval_result",
        "is_error": False,
        "signature": "c2ln",
    },
    "function_result": {
        "type": "function_result",
        "call_id": "call-function",
        "name": "lookup_weather",
        "result": {"temperature_c": 20},
        "is_error": False,
    },
    "unknown": {"type": "future_delta", "payload": "preserved"},
}

TOOL_PAYLOADS: dict[str, JsonObject] = {
    "function": {
        "type": "function",
        "name": "lookup_weather",
        "description": "Return public weather data.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    "code_execution": {"type": "code_execution"},
    "url_context": {"type": "url_context"},
    "computer_use": {
        "type": "computer_use",
        "environment": "browser",
        "enable_prompt_injection_detection": True,
    },
    "mcp_server": {
        "type": "mcp_server",
        "name": "sanitized-server",
        "url": "https://mcp.example.test",
        "allowed_tools": [{"name": "lookup"}],
    },
    "google_search": {"type": "google_search", "search_types": ["web_search"]},
    "file_search": {
        "type": "file_search",
        "file_search_store_names": ["fileSearchStores/sanitized"],
        "top_k": 5,
    },
    "google_maps": {
        "type": "google_maps",
        "latitude": 45.5,
        "longitude": -73.6,
        "enable_widget": True,
    },
    "retrieval": {
        "type": "retrieval",
        "retrieval_types": ["vertex_ai_search"],
        "vertex_ai_search_config": {"datastores": ["projects/p/locations/l/dataStores/d"]},
    },
    "unknown": {"type": "future_tool", "payload": "preserved"},
}

USAGE_PAYLOAD: JsonObject = {
    "cached_tokens_by_modality": [{"modality": "text", "tokens": 2}],
    "grounding_tool_count": [{"type": "google_search", "count": 1}],
    "input_tokens_by_modality": [{"modality": "text", "tokens": 5}],
    "output_tokens_by_modality": [{"modality": "image", "tokens": 3}],
    "tool_use_tokens_by_modality": [{"modality": "text", "tokens": 4}],
    "total_cached_tokens": 2,
    "total_input_tokens": 5,
    "total_output_tokens": 7,
    "total_thought_tokens": 2,
    "total_tool_use_tokens": 4,
    "total_tokens": 20,
}

INTERACTION_STATUSES = (
    "in_progress",
    "requires_action",
    "completed",
    "failed",
    "cancelled",
    "incomplete",
    "budget_exceeded",
)

ERROR_PAYLOADS: dict[str, JsonObject] = {
    "stream": {
        "event_type": "error",
        "event_id": "event-error",
        "error": {"code": "INTERNAL", "message": "sanitized request failure"},
    },
    "model_output": {
        "type": "model_output",
        "content": [],
        "error": {
            "code": 13,
            "message": "sanitized model failure",
            "details": [{"reason": "fixture"}],
        },
    },
}

EVENT_PAYLOADS: dict[str, JsonObject] = {
    "interaction.created": {
        "event_type": "interaction.created",
        "event_id": "event-created",
        "interaction": {
            "id": "interaction-corpus",
            "model": "gemini-2.5-flash",
            "status": "in_progress",
        },
    },
    "interaction.completed": {
        "event_type": "interaction.completed",
        "event_id": "event-completed",
        "interaction": {
            "id": "interaction-corpus",
            "model": "gemini-2.5-flash",
            "status": "completed",
            "steps": [STEP_PAYLOADS["model_output"]],
            "usage": USAGE_PAYLOAD,
        },
    },
    "interaction.status_update": {
        "event_type": "interaction.status_update",
        "event_id": "event-status",
        "interaction_id": "interaction-corpus",
        "status": "requires_action",
    },
    "error": ERROR_PAYLOADS["stream"],
    "step.start": {
        "event_type": "step.start",
        "event_id": "event-step-start",
        "index": 0,
        "step": STEP_PAYLOADS["model_output"],
    },
    "step.delta": {
        "event_type": "step.delta",
        "event_id": "event-step-delta",
        "index": 0,
        "delta": DELTA_PAYLOADS["text"],
    },
    "step.stop": {
        "event_type": "step.stop",
        "event_id": "event-step-stop",
        "index": 0,
        "usage": USAGE_PAYLOAD,
    },
    "unknown": {"event_type": "interaction.future", "payload": "preserved"},
}


def completed_interaction(
    text: str = "Hello from Gemini.",
    *,
    interaction_id: str = "interaction-test-1",
    model: interactions.Model = DEFAULT_MODEL,
) -> interactions.Interaction:
    return interactions.Interaction.model_validate(
        {
            "id": interaction_id,
            "model": model,
            "status": "completed",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
            "usage": {
                "total_input_tokens": 3,
                "total_output_tokens": 4,
                "total_tokens": 7,
            },
        }
    )


def completed_stream(
    text: str = "Hello from Gemini.",
    *,
    interaction_id: str = "interaction-test-1",
    model: interactions.Model = DEFAULT_MODEL,
) -> list[interactions.InteractionSSEEvent]:
    interaction = completed_interaction(text, interaction_id=interaction_id, model=model)
    payloads: tuple[JsonObject, ...] = (
        {
            "event_type": "interaction.created",
            "event_id": "event-1",
            "interaction": {"id": interaction_id, "model": model, "status": "in_progress"},
        },
        {
            "event_type": "step.start",
            "event_id": "event-2",
            "index": 0,
            "step": {"type": "model_output", "content": []},
        },
        {
            "event_type": "step.delta",
            "event_id": "event-3",
            "index": 0,
            "delta": {"type": "text", "text": text},
        },
        {
            "event_type": "step.stop",
            "event_id": "event-4",
            "index": 0,
            "usage": interaction.usage.model_dump(mode="json", exclude_none=True)
            if interaction.usage
            else {},
        },
        {
            "event_type": "interaction.completed",
            "event_id": "event-5",
            "interaction": interaction.model_dump(mode="json", exclude_none=True),
        },
    )
    return [EVENT_ADAPTER.validate_python(payload) for payload in payloads]


def completed_reasoning_stream(
    *,
    signature: str,
    interaction_id: str = "interaction-reasoning-1",
    model: interactions.Model = DEFAULT_MODEL,
) -> list[interactions.InteractionSSEEvent]:
    usage: JsonObject = {
        "total_input_tokens": 5,
        "total_output_tokens": 4,
        "total_thought_tokens": 2,
        "total_tokens": 11,
    }
    steps: list[JsonValue] = [
        {
            "type": "thought",
            "signature": signature,
            "summary": [{"type": "text", "text": "brief rationale"}],
        },
        {
            "type": "model_output",
            "content": [{"type": "text", "text": "reasoned answer"}],
        },
    ]
    payloads: tuple[JsonObject, ...] = (
        {
            "event_type": "interaction.created",
            "event_id": "reasoning-event-1",
            "interaction": {"id": interaction_id, "model": model, "status": "in_progress"},
        },
        {
            "event_type": "step.start",
            "event_id": "reasoning-event-2",
            "index": 0,
            "step": {"type": "thought", "summary": []},
        },
        {
            "event_type": "step.delta",
            "event_id": "reasoning-event-3",
            "index": 0,
            "delta": {
                "type": "thought_summary",
                "content": {"type": "text", "text": "brief rationale"},
            },
        },
        {"event_type": "step.stop", "event_id": "reasoning-event-4", "index": 0},
        {
            "event_type": "step.start",
            "event_id": "reasoning-event-5",
            "index": 1,
            "step": {"type": "model_output", "content": []},
        },
        {
            "event_type": "step.delta",
            "event_id": "reasoning-event-6",
            "index": 1,
            "delta": {"type": "text", "text": "reasoned answer"},
        },
        {
            "event_type": "step.stop",
            "event_id": "reasoning-event-7",
            "index": 1,
            "usage": usage,
        },
        {
            "event_type": "interaction.completed",
            "event_id": "reasoning-event-8",
            "interaction": {
                "id": interaction_id,
                "model": model,
                "status": "completed",
                "steps": steps,
                "usage": usage,
            },
        },
    )
    return [EVENT_ADAPTER.validate_python(payload) for payload in payloads]
