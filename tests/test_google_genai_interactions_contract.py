"""Executable contract checks for google-genai 2.11.0 Interactions.

These tests intentionally import the public ``google.genai.interactions`` module.
The generated models currently live under a private ``_gaos`` package internally,
but the public re-export is the boundary the plugin should depend on.
"""

import json
from typing import Protocol, cast

import httpx
import pytest
from google import genai
from google.auth.credentials import Credentials
from google.genai import interactions, types
from pydantic import TypeAdapter

MODEL_ID = "gemini-2.5-flash"


class _StaticCredentials(Credentials):
    def __init__(self) -> None:
        super().__init__()
        self.token = "test-access-token"

    @property
    def expired(self) -> bool:
        return False

    def refresh(self, request: object) -> None:
        self.token = "test-access-token"


class _EventWithDiscriminator(Protocol):
    event_type: str


def test_exact_google_genai_version_and_public_interactions_surface() -> None:
    assert genai.__version__ == "2.11.0"

    expected_public_types = {
        "Content",
        "CreateModelInteraction",
        "Interaction",
        "InteractionSSEEvent",
        "Step",
        "StepDeltaData",
        "Tool",
        "Usage",
    }

    assert expected_public_types <= set(dir(interactions))
    assert hasattr(genai.Client, "interactions")


def test_create_model_interaction_contract_covers_input_tools_and_generation() -> None:
    request = interactions.CreateModelInteraction(
        model=MODEL_ID,
        input=[
            interactions.UserInputStep(
                content=[
                    interactions.TextContent(text="Describe this image."),
                    interactions.ImageContent(
                        uri="gs://example-bucket/image.png",
                        mime_type="image/png",
                    ),
                ],
            )
        ],
        stream=True,
        system_instruction="Answer concisely.",
        tools=[
            interactions.GoogleSearch(),
            interactions.CodeExecution(),
            interactions.URLContext(),
            interactions.GoogleMaps(latitude=45.5, longitude=-73.6),
        ],
        generation_config=interactions.GenerationConfig(
            max_output_tokens=256,
            temperature=0.2,
            thinking_level="high",
            thinking_summaries="auto",
        ),
    )

    payload = request.model_dump(mode="json", exclude_none=True)

    assert payload["model"] == MODEL_ID
    assert payload["stream"] is True
    assert payload["input"] == [
        {
            "content": [
                {"text": "Describe this image.", "type": "text"},
                {
                    "mime_type": "image/png",
                    "type": "image",
                    "uri": "gs://example-bucket/image.png",
                },
            ],
            "type": "user_input",
        }
    ]
    assert [tool["type"] for tool in payload["tools"]] == [
        "google_search",
        "code_execution",
        "url_context",
        "google_maps",
    ]
    assert payload["generation_config"] == {
        "max_output_tokens": 256,
        "temperature": 0.2,
        "thinking_level": "high",
        "thinking_summaries": "auto",
    }


@pytest.mark.parametrize(
    ("payload", "expected_type", "expected_discriminator"),
    [
        (
            {
                "event_type": "interaction.created",
                "interaction": {
                    "id": "interaction-1",
                    "status": "in_progress",
                    "model": MODEL_ID,
                },
            },
            interactions.InteractionCreatedEvent,
            "interaction.created",
        ),
        (
            {
                "event_type": "step.start",
                "index": 0,
                "step": {"type": "model_output", "content": []},
            },
            interactions.StepStart,
            "step.start",
        ),
        (
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "text", "text": "Hello"},
            },
            interactions.StepDelta,
            "step.delta",
        ),
        (
            {
                "event_type": "step.stop",
                "index": 0,
                "usage": {
                    "total_input_tokens": 3,
                    "total_output_tokens": 4,
                    "total_tokens": 7,
                },
            },
            interactions.StepStop,
            "step.stop",
        ),
        (
            {
                "event_type": "interaction.status_update",
                "interaction_id": "interaction-1",
                "status": "requires_action",
            },
            interactions.InteractionStatusUpdate,
            "interaction.status_update",
        ),
        (
            {
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "interaction-1",
                    "status": "completed",
                    "model": MODEL_ID,
                },
            },
            interactions.InteractionCompletedEvent,
            "interaction.completed",
        ),
        (
            {
                "event_type": "error",
                "error": {"code": 500, "message": "request failed"},
            },
            interactions.ErrorEvent,
            "error",
        ),
    ],
)
def test_stream_event_union_uses_exact_discriminators(
    payload: dict[str, object],
    expected_type: type[object],
    expected_discriminator: str,
) -> None:
    event = TypeAdapter(interactions.InteractionSSEEvent).validate_python(payload)

    assert isinstance(event, expected_type)
    assert cast(_EventWithDiscriminator, event).event_type == expected_discriminator


def test_nonstream_interaction_exposes_output_usage_and_status_contract() -> None:
    interaction = interactions.Interaction(
        id="interaction-1",
        model=MODEL_ID,
        status="completed",
        steps=[
            interactions.ModelOutputStep(
                content=[interactions.TextContent(text="Hello from Gemini.")]
            )
        ],
        usage=interactions.Usage(
            total_cached_tokens=1,
            total_input_tokens=3,
            total_output_tokens=4,
            total_thought_tokens=2,
            total_tokens=9,
        ),
    )

    assert interaction.status == "completed"
    assert interaction.output_text == "Hello from Gemini."
    assert interaction.usage is not None
    assert interaction.usage.total_cached_tokens == 1
    assert interaction.usage.total_input_tokens == 3
    assert interaction.usage.total_output_tokens == 4
    assert interaction.usage.total_thought_tokens == 2
    assert interaction.usage.total_tokens == 9


def test_open_unions_preserve_unknown_variants() -> None:
    event_payload = {"event_type": "interaction.future", "payload": "preserved"}
    event = TypeAdapter(interactions.InteractionSSEEvent).validate_python(event_payload)
    assert isinstance(event, interactions.UnknownInteractionSSEEvent)
    assert event.is_unknown is True
    assert event.raw == event_payload

    step_payload = {"type": "future_step", "payload": "preserved"}
    step = TypeAdapter(interactions.Step).validate_python(step_payload)
    assert isinstance(step, interactions.UnknownStep)
    assert step.is_unknown is True
    assert step.raw == step_payload

    content_payload = {"type": "future_content", "payload": "preserved"}
    content = TypeAdapter(interactions.Content).validate_python(content_payload)
    assert isinstance(content, interactions.UnknownContent)
    assert content.is_unknown is True
    assert content.raw == content_payload

    tool_payload = {"type": "future_tool", "payload": "preserved"}
    tool = TypeAdapter(interactions.Tool).validate_python(tool_payload)
    assert isinstance(tool, interactions.UnknownTool)
    assert tool.is_unknown is True
    assert tool.raw == tool_payload


@pytest.mark.asyncio
async def test_stable_v1_custom_base_url_composes_interactions_request_path() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "interaction-1",
                "model": MODEL_ID,
                "status": "completed",
                "steps": [],
            },
        )

    client = genai.Client(
        api_key="test-api-key",
        http_options=types.HttpOptions(
            api_version="v1",
            base_url="https://gateway.example.test/gemini",
            async_client_args={"transport": httpx.MockTransport(handler)},
        ),
    )

    try:
        response = await client.aio.interactions.create(model=MODEL_ID, input="Hello")
    finally:
        await client.aio.aclose()

    assert isinstance(response, interactions.Interaction)
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "https://gateway.example.test/gemini/v1/interactions"
    assert json.loads(requests[0].content) == {"model": MODEL_ID, "input": "Hello"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service", "api_version", "expected_url", "expected_authentication"),
    [
        (
            "developer",
            "v1",
            "https://generativelanguage.googleapis.com/v1/interactions",
            "x-goog-api-key",
        ),
        (
            "enterprise",
            "v1beta1",
            "https://aiplatform.googleapis.com/v1beta1/projects/test-project/locations/global/interactions",
            "authorization",
        ),
    ],
)
async def test_service_specific_interactions_paths(
    service: str,
    api_version: str,
    expected_url: str,
    expected_authentication: str,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"id": "interaction-1", "status": "completed", "steps": []},
        )

    http_options = types.HttpOptions(
        api_version=api_version,
        async_client_args={"transport": httpx.MockTransport(handler)},
    )
    if service == "enterprise":
        client = genai.Client(
            enterprise=True,
            project="test-project",
            location="global",
            credentials=_StaticCredentials(),
            http_options=http_options,
        )
    else:
        client = genai.Client(api_key="test-api-key", http_options=http_options)
    try:
        await client.aio.interactions.create(model=MODEL_ID, input="Hello")
    finally:
        await client.aio.aclose()

    assert len(requests) == 1
    captured = requests[0]
    snapshot = {
        "method": captured.method,
        "url": str(captured.url),
        "authentication": expected_authentication,
        "body": json.loads(captured.content),
    }
    assert expected_authentication in captured.headers
    assert snapshot == {
        "method": "POST",
        "url": expected_url,
        "authentication": expected_authentication,
        "body": {"model": MODEL_ID, "input": "Hello"},
    }
    assert "test-access-token" not in json.dumps(snapshot)


@pytest.mark.asyncio
async def test_developer_request_snapshot_serializes_canonical_interactions_options() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"id": "interaction-2", "status": "completed", "steps": []},
        )

    client = genai.Client(
        api_key="secret-that-must-not-appear-in-snapshot",
        http_options=types.HttpOptions(
            api_version="v1",
            async_client_args={"transport": httpx.MockTransport(handler)},
        ),
    )
    request = interactions.CreateModelInteraction(
        model=MODEL_ID,
        input="Continue safely.",
        stream=False,
        store=True,
        previous_interaction_id="interaction-1",
        response_format={"type": "json_schema", "name": "answer"},
        safety_settings=[interactions.SafetySetting(type="harassment", threshold="off")],
        tools=[interactions.GoogleSearch()],
        generation_config=interactions.GenerationConfig(
            temperature=0.2,
            max_output_tokens=64,
        ),
    )

    try:
        await client.aio.interactions.create(**request.model_dump(mode="python", exclude_none=True))
    finally:
        await client.aio.aclose()

    assert len(requests) == 1
    captured = requests[0]
    snapshot = {
        "method": captured.method,
        "url": str(captured.url),
        "authentication": "x-goog-api-key" if "x-goog-api-key" in captured.headers else None,
        "content_type": captured.headers["content-type"],
        "body": json.loads(captured.content),
    }
    assert snapshot == {
        "method": "POST",
        "url": "https://generativelanguage.googleapis.com/v1/interactions",
        "authentication": "x-goog-api-key",
        "content_type": "application/json",
        "body": request.model_dump(mode="json", exclude_none=True),
    }
    assert "secret-that-must-not-appear-in-snapshot" not in json.dumps(snapshot)


@pytest.mark.asyncio
async def test_stream_request_snapshot_uses_sse_accept_header_and_stream_body() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=(
                "event: interaction.completed\n"
                'data: {"event_type":"interaction.completed",'
                '"interaction":{"id":"interaction-stream","status":"completed"}}\n\n'
            ),
        )

    client = genai.Client(
        api_key="sanitized-test-key",
        http_options=types.HttpOptions(
            api_version="v1",
            async_client_args={"transport": httpx.MockTransport(handler)},
        ),
    )
    try:
        stream = await client.aio.interactions.create(
            model=MODEL_ID, input="Hello", stream=True, store=False
        )
        async with stream:
            events = [event async for event in stream]
    finally:
        await client.aio.aclose()

    assert len(events) == 1
    assert len(requests) == 1
    assert requests[0].headers["accept"] == "text/event-stream"
    assert json.loads(requests[0].content) == {
        "model": MODEL_ID,
        "input": "Hello",
        "stream": True,
        "store": False,
    }
