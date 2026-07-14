"""Credential-gated, output-redacted Gemini Interactions smoke tests."""

import asyncio
import os

import pytest
from google import genai
from google.genai import interactions, types


def _require_live() -> None:
    if os.getenv("RUN_GEMINI_LIVE_TESTS") != "1":
        pytest.skip("set RUN_GEMINI_LIVE_TESTS=1 to enable external Gemini calls")


def _developer_api_key() -> str:
    """Require credentials after the operator explicitly opts into live calls."""
    _require_live()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.fail("RUN_GEMINI_LIVE_TESTS=1 requires GEMINI_API_KEY")
    return api_key


@pytest.mark.live
@pytest.mark.developer_api
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["unary", "sse"])
async def test_developer_text_interaction_live(stream: bool) -> None:
    api_key = _developer_api_key()
    model = os.getenv("GEMINI_LIVE_MODEL", "gemini-2.5-flash")
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1"),
    )
    async with asyncio.timeout(60):
        try:
            result = await client.aio.interactions.create(
                model=model,
                input="Reply with one short word.",
                stream=stream,
                store=False,
            )
            if stream:
                assert not isinstance(result, interactions.Interaction)
                terminal = False
                async with result:
                    async for event in result:
                        terminal = terminal or isinstance(
                            event, interactions.InteractionCompletedEvent
                        )
                assert terminal
            else:
                assert isinstance(result, interactions.Interaction)
                assert result.status == "completed"
                assert bool(result.output_text)
        finally:
            await client.aio.aclose()


@pytest.mark.live
@pytest.mark.developer_api
@pytest.mark.asyncio
async def test_developer_previous_interaction_live() -> None:
    api_key = _developer_api_key()
    model = os.getenv("GEMINI_LIVE_MODEL", "gemini-2.5-flash")
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1"),
    )
    created_interaction_ids: list[str] = []
    async with asyncio.timeout(60):
        try:
            first = await client.aio.interactions.create(
                model=model,
                input="Remember the sanitized token ALPHA.",
                stream=False,
                store=True,
            )
            assert isinstance(first, interactions.Interaction)
            assert first.id
            created_interaction_ids.append(first.id)
            second = await client.aio.interactions.create(
                model=model,
                input="Acknowledge the remembered token with one word.",
                previous_interaction_id=first.id,
                stream=False,
                store=True,
            )
            assert isinstance(second, interactions.Interaction)
            assert second.status == "completed"
            assert second.id
            created_interaction_ids.append(second.id)
        finally:
            for interaction_id in reversed(created_interaction_ids):
                await client.aio.interactions.delete(interaction_id)
            await client.aio.aclose()
