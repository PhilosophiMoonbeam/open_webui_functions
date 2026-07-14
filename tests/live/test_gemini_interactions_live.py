"""Credential-gated, output-redacted Gemini Interactions smoke tests."""

import asyncio
import os
import secrets

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


def _enterprise_settings() -> tuple[str, str, str, str]:
    """Require an explicitly selected Enterprise project, location, model, and API version."""
    if os.getenv("RUN_GEMINI_ENTERPRISE_LIVE_TESTS") != "1":
        pytest.skip("set RUN_GEMINI_ENTERPRISE_LIVE_TESTS=1 to enable Enterprise external calls")
    names = (
        "GEMINI_ENTERPRISE_PROJECT",
        "GEMINI_ENTERPRISE_LOCATION",
        "GEMINI_ENTERPRISE_MODEL",
        "GEMINI_ENTERPRISE_API_VERSION",
    )
    values = tuple(os.getenv(name) for name in names)
    missing = [name for name, value in zip(names, values, strict=True) if not value]
    if missing:
        pytest.fail("RUN_GEMINI_ENTERPRISE_LIVE_TESTS=1 requires " + ", ".join(missing))
    project, location, model, api_version = values
    assert project and location and model and api_version
    return project, location, model, api_version


def _enterprise_client() -> tuple[genai.Client, str]:
    project, location, model, api_version = _enterprise_settings()
    return (
        genai.Client(
            enterprise=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(api_version=api_version),
        ),
        model,
    )


async def _cleanup_stored_interactions(
    client: genai.Client, interaction_ids: list[str]
) -> list[str]:
    """Best-effort cleanup that never emits IDs or provider error details."""
    failures: list[str] = []
    try:
        for interaction_id in reversed(interaction_ids):
            try:
                async with asyncio.timeout(15):
                    await client.aio.interactions.delete(interaction_id)
            except Exception:
                failures.append("delete")
    finally:
        try:
            async with asyncio.timeout(15):
                await client.aio.aclose()
        except Exception:
            failures.append("close")
    return failures


async def _run_previous_interaction_probe(client: genai.Client, model: str) -> None:
    """Prove semantic continuation while guaranteeing provider cleanup before exit."""
    token = f"T{secrets.token_hex(12).upper()}"
    created_interaction_ids: list[str] = []
    primary_failure: BaseException | None = None
    try:
        async with asyncio.timeout(60):
            first = await client.aio.interactions.create(
                model=model,
                input=f"Remember this temporary token: {token}",
                stream=False,
                store=True,
            )
            assert isinstance(first, interactions.Interaction)
            assert first.id
            created_interaction_ids.append(first.id)
            second = await client.aio.interactions.create(
                model=model,
                input="Return only the temporary token I asked you to remember.",
                previous_interaction_id=first.id,
                stream=False,
                store=True,
            )
            assert isinstance(second, interactions.Interaction)
            assert second.id
            created_interaction_ids.append(second.id)
            if second.status != "completed" or (second.output_text or "").strip() != token:
                pytest.fail(
                    "Stored Interaction did not preserve semantic continuation.", pytrace=False
                )
    except BaseException as exc:
        primary_failure = exc

    cleanup_task = asyncio.create_task(
        _cleanup_stored_interactions(client, created_interaction_ids)
    )
    try:
        cleanup_failures = await asyncio.shield(cleanup_task)
    except asyncio.CancelledError as exc:
        cleanup_failures = await cleanup_task
        if primary_failure is None:
            primary_failure = exc
    if primary_failure is not None:
        if cleanup_failures:
            primary_failure.add_note("Provider cleanup also failed; details were redacted.")
        raise primary_failure
    if cleanup_failures:
        pytest.fail("Provider cleanup failed; details were redacted.", pytrace=False)


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
    await _run_previous_interaction_probe(client, model)


@pytest.mark.live
@pytest.mark.enterprise_api
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["unary", "sse"])
async def test_enterprise_text_interaction_live(stream: bool) -> None:
    client, model = _enterprise_client()
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
@pytest.mark.enterprise_api
@pytest.mark.asyncio
async def test_enterprise_previous_interaction_live() -> None:
    client, model = _enterprise_client()
    await _run_previous_interaction_probe(client, model)
