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
    project, location, model, api_version = (os.environ[name] for name in names)
    return project, location, model, api_version


def _enterprise_client() -> tuple[genai.Client, str]:
    project, location, model, api_version = _enterprise_settings()
    try:
        client = genai.Client(
            enterprise=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(api_version=api_version),
        )
    except Exception:
        pytest.fail(
            "Unable to initialize the Enterprise client; details were redacted.", pytrace=False
        )
    return client, model


def _developer_client(api_key: str) -> genai.Client:
    try:
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1"),
        )
    except Exception:
        pytest.fail(
            "Unable to initialize the Developer client; details were redacted.", pytrace=False
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
    provider_failure = False
    try:
        async with asyncio.timeout(60):
            first = await client.aio.interactions.create(
                model=model,
                input=f"Remember this temporary token: {token}",
                stream=False,
                store=True,
            )
            if not isinstance(first, interactions.Interaction) or not first.id:
                pytest.fail(
                    "First stored Interaction was invalid; details were redacted.", pytrace=False
                )
            created_interaction_ids.append(first.id)
            second = await client.aio.interactions.create(
                model=model,
                input="Return only the temporary token I asked you to remember.",
                previous_interaction_id=first.id,
                stream=False,
                store=True,
            )
            if not isinstance(second, interactions.Interaction) or not second.id:
                pytest.fail(
                    "Continuation Interaction was invalid; details were redacted.", pytrace=False
                )
            created_interaction_ids.append(second.id)
            if second.status != "completed" or (second.output_text or "").strip() != token:
                pytest.fail(
                    "Stored Interaction did not preserve semantic continuation.", pytrace=False
                )
    except (asyncio.CancelledError, pytest.fail.Exception) as exc:
        primary_failure = exc
    except Exception:
        provider_failure = True

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
    if provider_failure:
        message = "Gemini Interaction request failed; provider details were redacted."
        if cleanup_failures:
            message += " Provider cleanup also failed."
        pytest.fail(message, pytrace=False)
    if cleanup_failures:
        pytest.fail("Provider cleanup failed; details were redacted.", pytrace=False)


async def _run_text_interaction_probe(client: genai.Client, model: str, stream: bool) -> None:
    """Run an ephemeral text probe without exposing provider objects or errors."""
    primary_failure: BaseException | None = None
    provider_failure = False
    try:
        async with asyncio.timeout(60):
            result = await client.aio.interactions.create(
                model=model,
                input="Reply with one short word.",
                stream=stream,
                store=False,
            )
            if stream:
                if isinstance(result, interactions.Interaction):
                    pytest.fail("Streaming request returned a unary result.", pytrace=False)
                terminal = False
                async with result:
                    async for event in result:
                        terminal = terminal or isinstance(
                            event, interactions.InteractionCompletedEvent
                        )
                if not terminal:
                    pytest.fail("Streaming request had no completed terminal event.", pytrace=False)
            else:
                if not isinstance(result, interactions.Interaction):
                    pytest.fail("Unary request returned an invalid result.", pytrace=False)
                if result.status != "completed" or not result.output_text:
                    pytest.fail("Unary request did not complete with text.", pytrace=False)
    except (asyncio.CancelledError, pytest.fail.Exception) as exc:
        primary_failure = exc
    except Exception:
        provider_failure = True

    close_failure = False
    try:
        async with asyncio.timeout(15):
            await client.aio.aclose()
    except Exception:
        close_failure = True

    if primary_failure is not None:
        if close_failure:
            primary_failure.add_note("Client cleanup also failed; details were redacted.")
        raise primary_failure
    if provider_failure:
        pytest.fail(
            "Gemini Interaction request failed; provider details were redacted.", pytrace=False
        )
    if close_failure:
        pytest.fail("Client cleanup failed; details were redacted.", pytrace=False)


@pytest.mark.live
@pytest.mark.developer_api
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["unary", "sse"])
async def test_developer_text_interaction_live(stream: bool) -> None:
    api_key = _developer_api_key()
    model = os.getenv("GEMINI_LIVE_MODEL", "gemini-2.5-flash")
    await _run_text_interaction_probe(_developer_client(api_key), model, stream)


@pytest.mark.live
@pytest.mark.developer_api
@pytest.mark.asyncio
async def test_developer_previous_interaction_live() -> None:
    api_key = _developer_api_key()
    model = os.getenv("GEMINI_LIVE_MODEL", "gemini-2.5-flash")
    await _run_previous_interaction_probe(_developer_client(api_key), model)


@pytest.mark.live
@pytest.mark.enterprise_api
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["unary", "sse"])
async def test_enterprise_text_interaction_live(stream: bool) -> None:
    client, model = _enterprise_client()
    await _run_text_interaction_probe(client, model, stream)


@pytest.mark.live
@pytest.mark.enterprise_api
@pytest.mark.asyncio
async def test_enterprise_previous_interaction_live() -> None:
    client, model = _enterprise_client()
    await _run_previous_interaction_probe(client, model)
