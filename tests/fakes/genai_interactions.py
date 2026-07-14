"""Strict google-genai Interactions fakes.

The fake client intentionally exposes only ``aio.interactions`` and ``aio.aclose``.
Adding a maintained GenerateContent call to the pipe therefore fails loudly instead
of being accidentally accepted by a permissive ``MagicMock``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import cast

from google.genai import interactions


class FakeInteractionStream(AsyncIterator[interactions.InteractionSSEEvent]):
    """Deterministic async stream with observable ownership and cleanup."""

    def __init__(
        self,
        events: Iterable[interactions.InteractionSSEEvent],
        *,
        error: BaseException | None = None,
    ) -> None:
        self._events = iter(events)
        self._error = error
        self._error_raised = False
        self.enter_count = 0
        self.close_count = 0

    @property
    def closed(self) -> bool:
        return self.close_count > 0

    def __aiter__(self) -> FakeInteractionStream:
        return self

    async def __anext__(self) -> interactions.InteractionSSEEvent:
        if self.closed:
            raise StopAsyncIteration
        try:
            return next(self._events)
        except StopIteration:
            if self._error is not None and not self._error_raised:
                self._error_raised = True
                raise self._error from None
            raise StopAsyncIteration from None

    async def close(self) -> None:
        if not self.closed:
            self.close_count += 1

    async def __aenter__(self) -> FakeInteractionStream:
        self.enter_count += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        await self.close()


InteractionResult = interactions.Interaction | FakeInteractionStream | BaseException


@dataclass(slots=True)
class FakeInteractions:
    """Scripted boundary that validates every create request with public 2.11 types."""

    outcomes: deque[InteractionResult]
    resume_outcomes: deque[FakeInteractionStream] = field(default_factory=deque)
    requests: list[interactions.CreateModelInteraction] = field(default_factory=list)
    raw_requests: list[dict[str, object]] = field(default_factory=list)
    gets: list[tuple[str, bool, str | None]] = field(default_factory=list)

    def __init__(
        self,
        outcomes: Iterable[InteractionResult],
        *,
        resume_outcomes: Iterable[FakeInteractionStream] = (),
    ) -> None:
        self.outcomes = deque(outcomes)
        self.resume_outcomes = deque(resume_outcomes)
        self.requests = []
        self.raw_requests = []
        self.gets = []

    async def create(self, **request: object) -> interactions.Interaction | FakeInteractionStream:
        self.raw_requests.append(dict(request))
        validated = interactions.CreateModelInteraction.model_validate(request)
        self.requests.append(validated)
        if not self.outcomes:
            raise AssertionError("Unexpected interactions.create call")
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        requested_stream = validated.stream is True
        if requested_stream != isinstance(outcome, FakeInteractionStream):
            raise AssertionError("Scripted outcome does not match request.stream")
        return outcome

    async def get(
        self, id: str, *, stream: bool, last_event_id: str | None = None
    ) -> FakeInteractionStream:
        self.gets.append((id, stream, last_event_id))
        if not stream:
            raise AssertionError("The pipe may only resume an Interaction as a stream")
        if not self.resume_outcomes:
            raise AssertionError("Unexpected interactions.get call")
        return self.resume_outcomes.popleft()

    async def cancel(self, id: str) -> None:
        raise AssertionError(f"Unexpected interactions.cancel call for {id}")

    def assert_exhausted(self) -> None:
        assert not self.outcomes, "Not all scripted create outcomes were consumed"
        assert not self.resume_outcomes, "Not all scripted resume outcomes were consumed"


@dataclass(slots=True)
class FakeAsyncClient:
    interactions: FakeInteractions
    close_count: int = 0

    async def aclose(self) -> None:
        self.close_count += 1


class FakeGenAIClient:
    """Minimal SDK client; notably has no ``models`` or top-level interactions fake."""

    __slots__ = ("aio",)

    def __init__(self, scripted: FakeInteractions) -> None:
        self.aio = FakeAsyncClient(scripted)

    @property
    def interactions(self) -> object:
        raise AssertionError("Production generation must use client.aio.interactions")


def as_sdk_client(fake: FakeGenAIClient) -> object:
    """Keep the unsafe SDK cast at the test boundary."""
    return cast(object, fake)
