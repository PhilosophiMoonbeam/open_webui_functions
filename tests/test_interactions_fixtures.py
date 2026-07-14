"""Coverage and validity checks for the exact 2.11 JSON fixture corpora."""

from google import genai
from google.genai import interactions

from tests.fixtures.interactions import (
    ANNOTATION_ADAPTER,
    ANNOTATION_PAYLOADS,
    CONTENT_ADAPTER,
    CONTENT_PAYLOADS,
    DELTA_ADAPTER,
    DELTA_PAYLOADS,
    ERROR_PAYLOADS,
    EVENT_ADAPTER,
    EVENT_PAYLOADS,
    INTERACTION_STATUSES,
    STEP_ADAPTER,
    STEP_PAYLOADS,
    TOOL_ADAPTER,
    TOOL_PAYLOADS,
    USAGE_PAYLOAD,
    completed_interaction,
    completed_reasoning_stream,
    completed_stream,
    validate_corpus,
)


def test_fixture_corpora_target_the_locked_public_sdk() -> None:
    assert genai.__version__ == "2.11.0"
    assert all(adapter is not None for adapter in (EVENT_ADAPTER, STEP_ADAPTER, DELTA_ADAPTER))


def test_event_corpus_covers_every_exact_and_unknown_variant() -> None:
    parsed = validate_corpus(EVENT_ADAPTER, EVENT_PAYLOADS)

    assert {type(event).__name__ for event in parsed.values()} == {
        "InteractionCreatedEvent",
        "InteractionCompletedEvent",
        "InteractionStatusUpdate",
        "ErrorEvent",
        "StepStart",
        "StepDelta",
        "StepStop",
        "UnknownInteractionSSEEvent",
    }
    unknown = parsed["unknown"]
    assert isinstance(unknown, interactions.UnknownInteractionSSEEvent)
    assert unknown.raw == EVENT_PAYLOADS["unknown"]


def test_step_corpus_covers_every_exact_and_unknown_variant() -> None:
    parsed = validate_corpus(STEP_ADAPTER, STEP_PAYLOADS)

    assert {type(step).__name__ for step in parsed.values()} == {
        "UserInputStep",
        "ModelOutputStep",
        "ThoughtStep",
        "FunctionCallStep",
        "CodeExecutionCallStep",
        "URLContextCallStep",
        "MCPServerToolCallStep",
        "GoogleSearchCallStep",
        "FileSearchCallStep",
        "GoogleMapsCallStep",
        "FunctionResultStep",
        "CodeExecutionResultStep",
        "URLContextResultStep",
        "GoogleSearchResultStep",
        "MCPServerToolResultStep",
        "FileSearchResultStep",
        "GoogleMapsResultStep",
        "UnknownStep",
    }
    unknown = parsed["unknown"]
    assert isinstance(unknown, interactions.UnknownStep)
    assert unknown.raw == STEP_PAYLOADS["unknown"]


def test_content_and_annotation_corpora_cover_open_unions() -> None:
    contents = validate_corpus(CONTENT_ADAPTER, CONTENT_PAYLOADS)
    annotations = validate_corpus(ANNOTATION_ADAPTER, ANNOTATION_PAYLOADS)

    assert {type(content).__name__ for content in contents.values()} == {
        "TextContent",
        "ImageContent",
        "AudioContent",
        "DocumentContent",
        "VideoContent",
        "UnknownContent",
    }
    assert {type(annotation).__name__ for annotation in annotations.values()} == {
        "URLCitation",
        "FileCitation",
        "PlaceCitation",
        "UnknownAnnotation",
    }
    assert isinstance(contents["unknown"], interactions.UnknownContent)
    assert isinstance(annotations["unknown"], interactions.UnknownAnnotation)


def test_delta_corpus_covers_every_exact_and_unknown_variant() -> None:
    parsed = validate_corpus(DELTA_ADAPTER, DELTA_PAYLOADS)

    assert {type(delta).__name__ for delta in parsed.values()} == {
        "TextDelta",
        "ImageDelta",
        "AudioDelta",
        "DocumentDelta",
        "VideoDelta",
        "ThoughtSummaryDelta",
        "ThoughtSignatureDelta",
        "TextAnnotationDelta",
        "ArgumentsDelta",
        "CodeExecutionCallDelta",
        "URLContextCallDelta",
        "GoogleSearchCallDelta",
        "MCPServerToolCallDelta",
        "FileSearchCallDelta",
        "GoogleMapsCallDelta",
        "RetrievalCallDelta",
        "CodeExecutionResultDelta",
        "URLContextResultDelta",
        "GoogleSearchResultDelta",
        "MCPServerToolResultDelta",
        "FileSearchResultDelta",
        "GoogleMapsResultDelta",
        "RetrievalResultDelta",
        "FunctionResultDelta",
        "UnknownStepDeltaData",
    }
    unknown = parsed["unknown"]
    assert isinstance(unknown, interactions.UnknownStepDeltaData)
    assert unknown.raw == DELTA_PAYLOADS["unknown"]


def test_tool_corpus_covers_every_exact_and_unknown_variant() -> None:
    parsed = validate_corpus(TOOL_ADAPTER, TOOL_PAYLOADS)

    assert {type(tool).__name__ for tool in parsed.values()} == {
        "Function",
        "CodeExecution",
        "URLContext",
        "ComputerUse",
        "MCPServer",
        "GoogleSearch",
        "FileSearch",
        "GoogleMaps",
        "Retrieval",
        "UnknownTool",
    }
    unknown = parsed["unknown"]
    assert isinstance(unknown, interactions.UnknownTool)
    assert unknown.raw == TOOL_PAYLOADS["unknown"]


def test_status_usage_and_error_payloads_are_complete_public_models() -> None:
    statuses = {interactions.Interaction(status=status).status for status in INTERACTION_STATUSES}
    usage = interactions.Usage.model_validate(USAGE_PAYLOAD)
    stream_error = EVENT_ADAPTER.validate_python(ERROR_PAYLOADS["stream"])
    model_error = STEP_ADAPTER.validate_python(ERROR_PAYLOADS["model_output"])

    assert statuses == set(INTERACTION_STATUSES)
    assert usage.total_tokens == 20
    assert usage.grounding_tool_count and usage.grounding_tool_count[0].count == 1
    assert isinstance(stream_error, interactions.ErrorEvent)
    assert stream_error.error and stream_error.error.code == "INTERNAL"
    assert isinstance(model_error, interactions.ModelOutputStep)
    assert model_error.error and model_error.error.code == 13


def test_coherent_builders_are_valid_without_incomplete_constructed_models() -> None:
    interaction = completed_interaction()
    stream = completed_stream()
    reasoning_stream = completed_reasoning_stream(signature="c2ln")

    assert interaction.status == "completed"
    assert interaction.output_text == "Hello from Gemini."
    assert isinstance(stream[-1], interactions.InteractionCompletedEvent)
    assert isinstance(reasoning_stream[-1], interactions.InteractionCompletedEvent)
    assert len(reasoning_stream) == 8
