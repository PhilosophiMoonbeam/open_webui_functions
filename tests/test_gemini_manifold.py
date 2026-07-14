import asyncio
import base64
import hashlib
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import httpx
import pytest
import pytest_asyncio
import yaml
from aiocache.backends.memory import SimpleMemoryCache
from aiocache.base import BaseCache
from aiocache.serializers import NullSerializer
from fastapi import FastAPI
from google import genai
from google.genai import interactions as interaction_types
from plugins.filters.gemini_map_grounding_toggle import Filter as MapsToggleFilter
from plugins.filters.gemini_reasoning_toggle import Filter as ReasoningToggleFilter
from plugins.filters.gemini_url_context_toggle import Filter as URLContextToggleFilter
from pydantic import ValidationError
from utils.manifold_types import (
    AssistantMessage,
    Body,
    ChatMessageTD,
    Event,
    Metadata,
    UserData,
)

# --- Mock problematic Open WebUI modules BEFORE they are imported by your plugin ---
mock_chats_module = MagicMock()
mock_files_module = MagicMock()
mock_functions_module = MagicMock()
mock_storage_module = MagicMock()
mock_misc_module = MagicMock()

# Use AsyncMock for async methods in Chats
mock_chats_module.Chats = MagicMock()
mock_chats_module.Chats.get_chat_by_id_and_user_id = AsyncMock()

mock_files_module.FileForm = MagicMock()
mock_files_module.Files = MagicMock()
# Mock the return of get_function_by_id for toggle filter tests
mock_functions_module.Functions.get_function_by_id.return_value = MagicMock(
    is_active=True, is_global=True
)
mock_storage_module.Storage = MagicMock()
# Provide a default return value for pop_system_message to avoid TypeErrors in tests that don't set it.
mock_misc_module.pop_system_message.return_value = (None, [])

sys.modules["open_webui.models.chats"] = mock_chats_module
sys.modules["open_webui.models.files"] = mock_files_module
sys.modules["open_webui.models.functions"] = mock_functions_module
sys.modules["open_webui.storage.provider"] = mock_storage_module
sys.modules["open_webui.utils.misc"] = mock_misc_module


import plugins.filters.gemini_manifold_companion as companion_module
import plugins.pipes.gemini_manifold as gemini_manifold_module
from plugins.filters.gemini_manifold_companion import (
    DEFAULT_MODEL_CONFIG_PATH,
    EventEmitter,
    ModelCatalogError,
)
from plugins.filters.gemini_manifold_companion import (
    CatalogAppStateEnvelope as CompanionCatalogEnvelope,
)
from plugins.filters.gemini_manifold_companion import (
    Filter as CompanionFilter,
)
from plugins.filters.gemini_manifold_companion import (
    GroundingEnvelope as CompanionGroundingEnvelope,
)
from plugins.filters.gemini_manifold_companion import (
    ModelCatalog as CompanionModelCatalog,
)
from plugins.pipes.gemini_manifold import (
    CATALOG_MODEL_IDS,
    DEVELOPER_CATALOG_MODEL_IDS,
    ENTERPRISE_CATALOG_MODEL_IDS,
    AppStateModelCatalog,
    AsyncInteractionsBoundary,
    AsyncInteractionStream,
    CatalogInteractions,
    CatalogPricedRate,
    CatalogSupportedService,
    CatalogUnavailableService,
    CatalogUnpricedRate,
    ContentBuildError,
    EndpointIdentity,
    FilesAPIError,
    FilesAPIManager,
    GeminiContentBuilder,
    GeminiPDFProcessor,
    GenAIClientBinding,
    GenerationFailureKind,
    InteractionEnvelopeV1,
    InteractionExecutionError,
    InteractionReducer,
    InteractionRequestOptions,
    InteractionsSDKBoundary,
    LocalFileAccessError,
    LocalFileSource,
    NormalizedInteractionUsage,
    PDFMitigationManager,
    PDFMitigationOutcome,
    PDFProcessingError,
    Pipe,
    PipeEventEmitter,
    PreparedPDFPart,
    PreparedPDFResult,
    SelectedCatalogService,
    genai_errors,
)
from plugins.pipes.gemini_manifold import (
    CatalogAppStateEnvelope as PipeCatalogEnvelope,
)
from plugins.pipes.gemini_manifold import (
    canonical_catalog_bytes as pipe_canonical_catalog_bytes,
)
from plugins.pipes.gemini_manifold import (
    types as gemini_types,
)

from tests.fakes.genai_interactions import (
    FakeGenAIClient,
    FakeInteractions,
    FakeInteractionStream,
)
from tests.fixtures.interactions import (
    completed_interaction,
    completed_reasoning_stream,
    completed_stream,
)

# region Test Constants
# General Users
USER_EMAIL_REGULAR = "regular_user@example.com"
USER_EMAIL_UNPRIVILEGED = "unprivileged_user@example.com"
USER_EMAIL_WHITELISTED = "whitelisted_user@example.com"

# Admin Credentials
ADMIN_FREE_KEY = "admin_default_free_key"
ADMIN_PAID_KEY = "admin_default_paid_key"
ADMIN_GEMINI_BASE_URL = "https://admin.default.gemini.api.com"
ADMIN_ENTERPRISE_PROJECT = "admin_default_enterprise_project"
ADMIN_ENTERPRISE_LOCATION = "admin_default_enterprise_location"

# User Credentials
USER_FREE_KEY = "user_specific_free_key"
USER_PAID_KEY = "user_specific_paid_key"
USER_GEMINI_BASE_URL = "https://user.specific.gemini.api.com"
USER_ENTERPRISE_PROJECT = "user_specific_enterprise_project"
USER_ENTERPRISE_LOCATION = "user_specific_enterprise_location"
# endregion Test Constants


def _developer_identity() -> EndpointIdentity:
    return EndpointIdentity(
        service="developer",
        credential_fingerprint="test-credential",
        api_version="v1",
    )


def _client_error(code: int, status: str, message: str) -> genai_errors.ClientError:
    return genai_errors.ClientError(
        code,
        {"error": {"code": code, "message": message, "status": status}},
    )


def _server_error(code: int, status: str, message: str) -> genai_errors.ServerError:
    return genai_errors.ServerError(
        code,
        {"error": {"code": code, "message": message, "status": status}},
    )


def test_catalogued_model_ids_and_image_policy_default_deny() -> None:
    catalog_path = Path(__file__).parents[1] / "plugins" / "pipes" / "gemini_models.yaml"
    raw_catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    validated = AppStateModelCatalog.model_validate(raw_catalog)
    catalog_models = validated.runtime_models()
    image_policy = catalog_models["gemini-3-pro-image"].services.developer

    assert set(catalog_models) == CATALOG_MODEL_IDS
    assert DEVELOPER_CATALOG_MODEL_IDS == CATALOG_MODEL_IDS
    assert ENTERPRISE_CATALOG_MODEL_IDS == frozenset()
    assert isinstance(image_policy, CatalogSupportedService)
    assert Pipe._is_image_model(image_policy) is True
    assert "gemini-future-unknown" not in catalog_models
    for model_id in CATALOG_MODEL_IDS:
        services = catalog_models[model_id].services
        assert isinstance(services.developer, CatalogSupportedService)
        assert services.developer.availability == "supported"
        assert isinstance(services.enterprise, CatalogUnavailableService)
        assert services.enterprise.availability == "unverified"


def test_pipe_and_companion_protocol_3_catalog_models_round_trip_in_parity() -> None:
    catalog_path = Path(__file__).parents[1] / "plugins" / "pipes" / "gemini_models.yaml"
    raw_catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    companion = CompanionModelCatalog.model_validate(raw_catalog)
    pipe_catalog = AppStateModelCatalog.model_validate(raw_catalog)
    companion_envelope = CompanionCatalogEnvelope.from_catalog(companion)
    pipe_envelope = PipeCatalogEnvelope.model_validate(
        companion_envelope.model_dump(mode="json", exclude_none=False)
    )

    assert pipe_canonical_catalog_bytes(pipe_catalog) == companion_module.canonical_catalog_bytes(
        companion
    )
    assert pipe_envelope.canonical_digest == companion_envelope.canonical_digest
    assert {
        key: value.model_dump(mode="json") for key, value in pipe_catalog.runtime_models().items()
    } == {key: value.model_dump(mode="json") for key, value in companion.runtime_models().items()}


def test_protocol_3_standalone_models_are_frozen_and_reject_the_same_mutation() -> None:
    catalog_path = Path(__file__).parents[1] / "plugins" / "pipes" / "gemini_models.yaml"
    raw_catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    raw_catalog["provider_claims"]["gemini-2.5-flash"]["capabilities"]["google_search"] = False

    assert CompanionModelCatalog.model_config.get("frozen") is True
    assert AppStateModelCatalog.model_config.get("frozen") is True
    with pytest.raises(ValidationError, match="exceeds provider capability"):
        CompanionModelCatalog.model_validate(raw_catalog)
    with pytest.raises(ValidationError, match="exceeds provider capability"):
        AppStateModelCatalog.model_validate(raw_catalog)
    assert "plugins.pipes" not in Path(companion_module.__file__).read_text(encoding="utf-8")
    assert "plugins.filters" not in Path(gemini_manifold_module.__file__).read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("version", [None, "2.1.0", "3.0.1", "invalid"])
def test_pipe_rejects_mismatched_companion_protocol(version: str | None) -> None:
    features = {} if version is None else {"gemini_manifold_companion_version": version}

    with pytest.raises(ValueError, match="protocol mismatch"):
        Pipe._check_companion_filter_version(MagicMock(), features)


def test_pipe_accepts_exact_companion_protocol_pair() -> None:
    Pipe._check_companion_filter_version(
        MagicMock(), {"gemini_manifold_companion_version": "3.0.0"}
    )


@pytest.mark.asyncio
async def test_pipe_rejects_malformed_app_state_catalog_before_routing_or_client_creation(
    pipe_instance_fixture,
) -> None:
    pipe, _ = pipe_instance_fixture
    catalog_path = Path(__file__).parents[1] / "plugins" / "pipes" / "gemini_models.yaml"
    companion = CompanionModelCatalog.model_validate(
        yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    )
    malformed = CompanionCatalogEnvelope.from_catalog(companion).model_dump(
        mode="json", exclude_none=False
    )
    malformed["payload"]["product_authorizations"]["gemini-2.5-flash"]["interactions"].pop("tools")
    request = MagicMock()
    request.app = FastAPI()
    request.app.state._state.update({"gemini_model_catalog": malformed})
    determine_order = AsyncMock()
    get_client = MagicMock()

    with (
        patch.object(pipe, "_determine_execution_order", determine_order),
        patch.object(pipe, "_get_user_client", get_client),
        pytest.raises(ValueError, match="catalog protocol 3"),
    ):
        await pipe.pipe(
            body=cast(Body, {"model": "gemini-2.5-flash", "messages": []}),
            __user__=cast(UserData, {"email": "user@example.test"}),
            __request__=request,
            __event_emitter__=None,
            __metadata__=cast(
                Metadata,
                {"features": {"gemini_manifold_companion_version": "3.0.0"}},
            ),
        )

    determine_order.assert_not_awaited()
    get_client.assert_not_called()


def test_companion_catalog_default_is_version_pinned_and_fails_visibly() -> None:
    assert "/gemini-suite/v3.0.0/" in DEFAULT_MODEL_CONFIG_PATH
    assert "/master/" not in DEFAULT_MODEL_CONFIG_PATH
    CompanionFilter._load_model_config.cache_clear()
    with (
        patch(
            "plugins.filters.gemini_manifold_companion.urllib.request.urlopen",
            side_effect=TimeoutError("offline"),
        ),
        pytest.raises(ModelCatalogError, match="unavailable or invalid"),
    ):
        CompanionFilter._load_model_config(DEFAULT_MODEL_CONFIG_PATH)


# region Fixtures
@pytest.fixture
def mock_pipe_valves_data():
    """
    Fixture to provide a base set of valves data.
    Updated to include all fields from Pipe.Valves for robust Pydantic initialization.
    """
    return {
        "GEMINI_FREE_API_KEY": ADMIN_FREE_KEY,
        "GEMINI_PAID_API_KEY": ADMIN_PAID_KEY,
        "USER_MUST_PROVIDE_AUTH_CONFIG": False,
        "AUTH_WHITELIST": None,
        "GEMINI_API_BASE_URL": ADMIN_GEMINI_BASE_URL,
        "USE_ENTERPRISE": False,
        "ENTERPRISE_PROJECT": None,
        "ENTERPRISE_LOCATION": "global",
        "MODEL_WHITELIST": "*",
        "MODEL_BLACKLIST": None,
        "CACHE_MODELS": True,
        "THINKING_LEVEL": "high",
        "THINKING_SUMMARIES": "auto",
        "USE_FILES_API": True,
        "PDF_LIMIT_MITIGATION": True,
        "THINKING_MODEL_PATTERN": r"gemini-2.5",
        "LOG_LEVEL": "INFO",
    }


@pytest_asyncio.fixture
async def pipe_instance_fixture(mock_pipe_valves_data):
    """
    Helper fixture to setup a Pipe instance with mocked genai.Client constructor
    and yields both the pipe instance and the mock constructor.
    """
    mock_gemini_client_actual_instance = MagicMock()

    with (
        patch(
            "plugins.pipes.gemini_manifold.genai.Client",
            return_value=mock_gemini_client_actual_instance,
        ) as MockedGenAIClientConstructor,
        patch.object(Pipe, "_add_log_handler", MagicMock()),
        patch("sys.stdout", MagicMock()),
    ):
        pipe = Pipe()
        # Initialize with base data from mock_pipe_valves_data
        pipe.valves = Pipe.Valves(**mock_pipe_valves_data)
        # Yield both the pipe instance and the mock for genai.Client constructor
        yield pipe, MockedGenAIClientConstructor

    # Teardown: Clear caches to ensure clean state for subsequent tests
    Pipe._get_or_create_genai_client.cache_clear()
    cache_instance = getattr(pipe._get_genai_models, "cache", None)
    if isinstance(cache_instance, BaseCache):
        await cache_instance.clear()


# endregion Fixtures


# region Test _get_or_create_genai_client
def test_pipe_initialization_with_api_key_prefers_free(mock_pipe_valves_data):
    """
    Tests that when both free and paid API keys are available, the free key is preferred by default.
    """
    mock_gemini_client_instance = MagicMock()

    with (
        patch(
            "plugins.pipes.gemini_manifold.genai.Client",
            return_value=mock_gemini_client_instance,
        ) as MockedGenAIClientConstructor,
        patch.object(Pipe, "_add_log_handler", MagicMock()),
        patch("sys.stdout", MagicMock()),
    ):
        try:
            pipe_instance = Pipe()
            pipe_instance.valves = Pipe.Valves(**mock_pipe_valves_data)

            assert isinstance(pipe_instance.valves, Pipe.Valves)
            assert pipe_instance.valves.GEMINI_FREE_API_KEY == ADMIN_FREE_KEY
            assert pipe_instance.valves.GEMINI_PAID_API_KEY == ADMIN_PAID_KEY

            # Trigger client creation
            pipe_instance._get_user_client(pipe_instance.valves, USER_EMAIL_REGULAR)

            MockedGenAIClientConstructor.assert_called_once_with(
                api_key=ADMIN_FREE_KEY,  # Should prefer the free key
                http_options=gemini_types.HttpOptions(
                    api_version="v1", base_url=ADMIN_GEMINI_BASE_URL
                ),
            )
        finally:
            Pipe._get_or_create_genai_client.cache_clear()


def test_get_user_client_no_auth_provided_raises_error(mock_pipe_valves_data):
    """
    Tests that genai.Client is NOT called and an error is raised when no API keys
    or Enterprise project are provided.
    """
    # Configure valves to have no API keys and no Gemini Enterprise project
    mock_pipe_valves_data["GEMINI_FREE_API_KEY"] = None
    mock_pipe_valves_data["GEMINI_PAID_API_KEY"] = None
    mock_pipe_valves_data["USE_ENTERPRISE"] = False
    mock_pipe_valves_data["ENTERPRISE_PROJECT"] = None

    with (
        patch("plugins.pipes.gemini_manifold.genai.Client") as MockedGenAIClientConstructor,
        patch.object(Pipe, "_add_log_handler", MagicMock()),
        patch("sys.stdout", MagicMock()),
    ):
        pipe_instance = Pipe()
        pipe_instance.valves = Pipe.Valves(**mock_pipe_valves_data)

        with pytest.raises(ValueError, match="Failed to initialize the configured Gemini client"):
            pipe_instance._get_user_client(pipe_instance.valves, USER_EMAIL_REGULAR)

        MockedGenAIClientConstructor.assert_not_called()
        Pipe._get_or_create_genai_client.cache_clear()


@pytest.mark.parametrize(
    "keys_provided, expected_key",
    [
        (
            {"GEMINI_FREE_API_KEY": "free_only_key", "GEMINI_PAID_API_KEY": None},
            "free_only_key",
        ),
        (
            {"GEMINI_FREE_API_KEY": None, "GEMINI_PAID_API_KEY": "paid_only_key"},
            "paid_only_key",
        ),
    ],
)
def test_client_creation_uses_available_gemini_api_key(
    pipe_instance_fixture, keys_provided, expected_key
):
    """
    Tests that the Gemini Developer API client is created using whichever API key (free or paid) is available.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.GEMINI_FREE_API_KEY = keys_provided["GEMINI_FREE_API_KEY"]
    pipe.valves.GEMINI_PAID_API_KEY = keys_provided["GEMINI_PAID_API_KEY"]
    pipe.valves.USE_ENTERPRISE = False

    pipe._get_user_client(pipe.valves, USER_EMAIL_REGULAR)

    MockedGenAIClientConstructor.assert_called_once_with(
        api_key=expected_key,
        http_options=gemini_types.HttpOptions(api_version="v1", base_url=ADMIN_GEMINI_BASE_URL),
    )
    Pipe._get_or_create_genai_client.cache_clear()


def test_client_creation_uses_enterprise_when_configured(pipe_instance_fixture):
    """
    Tests that Gemini Enterprise client is created when USE_ENTERPRISE is True and ENTERPRISE_PROJECT is provided.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USE_ENTERPRISE = True
    pipe.valves.ENTERPRISE_PROJECT = "test_enterprise_project_id"
    pipe.valves.ENTERPRISE_LOCATION = "europe-west4"

    pipe._get_user_client(pipe.valves, USER_EMAIL_REGULAR)

    MockedGenAIClientConstructor.assert_called_once_with(
        enterprise=True,
        project="test_enterprise_project_id",
        location="europe-west4",
        http_options=gemini_types.HttpOptions(
            api_version="v1beta1", base_url=ADMIN_GEMINI_BASE_URL
        ),
    )
    Pipe._get_or_create_genai_client.cache_clear()


def test_client_creation_falls_back_to_gemini_api_with_warning(pipe_instance_fixture):
    """
    Tests fallback to Gemini Developer API with a warning when USE_ENTERPRISE is True,
    but ENTERPRISE_PROJECT is not provided.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USE_ENTERPRISE = True
    pipe.valves.ENTERPRISE_PROJECT = None
    pipe.valves.GEMINI_FREE_API_KEY = "fallback_free_key"
    pipe.valves.GEMINI_PAID_API_KEY = "fallback_paid_key"  # Should not be used

    with patch("plugins.pipes.gemini_manifold.log.warning") as mock_log_warning:
        pipe._get_user_client(pipe.valves, USER_EMAIL_REGULAR)
        mock_log_warning.assert_called_once_with(
            "Gemini Enterprise is enabled but no project is set. Using Gemini Developer API."
        )

    MockedGenAIClientConstructor.assert_called_once_with(
        api_key="fallback_free_key",  # Should fall back to the free key
        http_options=gemini_types.HttpOptions(api_version="v1", base_url=ADMIN_GEMINI_BASE_URL),
    )
    Pipe._get_or_create_genai_client.cache_clear()


# endregion Test _get_or_create_genai_client


def _selected_service(
    *,
    model_id: str = "gemini-test",
    service: Literal["developer", "enterprise"] = "developer",
    image: bool = False,
    inputs: list[str] | None = None,
    files: bool = True,
    external_urls: bool = True,
    store: bool = True,
    url_context: bool = True,
    output_tokens: int = 65_536,
    free_tier: bool = True,
    input_price: float = 1.0,
    output_price: float = 2.0,
    image_price: float = 30.0,
    custom_function_calling: bool = True,
) -> SelectedCatalogService:
    payload = {
        "availability": "supported",
        "lifecycle": "stable",
        "limits": {"input_tokens": 1_048_576, "output_tokens": output_tokens},
        "content": {
            "inputs": inputs or ["text", "image", "video", "audio", "document"],
            "outputs": ["text", "image"] if image else ["text"],
        },
        "interactions": {
            "store": store,
            "files": files,
            "external_urls": external_urls,
            "response_format": True,
            "custom_function_calling": custom_function_calling,
            "thinking": {
                "supported": True,
                "control": "known",
                "levels": ["minimal", "low", "medium", "high"],
                "summaries": True,
            },
            "tools": {
                "google_search": True,
                "code_execution": True,
                "url_context": url_context,
                "google_maps": True,
                "file_search": False,
            },
        },
        "pricing": {
            "free_tier": free_tier,
            "excluded_features": [],
            "threshold_basis": "total_input_tokens_including_cached",
            "input": {
                modality: {
                    "state": "priced",
                    "tiers": [{"up_to_prompt_tokens": None, "price_per_million": input_price}],
                }
                for modality in (inputs or ["text", "image", "video", "audio", "document"])
            },
            "cached_input": {
                modality: {
                    "state": "priced",
                    "tiers": [{"up_to_prompt_tokens": None, "price_per_million": input_price / 10}],
                }
                for modality in (inputs or ["text", "image", "video", "audio", "document"])
            },
            "output": {
                "text": {
                    "state": "priced",
                    "tiers": [{"up_to_prompt_tokens": None, "price_per_million": output_price}],
                },
                **(
                    {
                        "image": {
                            "state": "priced",
                            "tiers": [
                                {
                                    "up_to_prompt_tokens": None,
                                    "price_per_million": image_price,
                                }
                            ],
                        }
                    }
                    if image
                    else {}
                ),
            },
        },
    }
    return SelectedCatalogService(
        model_id=model_id,
        service=service,
        policy=CatalogSupportedService.model_validate(payload),
    )


def _interaction_policy(*, image: bool = False) -> SelectedCatalogService:
    return _selected_service(image=image)


def _builder_service_policy(
    *,
    inputs: list[str] | None = None,
    files: bool = True,
    external_urls: bool = True,
) -> CatalogSupportedService:
    return _selected_service(
        inputs=inputs,
        files=files,
        external_urls=external_urls,
    ).policy


def _catalog_model_entry(*, image: bool = False, free_tier: bool = True) -> dict[str, object]:
    supported = _selected_service(image=image, free_tier=free_tier).policy.model_dump(mode="json")
    return {
        "services": {
            "developer": supported,
            "enterprise": {"availability": "unverified", "reason": "not verified"},
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filter_type", "feature"),
    [
        (ReasoningToggleFilter, "reasoning"),
        (URLContextToggleFilter, "url_context"),
        (MapsToggleFilter, "google_maps"),
    ],
)
async def test_toggle_filters_write_canonical_feature_schema(filter_type, feature):
    body = {"metadata": {"features": {}}}
    result = await filter_type().inlet(body)
    assert result["metadata"]["features"] == {feature: True}


@pytest.mark.asyncio
async def test_interaction_options_map_generation_thinking_and_schema(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    metadata = {
        "canonical_model_id": "gemini-test",
        "features": {"reasoning": True},
        "merged_custom_params": {"reasoning_effort": "medium"},
    }
    options = await pipe._build_interaction_request_options(
        {
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": 123,
            "stop": "END",
            "seed": 7,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}},
            },
        },
        metadata,
        pipe.valves,
        _interaction_policy(),
    )

    assert isinstance(options, InteractionRequestOptions)
    assert options.generation_config.thinking_level == "medium"
    assert options.generation_config.thinking_summaries == "auto"
    assert options.generation_config.stop_sequences == ["END"]
    assert isinstance(options.response_format, interaction_types.TextResponseFormat)
    assert options.response_format.model_dump(by_alias=True)["schema"] == {"type": "object"}


@pytest.mark.asyncio
async def test_interaction_options_build_image_and_maps_formats(pipe_instance_fixture):
    pipe, _ = pipe_instance_fixture
    pipe.valves.MAPS_GROUNDING_COORDINATES = "45.5,-73.6"
    options = await pipe._build_interaction_request_options(
        {},
        {
            "canonical_model_id": "gemini-test",
            "features": {"google_maps": True},
        },
        pipe.valves,
        _interaction_policy(image=True),
    )
    assert options.response_format is not None
    assert options.response_format.type == "image"
    assert options.tools[0].type == "google_maps"
    assert options.tools[0].latitude == 45.5


@pytest.mark.asyncio
async def test_task_interaction_options_disable_tools(pipe_instance_fixture):
    pipe, _ = pipe_instance_fixture
    options = await pipe._build_interaction_request_options(
        {},
        {
            "canonical_model_id": "gemini-test",
            "task": "title_generation",
            "features": {"google_search_tool": True},
        },
        pipe.valves,
        _interaction_policy(),
    )
    assert options.tools == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "metadata", "message"),
    [
        ({"top_k": 10}, {"features": {}}, "top_k"),
        (
            {},
            {"features": {}, "merged_custom_params": {"reasoning_effort": 8192}},
            "reasoning_effort",
        ),
        ({}, {"features": {"url_context": True}}, "does not support requested tools"),
    ],
)
async def test_interaction_options_reject_legacy_or_unsupported_configuration(
    pipe_instance_fixture, body, metadata, message
):
    pipe, _ = pipe_instance_fixture
    metadata["canonical_model_id"] = "gemini-test"
    url_context_supported = "url_context" not in metadata.get("features", {})
    policy = _selected_service(
        url_context=url_context_supported,
        external_urls=url_context_supported,
    )
    with pytest.raises(ValueError, match=message):
        await pipe._build_interaction_request_options(body, metadata, pipe.valves, policy)


@pytest.mark.asyncio
async def test_installed_catalog_request_options_match_every_developer_policy(
    pipe_instance_fixture,
) -> None:
    pipe, _ = pipe_instance_fixture
    catalog = Pipe._validate_app_state_catalog(_installed_model_catalog()).payload.runtime_models()
    feature_keys = {
        "google_search": "google_search_tool",
        "code_execution": "google_code_execution",
        "url_context": "url_context",
        "google_maps": "google_maps",
    }

    for model_id, model in catalog.items():
        developer = model.services.developer
        assert isinstance(developer, CatalogSupportedService), model_id
        assert isinstance(model.services.enterprise, CatalogUnavailableService), model_id
        selected = SelectedCatalogService(
            model_id=model_id,
            service="developer",
            policy=developer,
        )
        metadata = cast(Metadata, {"canonical_model_id": model_id, "features": {}})
        options = await pipe._build_interaction_request_options({}, metadata, pipe.valves, selected)
        if "image" in developer.content.outputs:
            assert isinstance(options.response_format, interaction_types.ImageResponseFormat), (
                model_id
            )
        else:
            assert isinstance(options.response_format, interaction_types.TextResponseFormat), (
                model_id
            )

        with pytest.raises(ValueError, match="selected service limit"):
            await pipe._build_interaction_request_options(
                {"max_tokens": developer.limits.output_tokens + 1},
                metadata,
                pipe.valves,
                selected,
            )

        thinking = developer.interactions.thinking
        if thinking.supported:
            for level in thinking.levels:
                reasoning_metadata = cast(
                    Metadata,
                    {
                        "canonical_model_id": model_id,
                        "features": {"reasoning": True},
                        "merged_custom_params": {"reasoning_effort": level},
                    },
                )
                reasoning_options = await pipe._build_interaction_request_options(
                    {}, reasoning_metadata, pipe.valves, selected
                )
                assert reasoning_options.generation_config.thinking_level == level, model_id
            with pytest.raises(ValueError, match="not supported"):
                await pipe._build_interaction_request_options(
                    {},
                    cast(
                        Metadata,
                        {
                            "canonical_model_id": model_id,
                            "features": {"reasoning": True},
                            "merged_custom_params": {"reasoning_effort": "future"},
                        },
                    ),
                    pipe.valves,
                    selected,
                )
        else:
            with pytest.raises(ValueError, match="does not support thinking"):
                await pipe._build_interaction_request_options(
                    {},
                    cast(
                        Metadata,
                        {"canonical_model_id": model_id, "features": {"reasoning": True}},
                    ),
                    pipe.valves,
                    selected,
                )

        for tool_name, feature_key in feature_keys.items():
            tool_metadata = cast(
                Metadata,
                {"canonical_model_id": model_id, "features": {feature_key: True}},
            )
            if getattr(developer.interactions.tools, tool_name):
                tool_options = await pipe._build_interaction_request_options(
                    {}, tool_metadata, pipe.valves, selected
                )
                assert [tool.type for tool in tool_options.tools] == [tool_name], model_id
            else:
                with pytest.raises(ValueError, match="does not support requested tools"):
                    await pipe._build_interaction_request_options(
                        {}, tool_metadata, pipe.valves, selected
                    )

        structured_body = cast(
            Body,
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": {"type": "object"}},
                }
            },
        )
        if developer.interactions.response_format:
            structured = await pipe._build_interaction_request_options(
                structured_body, metadata, pipe.valves, selected
            )
            assert isinstance(structured.response_format, interaction_types.TextResponseFormat)
            assert structured.response_format.mime_type == "application/json"
        else:
            with pytest.raises(ValueError, match="does not support structured output"):
                await pipe._build_interaction_request_options(
                    structured_body, metadata, pipe.valves, selected
                )


def test_endpoint_identity_isolates_model_project_service_and_base_url():
    developer = _developer_identity().for_model("gemini-2.5-flash")
    assert developer.scope == developer.model_copy().scope
    assert developer.scope != developer.for_model("gemini-2.5-pro").scope
    assert (
        developer.scope != developer.model_copy(update={"base_url": "https://proxy.example"}).scope
    )
    assert (
        developer.scope
        != EndpointIdentity(
            service="enterprise",
            credential_fingerprint="test-credential",
            project="project-a",
            location="global",
            api_version="v1beta1",
            model="gemini-2.5-flash",
        ).scope
    )


@pytest.mark.asyncio
async def test_execute_attempt_rejects_endpoint_service_mismatch(pipe_instance_fixture) -> None:
    pipe, _ = pipe_instance_fixture
    request = MagicMock()
    request.app = FastAPI()
    emitter = MagicMock(spec=EventEmitter)
    binding = GenAIClientBinding(client=MagicMock(), identity=_developer_identity())

    with (
        patch.object(pipe, "_get_user_client", return_value=binding),
        pytest.raises(ValueError, match="does not match the initialized endpoint service"),
    ):
        await pipe._execute_generation_attempt(
            tier="enterprise",
            valves=pipe.valves,
            body=cast(Body, {"messages": []}),
            __user__=cast(UserData, {"email": "user@example.test"}),
            __metadata__=cast(Metadata, {"canonical_model_id": "gemini-test"}),
            __request__=request,
            event_emitter=emitter,
            selected_service=_selected_service(service="enterprise"),
        )


@pytest.mark.asyncio
async def test_cached_client_lifecycle_closes_async_resources(pipe_instance_fixture):
    pipe, constructor = pipe_instance_fixture
    client = constructor.return_value
    client.aio.aclose = AsyncMock()
    Pipe._cached_client_bindings = []

    pipe._get_user_client(pipe.valves, USER_EMAIL_REGULAR)
    await Pipe.aclose_cached_clients()

    client.aio.aclose.assert_awaited_once_with()
    assert Pipe._get_or_create_genai_client.cache_info().currsize == 0


def test_logger_capture_redacts_signed_state_credentials_and_locations(
    pipe_instance_fixture,
) -> None:
    pipe, _constructor = pipe_instance_fixture
    companion = object.__new__(CompanionFilter)
    secrets = {
        "api_key": "raw-api-key",
        "authorization": "Bearer raw-token",
        "credential_fingerprint": "credential-hash",
        "uri": "https://signed.example/file?token=raw-url-token",
        "gemini_interaction": {
            "steps": [{"type": "thought", "signature": "opaque-thought-signature"}]
        },
        "nested": {"tool_signature": "opaque-tool-signature", "safe": "visible"},
    }

    captures: list[str] = []
    for formatter in (pipe.plugin_stdout_format, companion.plugin_stdout_format):
        extra = {"payload": secrets}
        record = MagicMock()
        record.__getitem__.side_effect = {"extra": extra}.__getitem__
        formatter(record)
        captures.append(str(extra["_plugin_serialized_data"]))

    captured = "\n".join(captures)
    assert "visible" in captured
    for secret in (
        "raw-api-key",
        "raw-token",
        "credential-hash",
        "raw-url-token",
        "opaque-thought-signature",
        "opaque-tool-signature",
    ):
        assert secret not in captured
    assert captured.count("[REDACTED]") >= 12


@pytest.mark.parametrize(
    ("error", "kind", "retryable"),
    [
        (_client_error(429, "RESOURCE_EXHAUSTED", "quota"), GenerationFailureKind.RATE_LIMIT, True),
        (
            _client_error(403, "PERMISSION_DENIED", "denied"),
            GenerationFailureKind.PERMISSION,
            False,
        ),
        (
            _client_error(401, "UNAUTHENTICATED", "invalid credential"),
            GenerationFailureKind.PERMISSION,
            False,
        ),
        (
            _client_error(400, "INVALID_ARGUMENT", "PUBLIC_PROVIDER_CANARY"),
            GenerationFailureKind.INVALID_REQUEST,
            False,
        ),
        (
            _server_error(503, "UNAVAILABLE", "temporarily unavailable"),
            GenerationFailureKind.UNAVAILABLE,
            True,
        ),
        (httpx.ReadTimeout("timed out"), GenerationFailureKind.TRANSPORT, True),
        (
            InteractionExecutionError(GenerationFailureKind.INTERACTION_STATUS, "budget_exceeded"),
            GenerationFailureKind.INTERACTION_STATUS,
            False,
        ),
        (RuntimeError("unknown failure"), GenerationFailureKind.UNKNOWN, False),
    ],
)
def test_generation_failure_policy_is_typed(error, kind, retryable):
    failure = Pipe._classify_generation_failure(error)
    assert failure.kind is kind
    assert failure.retryable_across_endpoint is retryable


@pytest.mark.parametrize(
    "error",
    [
        _client_error(400, "INVALID_ARGUMENT", "PROVIDER_FAILURE_CANARY"),
        _server_error(503, "UNAVAILABLE", "PROVIDER_FAILURE_CANARY"),
        httpx.ReadError("PROVIDER_FAILURE_CANARY"),
        InteractionExecutionError(
            GenerationFailureKind.INTERACTION_ERROR, "PROVIDER_FAILURE_CANARY"
        ),
        RuntimeError("PROVIDER_FAILURE_CANARY"),
    ],
)
def test_generation_failure_user_details_never_contain_provider_text(error: Exception) -> None:
    failure = Pipe._classify_generation_failure(error)

    assert "PROVIDER_FAILURE_CANARY" not in failure.detail
    assert failure.detail.startswith("Gemini")


def _emitted_text(emissions, kind: str) -> str:
    return "".join(emission.text or "" for emission in emissions if emission.kind == kind)


def _stream_text(chunks: list[dict | str]) -> str:
    return "".join(
        cast(str, chunk["choices"][0]["delta"].get("content", ""))
        for chunk in chunks
        if isinstance(chunk, dict) and chunk.get("choices")
    )


class _FakeInteractionStream:
    def __init__(self, events: list[interaction_types.InteractionSSEEvent], error=None):
        self.events = iter(events)
        self.error = error
        self.error_raised = False
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.events)
        except StopIteration:
            if self.error is not None and not self.error_raised:
                self.error_raised = True
                raise self.error from None
            raise StopAsyncIteration from None

    async def close(self):
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.close()


def test_interaction_reducer_stream_and_unary_are_golden_equivalent():
    steps = [
        interaction_types.ModelOutputStep(
            type="model_output",
            content=[interaction_types.TextContent(type="text", text="answer")],
        ),
        interaction_types.ThoughtStep(
            type="thought",
            signature=base64.b64encode(b"opaque").decode("ascii"),
            summary=[interaction_types.TextContent(type="text", text="reason")],
        ),
    ]
    usage = interaction_types.Usage(
        total_input_tokens=10,
        total_output_tokens=5,
        total_thought_tokens=2,
        total_cached_tokens=3,
        total_tool_use_tokens=1,
        total_tokens=18,
    )
    unary = InteractionReducer()
    unary_emissions = unary.consume_interaction(
        interaction_types.Interaction(
            id="interaction-1", status="completed", steps=steps, usage=usage
        )
    )
    unary.finalize_steps()

    streamed = InteractionReducer()
    stream_emissions = []
    stream_emissions += streamed.consume_event(
        interaction_types.InteractionCreatedEvent(
            event_id="e1",
            interaction=interaction_types.InteractionSseEventInteraction(
                id="interaction-1", status="in_progress"
            ),
        )
    )
    stream_emissions += streamed.consume_event(
        interaction_types.StepStart(
            event_id="e2",
            index=0,
            step=interaction_types.ModelOutputStep(type="model_output", content=[]),
        )
    )
    stream_emissions += streamed.consume_event(
        interaction_types.StepDelta(
            event_id="e3", index=0, delta=interaction_types.TextDelta(text="answer")
        )
    )
    streamed.consume_event(interaction_types.StepStop(event_id="e4", index=0))
    stream_emissions += streamed.consume_event(
        interaction_types.StepStart(
            event_id="e5",
            index=1,
            step=interaction_types.ThoughtStep(type="thought"),
        )
    )
    stream_emissions += streamed.consume_event(
        interaction_types.StepDelta(
            event_id="e6",
            index=1,
            delta=interaction_types.ThoughtSummaryDelta(
                content=interaction_types.TextContent(type="text", text="reason")
            ),
        )
    )
    streamed.consume_event(interaction_types.StepStop(event_id="e7", index=1))
    stream_emissions += streamed.consume_event(
        interaction_types.InteractionCompletedEvent(
            event_id="e8",
            interaction=interaction_types.InteractionSseEventInteraction(
                id="interaction-1", status="completed", steps=steps, usage=usage
            ),
        )
    )
    streamed.finalize_steps()

    assert _emitted_text(stream_emissions, "content") == _emitted_text(unary_emissions, "content")
    assert _emitted_text(stream_emissions, "reasoning") == _emitted_text(
        unary_emissions, "reasoning"
    )
    assert streamed.state.steps == unary.state.steps
    assert streamed.state.usage == unary.state.usage
    assert streamed.state.status == unary.state.status == "completed"


def test_interaction_reducer_dedupes_events_and_rejects_illegal_lifecycle():
    reducer = InteractionReducer()
    start = interaction_types.StepStart(
        event_id="same",
        index=0,
        step=interaction_types.ModelOutputStep(type="model_output", content=[]),
    )
    reducer.consume_event(start)
    assert reducer.consume_event(start) == []
    with pytest.raises(InteractionExecutionError, match="inactive step"):
        reducer.consume_event(
            interaction_types.StepDelta(
                event_id="new", index=1, delta=interaction_types.TextDelta(text="bad")
            )
        )
    reducer.consume_event(interaction_types.StepStop(event_id="stop", index=0))
    with pytest.raises(InteractionExecutionError, match="Invalid step.stop"):
        reducer.consume_event(interaction_types.StepStop(event_id="stop-again", index=0))


@pytest.mark.parametrize("status", ["failed", "cancelled", "incomplete", "budget_exceeded"])
def test_interaction_reducer_terminal_failure_statuses_are_not_success(status: str):
    reducer = InteractionReducer()
    with pytest.raises(InteractionExecutionError, match=status):
        reducer.consume_event(
            interaction_types.InteractionStatusUpdate(
                event_id="terminal",
                interaction_id="interaction",
                status=cast(interaction_types.InteractionStatusUpdateStatus, status),
            )
        )
    assert reducer.state.terminal is True
    assert reducer.state.status == status


def test_interaction_reducer_error_unknown_event_step_content_and_delta_fail_closed():
    with pytest.raises(InteractionExecutionError, match="Interaction error event") as error_info:
        InteractionReducer().consume_event(
            interaction_types.ErrorEvent(
                event_id="error",
                error=interaction_types.Error(code="500", message="PROVIDER_EVENT_CANARY"),
            )
        )
    assert "PROVIDER_EVENT_CANARY" not in str(error_info.value)

    unknown_canary = "UNKNOWN_PROVIDER_PAYLOAD_CANARY"
    unknown_event = interaction_types.UnknownInteractionSSEEvent(raw={"event_type": unknown_canary})
    with pytest.raises(
        InteractionExecutionError, match="unknown Interaction event"
    ) as unknown_info:
        InteractionReducer().consume_event(unknown_event)
    assert unknown_canary not in str(unknown_info.value)

    model_output_canary = "MODEL_OUTPUT_PROVIDER_CANARY"
    with pytest.raises(InteractionExecutionError, match="model-output error") as model_output_info:
        InteractionReducer().consume_interaction(
            interaction_types.Interaction(
                status="completed",
                steps=cast(
                    list[interaction_types.Step],
                    [
                        interaction_types.ModelOutputStep(
                            type="model_output",
                            error=interaction_types.Status(code=500, message=model_output_canary),
                        )
                    ],
                ),
            )
        )
    assert model_output_canary not in str(model_output_info.value)

    unknown_step_reducer = InteractionReducer()
    with pytest.raises(InteractionExecutionError, match="unknown Interaction step"):
        unknown_step_reducer.consume_interaction(
            interaction_types.Interaction(
                status="completed",
                steps=[interaction_types.UnknownStep(raw={"type": "future"})],
            )
        )

    unknown_content_reducer = InteractionReducer()
    with pytest.raises(InteractionExecutionError, match="unknown Interaction content"):
        unknown_content_reducer.consume_interaction(
            interaction_types.Interaction(
                status="completed",
                steps=[
                    interaction_types.ModelOutputStep(
                        type="model_output",
                        content=[interaction_types.UnknownContent(raw={"type": "future"})],
                    )
                ],
            )
        )

    unknown_delta_reducer = InteractionReducer()
    unknown_delta_reducer.consume_event(
        interaction_types.StepStart(
            index=0,
            step=interaction_types.ModelOutputStep(type="model_output", content=[]),
        )
    )
    with pytest.raises(InteractionExecutionError, match="unknown Interaction delta"):
        unknown_delta_reducer.consume_event(
            interaction_types.StepDelta(
                index=0,
                delta=interaction_types.UnknownStepDeltaData(raw={"type": "future"}),
            )
        )


def test_interaction_reducer_media_tool_source_and_usage_mapping():
    reducer = InteractionReducer()
    steps: list[interaction_types.Step] = [
        interaction_types.ModelOutputStep(
            type="model_output",
            content=[
                interaction_types.ImageContent(type="image", mime_type="image/png", data="aW1hZ2U=")
            ],
        ),
        interaction_types.FunctionCallStep(
            type="function_call", id="call", name="lookup", arguments={"q": "x"}
        ),
        interaction_types.GoogleSearchResultStep(
            type="google_search_result", call_id="search-call", result=[]
        ),
    ]
    emissions = reducer.consume_interaction(
        interaction_types.Interaction(
            id="interaction",
            status="completed",
            usage=interaction_types.Usage(
                total_input_tokens=100,
                total_output_tokens=20,
                total_thought_tokens=5,
                total_cached_tokens=40,
                total_tool_use_tokens=3,
                total_tokens=128,
            ),
            steps=steps,
        )
    )

    assert {emission.kind for emission in emissions} >= {"media", "tool", "source", "status"}
    assert reducer.state.usage == InteractionsSDKBoundary.normalize_usage(
        interaction_types.Usage(
            total_input_tokens=100,
            total_output_tokens=20,
            total_thought_tokens=5,
            total_cached_tokens=40,
            total_tool_use_tokens=3,
            total_tokens=128,
        )
    )
    reducer.finalize_steps()
    assert reducer.state.grounding.tool_records[0].tool == "google_search"


def test_interactions_usage_normalizes_modality_breakdowns():
    usage = interaction_types.Usage(
        total_input_tokens=12,
        total_output_tokens=8,
        total_tokens=20,
        output_tokens_by_modality=[interaction_types.ModalityTokens(modality="image", tokens=3)],
    )

    normalized = InteractionsSDKBoundary.normalize_usage(usage)

    assert normalized.output_by_modality == [{"modality": "image", "tokens": 3}]


@pytest.mark.parametrize(
    ("is_paid_api", "include_image_price", "expected"),
    [
        (
            True,
            True,
            {
                "input_cost": 0.00008,
                "cache_cost": 0.000002,
                "output_cost": 0.00009,
                "image_output_cost": 0.0003,
                "known_cost": 0.000472,
                "total_cost": 0.000472,
                "is_complete": True,
                "unpriced_tokens": 0,
            },
        ),
        (
            True,
            False,
            {
                "input_cost": 0.00008,
                "cache_cost": 0.000002,
                "output_cost": 0.00009,
                "image_output_cost": None,
                "known_cost": 0.000172,
                "total_cost": None,
                "is_complete": False,
                "unpriced_tokens": 10,
            },
        ),
        (
            False,
            True,
            {
                "input_cost": 0.0,
                "cache_cost": 0.0,
                "output_cost": 0.0,
                "image_output_cost": 0.0,
                "known_cost": 0.0,
                "total_cost": 0.0,
                "is_complete": True,
                "unpriced_tokens": 0,
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_interactions_usage_costs_image_output_once(
    pipe_instance_fixture,
    is_paid_api: bool,
    include_image_price: bool,
    expected: dict[str, object],
):
    pipe, _ = pipe_instance_fixture
    selected = _selected_service(image=include_image_price)
    metadata = cast(
        Metadata,
        {
            "canonical_model_id": "gemini-test",
            "is_paid_api": is_paid_api,
            "gemini_catalog_service": "developer",
        },
    )
    usage = NormalizedInteractionUsage(
        input_tokens=100,
        output_tokens=50,
        thought_tokens=5,
        cached_tokens=20,
        total_tokens=155,
        input_by_modality=[{"modality": "text", "tokens": 100}],
        cached_by_modality=[{"modality": "text", "tokens": 20}],
        output_by_modality=[{"modality": "image", "tokens": 10}],
    )

    result = pipe._get_interaction_usage_data(usage, selected, metadata, 0.0)

    assert result["cost_details"] == expected


def test_usage_pricing_is_resolved_from_completed_attempt_service(pipe_instance_fixture):
    pipe, _ = pipe_instance_fixture
    developer_selected = _selected_service(free_tier=False)
    enterprise_selected = _selected_service(
        service="enterprise",
        free_tier=False,
        input_price=10.0,
        output_price=20.0,
    )
    usage = NormalizedInteractionUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        input_by_modality=[{"modality": "text", "tokens": 100}],
    )

    developer = pipe._get_interaction_usage_data(
        usage,
        developer_selected,
        cast(
            Metadata,
            {
                "canonical_model_id": "gemini-test",
                "is_paid_api": True,
                "gemini_catalog_service": "developer",
            },
        ),
        0.0,
    )
    enterprise_result = pipe._get_interaction_usage_data(
        usage,
        enterprise_selected,
        cast(
            Metadata,
            {
                "canonical_model_id": "gemini-test",
                "is_paid_api": True,
                # Stale observability metadata must not override the completed selection.
                "gemini_catalog_service": "developer",
            },
        ),
        0.0,
    )

    assert cast(dict, developer["cost_details"])["total_cost"] == 0.0002
    assert cast(dict, enterprise_result["cost_details"])["total_cost"] == 0.002


def test_whole_prompt_pricing_selects_one_rate_at_threshold_boundary() -> None:
    rate = CatalogPricedRate.model_validate(
        {
            "state": "priced",
            "tiers": [
                {"up_to_prompt_tokens": 200_000, "price_per_million": 2.0},
                {"up_to_prompt_tokens": None, "price_per_million": 4.0},
            ],
        }
    )
    unknown = CatalogUnpricedRate(state="unpriced", reason="No exact price is published.")

    assert Pipe._calculate_cost(100_000, 200_000, rate) == 0.2
    assert Pipe._calculate_cost(100_000, 200_001, rate) == 0.4
    assert Pipe._calculate_cost(100_000, 200_000, unknown) is None


def test_all_catalog_models_account_every_modality_cache_and_unknown_price(
    pipe_instance_fixture,
) -> None:
    pipe, _ = pipe_instance_fixture
    catalog_path = Path(__file__).parents[1] / "plugins" / "pipes" / "gemini_models.yaml"
    catalog = AppStateModelCatalog.model_validate(
        yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    )

    for model_id, model in catalog.runtime_models().items():
        policy = model.services.developer
        assert isinstance(policy, CatalogSupportedService)
        selected = SelectedCatalogService(
            model_id=model_id,
            service="developer",
            policy=policy,
        )
        input_modalities = sorted(policy.content.inputs)
        output_modalities = sorted(policy.content.outputs)
        usage = NormalizedInteractionUsage(
            input_tokens=10 * len(input_modalities),
            output_tokens=5 * len(output_modalities),
            cached_tokens=len(input_modalities),
            total_tokens=15 * len(input_modalities) + 5 * len(output_modalities),
            input_by_modality=[
                {"modality": modality, "tokens": 10} for modality in input_modalities
            ],
            cached_by_modality=[
                {"modality": modality, "tokens": 1} for modality in input_modalities
            ],
            output_by_modality=[
                {"modality": modality, "tokens": 5} for modality in output_modalities
            ],
        )
        expected_unknown = sum(
            9
            for modality in input_modalities
            if isinstance(policy.pricing.input[modality], CatalogUnpricedRate)
        ) + sum(
            1
            for modality in input_modalities
            if isinstance(policy.pricing.cached_input[modality], CatalogUnpricedRate)
        )

        result = pipe._get_interaction_usage_data(
            usage,
            selected,
            cast(
                Metadata,
                {
                    "canonical_model_id": model_id,
                    "is_paid_api": True,
                    "gemini_catalog_service": "developer",
                },
            ),
            0.0,
        )
        costs = cast(dict[str, object], result["cost_details"])
        assert costs["unpriced_tokens"] == expected_unknown, model_id
        assert costs["is_complete"] is (expected_unknown == 0), model_id
        assert (costs["total_cost"] is None) is (expected_unknown > 0), model_id


def test_tool_use_tokens_remain_explicitly_unpriced(pipe_instance_fixture) -> None:
    pipe, _ = pipe_instance_fixture
    usage = NormalizedInteractionUsage(
        input_tokens=10,
        output_tokens=5,
        tool_use_tokens=3,
        total_tokens=18,
        input_by_modality=[{"modality": "text", "tokens": 10}],
        output_by_modality=[{"modality": "text", "tokens": 5}],
    )

    result = pipe._get_interaction_usage_data(
        usage,
        _selected_service(),
        cast(
            Metadata,
            {
                "canonical_model_id": "gemini-test",
                "is_paid_api": True,
                "gemini_catalog_service": "developer",
            },
        ),
        0.0,
    )

    costs = cast(dict[str, object], result["cost_details"])
    assert costs["unpriced_tokens"] == 3
    assert costs["known_cost"] is not None
    assert costs["total_cost"] is None
    assert costs["is_complete"] is False


def test_interactions_migration_removal_guard():
    source = (Path(__file__).parents[1] / "plugins" / "pipes" / "gemini_manifold.py").read_text(
        encoding="utf-8"
    )
    forbidden = [
        "generate_content",
        "GenerateContentResponse",
        "_unified_response_processor",
        "_process_part",
        "_get_usage_data",
        "gemini_parts",
        "usage_metadata",
        ".candidates",
        "types.Candidate",
        '"interaction_steps"',
        '"interaction_original_content"',
        '"interaction_endpoint_scope"',
        '"interaction_schema_version"',
    ]

    assert [name for name in forbidden if name in source] == []
    type_source = (Path(__file__).parents[1] / "utils" / "manifold_types.py").read_text(
        encoding="utf-8"
    )
    assert "gemini_parts" not in type_source


@pytest.mark.parametrize(
    "step",
    [
        interaction_types.UserInputStep.model_construct(type="user_input", content=[]),
        interaction_types.ModelOutputStep.model_construct(type="model_output", content=[]),
        interaction_types.ThoughtStep.model_construct(type="thought", summary=[]),
        interaction_types.FunctionCallStep.model_construct(
            type="function_call", id="call", name="fn", arguments={}
        ),
        interaction_types.FunctionResultStep.model_construct(
            type="function_result", call_id="call", result="ok"
        ),
        interaction_types.CodeExecutionCallStep.model_construct(
            type="code_execution_call",
            arguments=interaction_types.CodeExecutionCallArguments.model_construct(code=None),
        ),
        interaction_types.CodeExecutionResultStep.model_construct(
            type="code_execution_result", result=""
        ),
        interaction_types.URLContextCallStep.model_construct(type="url_context_call"),
        interaction_types.URLContextResultStep.model_construct(type="url_context_result"),
        interaction_types.GoogleSearchCallStep.model_construct(type="google_search_call"),
        interaction_types.GoogleSearchResultStep.model_construct(type="google_search_result"),
        interaction_types.FileSearchCallStep.model_construct(type="file_search_call"),
        interaction_types.FileSearchResultStep.model_construct(type="file_search_result"),
        interaction_types.GoogleMapsCallStep.model_construct(type="google_maps_call"),
        interaction_types.GoogleMapsResultStep.model_construct(type="google_maps_result"),
        interaction_types.MCPServerToolCallStep.model_construct(type="mcp_server_tool_call"),
        interaction_types.MCPServerToolResultStep.model_construct(type="mcp_server_tool_result"),
    ],
    ids=lambda step: str(step.type),
)
def test_interaction_reducer_accepts_every_exact_step_variant(step):
    reducer = InteractionReducer()
    reducer.consume_interaction(interaction_types.Interaction(status="completed", steps=[step]))
    reducer.finalize_steps()
    assert reducer.state.steps[0]["type"] == step.type


@pytest.mark.parametrize(
    "delta",
    [
        interaction_types.TextDelta.model_construct(type="text", text="text"),
        interaction_types.ImageDelta.model_construct(type="image", data="aQ=="),
        interaction_types.AudioDelta.model_construct(type="audio", data="YQ=="),
        interaction_types.DocumentDelta.model_construct(type="document", data="ZA=="),
        interaction_types.VideoDelta.model_construct(type="video", data="dg=="),
        interaction_types.ThoughtSummaryDelta.model_construct(type="thought_summary", content=None),
        interaction_types.ThoughtSignatureDelta.model_construct(
            type="thought_signature", signature="c2ln"
        ),
        interaction_types.TextAnnotationDelta.model_construct(
            type="text_annotation_delta", annotations=[]
        ),
        interaction_types.ArgumentsDelta.model_construct(type="arguments_delta", arguments='{"x":'),
        interaction_types.CodeExecutionCallDelta.model_construct(type="code_execution_call"),
        interaction_types.URLContextCallDelta.model_construct(type="url_context_call"),
        interaction_types.GoogleSearchCallDelta.model_construct(type="google_search_call"),
        interaction_types.MCPServerToolCallDelta.model_construct(type="mcp_server_tool_call"),
        interaction_types.FileSearchCallDelta.model_construct(type="file_search_call"),
        interaction_types.GoogleMapsCallDelta.model_construct(type="google_maps_call"),
        interaction_types.RetrievalCallDelta(
            type="retrieval_call",
            arguments=interaction_types.RetrievalCallArguments(queries=[]),
        ),
        interaction_types.CodeExecutionResultDelta.model_construct(type="code_execution_result"),
        interaction_types.URLContextResultDelta.model_construct(type="url_context_result"),
        interaction_types.GoogleSearchResultDelta.model_construct(type="google_search_result"),
        interaction_types.MCPServerToolResultDelta.model_construct(type="mcp_server_tool_result"),
        interaction_types.FileSearchResultDelta.model_construct(type="file_search_result"),
        interaction_types.GoogleMapsResultDelta.model_construct(type="google_maps_result"),
        interaction_types.RetrievalResultDelta.model_construct(type="retrieval_result"),
        interaction_types.FunctionResultDelta.model_construct(type="function_result"),
    ],
    ids=lambda delta: str(delta.type),
)
def test_interaction_reducer_accepts_every_exact_delta_variant(delta):
    reducer = InteractionReducer()
    start_step = (
        interaction_types.FunctionCallStep(type="function_call", id="call", name="fn", arguments={})
        if isinstance(delta, interaction_types.ArgumentsDelta)
        else interaction_types.ModelOutputStep(type="model_output", content=[])
    )
    reducer.consume_event(
        interaction_types.StepStart(
            index=0,
            step=start_step,
        )
    )
    emissions = reducer.consume_event(interaction_types.StepDelta(index=0, delta=delta))
    if delta.type in {"image", "audio", "document", "video"}:
        assert emissions == []
    assert reducer.state.terminal is False


def test_interaction_reducer_annotations_and_tool_calls_are_correlated_equally():
    annotation = interaction_types.URLCitation(
        type="url_citation", url="https://example.test", title="Example"
    )
    steps: list[interaction_types.Step] = [
        interaction_types.ModelOutputStep(
            type="model_output",
            content=[
                interaction_types.TextContent(type="text", text="cited", annotations=[annotation])
            ],
        ),
        interaction_types.GoogleSearchCallStep.model_construct(
            type="google_search_call", id="search-1"
        ),
        interaction_types.GoogleSearchResultStep.model_construct(
            type="google_search_result", call_id="search-1", result=[]
        ),
    ]
    reducer = InteractionReducer()
    reducer.consume_interaction(interaction_types.Interaction(status="completed", steps=steps))
    reducer.finalize_steps()

    assert reducer.state.grounding.sources[0].kind == "url"
    assert reducer.state.grounding.tool_records[0].phase == "call"
    assert reducer.state.grounding.tool_records[0].call_id == "search-1"
    assert reducer.state.grounding.tool_records[1].phase == "result"
    assert reducer.state.grounding.tool_records[1].call_id == "search-1"


@pytest.mark.parametrize("include_annotation_delta", [False, True])
def test_grounding_reducer_final_annotations_are_preserved_and_deduplicated(
    include_annotation_delta: bool,
) -> None:
    annotation = interaction_types.URLCitation(
        type="url_citation",
        url="https://example.test/source",
        title="Source",
        start_index=0,
        end_index=5,
    )
    reducer = InteractionReducer()
    reducer.consume_event(
        interaction_types.StepStart(
            index=0,
            step=interaction_types.ModelOutputStep(type="model_output", content=[]),
        )
    )
    reducer.consume_event(
        interaction_types.StepDelta(
            index=0,
            delta=interaction_types.TextDelta(type="text", text="cited"),
        )
    )
    if include_annotation_delta:
        reducer.consume_event(
            interaction_types.StepDelta(
                index=0,
                delta=interaction_types.TextAnnotationDelta(
                    type="text_annotation_delta", annotations=[annotation]
                ),
            )
        )
    reducer.consume_event(
        interaction_types.InteractionCompletedEvent(
            event_id="final",
            interaction=interaction_types.InteractionSseEventInteraction(
                id="interaction",
                status="completed",
                steps=[
                    interaction_types.ModelOutputStep(
                        type="model_output",
                        content=[
                            interaction_types.TextContent(
                                type="text", text="cited", annotations=[annotation]
                            )
                        ],
                    )
                ],
            ),
        )
    )
    reducer.finalize_steps()

    assert [source.uri for source in reducer.state.grounding.sources] == [
        "https://example.test/source"
    ]
    assert len(reducer.state.grounding.citations) == 1
    assert reducer.state.original_content == "cited"


def test_grounding_envelope_is_identical_for_stream_and_unary_snapshots() -> None:
    annotation = interaction_types.URLCitation(
        type="url_citation",
        url="https://example.test/source",
        start_index=0,
        end_index=5,
    )
    steps: list[interaction_types.Step] = [
        interaction_types.ModelOutputStep(
            type="model_output",
            content=[
                interaction_types.TextContent(type="text", text="cited", annotations=[annotation])
            ],
        ),
        interaction_types.GoogleSearchCallStep(
            type="google_search_call",
            id="search",
            arguments=interaction_types.GoogleSearchCallArguments(queries=["query"]),
        ),
        interaction_types.GoogleSearchResultStep(
            type="google_search_result",
            call_id="search",
            result=[interaction_types.GoogleSearchResult(search_suggestions="suggestion")],
        ),
    ]
    unary = InteractionReducer()
    unary.consume_interaction(interaction_types.Interaction(status="completed", steps=steps))
    unary.finalize_steps()

    streamed = InteractionReducer()
    streamed.consume_event(
        interaction_types.StepStart(
            index=0,
            step=interaction_types.ModelOutputStep(type="model_output", content=[]),
        )
    )
    streamed.consume_event(
        interaction_types.StepDelta(
            index=0, delta=interaction_types.TextDelta(type="text", text="cited")
        )
    )
    streamed.consume_event(
        interaction_types.InteractionCompletedEvent(
            event_id="complete",
            interaction=interaction_types.InteractionSseEventInteraction(
                id="interaction",
                status="completed",
                steps=steps,
            ),
        )
    )
    streamed.finalize_steps()

    assert streamed.state.grounding == unary.state.grounding


def test_grounding_reducer_normalizes_exact_tools_without_signatures() -> None:
    secret = "provider-signature-must-not-cross-envelope"
    steps: list[interaction_types.Step] = [
        interaction_types.GoogleSearchCallStep(
            type="google_search_call",
            id="search",
            arguments=interaction_types.GoogleSearchCallArguments(queries=["query"]),
            search_type="web_search",
            signature=secret,
        ),
        interaction_types.GoogleSearchResultStep(
            type="google_search_result",
            call_id="search",
            result=[interaction_types.GoogleSearchResult(search_suggestions="try this")],
            signature=secret,
        ),
        interaction_types.URLContextCallStep(
            type="url_context_call",
            id="url",
            arguments=interaction_types.URLContextCallArguments(urls=["https://example.test"]),
            signature=secret,
        ),
        interaction_types.URLContextResultStep(
            type="url_context_result",
            call_id="url",
            result=[
                interaction_types.URLContextResult(url="https://example.test", status="paywall")
            ],
            is_error=True,
            signature=secret,
        ),
        interaction_types.GoogleMapsCallStep(
            type="google_maps_call",
            id="maps",
            arguments=interaction_types.GoogleMapsCallArguments(queries=["coffee"]),
            signature=secret,
        ),
        interaction_types.GoogleMapsResultStep(
            type="google_maps_result",
            call_id="maps",
            result=[
                interaction_types.GoogleMapsResult(
                    widget_context_token="widget",
                    places=[
                        interaction_types.GoogleMapsResultPlaces(
                            name="Cafe", place_id="place", url="https://maps.example/place"
                        )
                    ],
                )
            ],
            signature=secret,
        ),
    ]
    reducer = InteractionReducer()
    reducer.consume_interaction(interaction_types.Interaction(status="completed", steps=steps))
    reducer.consume_event(
        interaction_types.StepStart(
            index=len(steps),
            step=interaction_types.ModelOutputStep(type="model_output", content=[]),
        )
    )
    reducer.consume_event(
        interaction_types.StepDelta(
            index=len(steps),
            delta=interaction_types.RetrievalCallDelta(
                type="retrieval_call",
                arguments=interaction_types.RetrievalCallArguments(queries=["private docs"]),
                retrieval_type="vertex_ai_search",
                signature=secret,
            ),
        )
    )
    reducer.consume_event(
        interaction_types.StepDelta(
            index=len(steps),
            delta=interaction_types.RetrievalResultDelta(
                type="retrieval_result", is_error=True, signature=secret
            ),
        )
    )
    reducer.finalize_steps()

    envelope = reducer.state.grounding
    assert [(record.tool, record.phase) for record in envelope.tool_records] == [
        ("google_search", "call"),
        ("google_search", "result"),
        ("url_context", "call"),
        ("url_context", "result"),
        ("google_maps", "call"),
        ("google_maps", "result"),
        ("retrieval", "call"),
        ("retrieval", "result"),
    ]
    assert envelope.tool_records[1].search_suggestions == ["try this"]
    assert envelope.tool_records[3].statuses == ["paywall"]
    assert envelope.tool_records[5].widget_context_tokens == ["widget"]
    assert envelope.tool_records[6].step_index == len(steps)
    serialized = envelope.model_dump_json()
    assert "signature" not in serialized
    assert secret not in serialized


def test_companion_multiblock_citations_fail_safe_for_provider_unicode() -> None:
    envelope = CompanionGroundingEnvelope.model_validate(
        {
            "protocol_version": 1,
            "visible_content_sha256": "unused",
            "grounded_text_sha256": "unused",
            "text_blocks": [
                {"step_index": 0, "content_index": 0, "text": "alpha"},
                {"step_index": 2, "content_index": 0, "text": "café"},
            ],
            "sources": [{"id": "url:one", "kind": "url", "uri": "https://example.test"}],
            "citations": [
                {
                    "source_id": "url:one",
                    "block_index": 0,
                    "start": 0,
                    "end": 5,
                    "index_unit": "provider",
                },
                {
                    "source_id": "url:one",
                    "block_index": 1,
                    "start": 0,
                    "end": 5,
                    "index_unit": "provider",
                },
                {
                    "source_id": "url:one",
                    "block_index": 1,
                    "start": 0,
                    "end": 4,
                    "index_unit": "unicode_codepoints",
                },
            ],
        }
    )

    cited, warnings = CompanionFilter._insert_citation_markers(envelope, "alpha <media> café")

    assert cited == "alpha[1] <media> café[1]"
    assert warnings == 1


def test_companion_overlapping_and_duplicate_citations_are_deterministic() -> None:
    envelope = CompanionGroundingEnvelope.model_validate(
        {
            "protocol_version": 1,
            "visible_content_sha256": "unused",
            "grounded_text_sha256": "unused",
            "text_blocks": [{"step_index": 0, "content_index": 0, "text": "answer"}],
            "sources": [
                {"id": "url:one", "kind": "url", "uri": "https://one.example"},
                {"id": "url:two", "kind": "url", "uri": "https://two.example"},
            ],
            "citations": [
                {
                    "source_id": source_id,
                    "block_index": 0,
                    "start": start,
                    "end": 6,
                    "index_unit": "unicode_codepoints",
                }
                for source_id, start in (("url:one", 0), ("url:one", 0), ("url:two", 2))
            ],
        }
    )

    cited, warnings = CompanionFilter._insert_citation_markers(envelope, "answer")

    assert cited == "answer[1][2]"
    assert warnings == 0


@pytest.mark.asyncio
async def test_companion_resolves_only_known_redirect_sources() -> None:
    companion = CompanionFilter()
    envelope = CompanionGroundingEnvelope.model_validate(
        {
            "protocol_version": 1,
            "visible_content_sha256": "unused",
            "grounded_text_sha256": "unused",
            "sources": [
                {
                    "id": "url:redirect",
                    "kind": "url",
                    "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/token",
                },
                {
                    "id": "url:direct",
                    "kind": "url",
                    "uri": "https://example.test/direct",
                },
            ],
        }
    )
    emitter = MagicMock(spec=EventEmitter)
    with patch.object(
        companion,
        "_resolve_url",
        AsyncMock(return_value=("https://resolved.example/source", True)),
    ) as resolve:
        await companion._emit_grounding_sources(envelope, emitter)

    resolve.assert_awaited_once()
    payload = emitter.emit_sources.call_args.args[0]
    assert [item["source"] for item in payload["metadata"]] == [
        "https://resolved.example/source",
        "https://example.test/direct",
    ]


@pytest.mark.asyncio
async def test_companion_url_and_catalog_failures_redact_sources_and_exceptions() -> None:
    companion = CompanionFilter()
    signed_url = "https://private.example/path?token=SIGNED_URL_CANARY"
    provider_canary = "CATALOG_PROVIDER_CANARY"
    captured: list[str] = []
    handler_id = companion_module.log.add(
        lambda message: captured.append(str(message)), format="{message}"
    )
    try:
        session = MagicMock()
        session.get.side_effect = RuntimeError(provider_canary)
        resolved, success = await companion._resolve_url(session, signed_url)
        assert (resolved, success) == (signed_url, False)

        CompanionFilter._load_model_config.cache_clear()
        with (
            patch(
                "plugins.filters.gemini_manifold_companion.urllib.request.urlopen",
                side_effect=RuntimeError(provider_canary),
            ),
            pytest.raises(ModelCatalogError) as error_info,
        ):
            CompanionFilter._load_model_config(signed_url)
    finally:
        companion_module.log.remove(handler_id)
        CompanionFilter._load_model_config.cache_clear()

    observable = "\n".join(captured) + str(error_info.value)
    assert "SIGNED_URL_CANARY" not in observable
    assert provider_canary not in observable
    assert str(error_info.value) == "Gemini model catalog is unavailable or invalid."


@pytest.mark.asyncio
async def test_companion_digest_prevents_duplicate_or_edited_grounding() -> None:
    visible = "cited"
    envelope = {
        "protocol_version": 1,
        "visible_content_sha256": hashlib.sha256(visible.encode()).hexdigest(),
        "grounded_text_sha256": hashlib.sha256(visible.encode()).hexdigest(),
        "text_blocks": [{"step_index": 0, "content_index": 0, "text": visible}],
        "sources": [
            {
                "id": "url:one",
                "kind": "url",
                "uri": "https://example.test/source",
                "title": "Source",
            }
        ],
        "citations": [
            {
                "source_id": "url:one",
                "block_index": 0,
                "start": 0,
                "end": 5,
                "index_unit": "provider",
            }
        ],
    }
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": visible,
                "gemini_interaction": {"grounding": envelope},
            }
        ]
    }
    events: list[dict[str, object]] = []

    async def collect(event: dict[str, object]) -> None:
        events.append(event)

    companion = CompanionFilter()
    await companion.outlet(body, MagicMock(), {}, collect)  # type: ignore[arg-type]
    await companion.outlet(body, MagicMock(), {}, collect)  # type: ignore[arg-type]

    assert body["messages"][-1]["content"] == "cited[1]"
    assert [event["type"] for event in events].count("source") == 1
    assert any(
        event["type"] == "status"
        and "edited" in str(cast(dict[str, object], event["data"])["description"])
        for event in events
    )


@pytest.mark.asyncio
async def test_companion_loads_grounding_from_durable_chat_across_workers() -> None:
    grounding = {
        "protocol_version": 1,
        "visible_content_sha256": hashlib.sha256(b"answer").hexdigest(),
        "grounded_text_sha256": hashlib.sha256(b"answer").hexdigest(),
        "text_blocks": [{"step_index": 0, "content_index": 0, "text": "answer"}],
    }
    chat = MagicMock()
    chat.chat = {
        "history": {"messages": {"message": {"gemini_interaction": {"grounding": grounding}}}}
    }
    with patch(
        "plugins.filters.gemini_manifold_companion.Chats.get_chat_by_id_and_user_id",
        new=AsyncMock(return_value=chat),
    ):
        loaded = await CompanionFilter._load_grounding_envelope(
            cast(Body, {"messages": [{"role": "assistant", "content": "answer"}]}),
            {"chat_id": "chat", "message_id": "message", "user_id": "user"},
        )

    assert loaded is not None
    assert loaded.visible_content_sha256 == grounding["visible_content_sha256"]


@pytest.mark.asyncio
async def test_request_local_pipe_emitters_are_concurrency_isolated() -> None:
    first_events: list[Event] = []
    second_events: list[Event] = []

    async def first(event: Event) -> None:
        first_events.append(event)

    async def second(event: Event) -> None:
        second_events.append(event)

    first_emitter = PipeEventEmitter(first)
    second_emitter = PipeEventEmitter(second)
    first_emitter.emit_status("first", done=True)
    second_emitter.emit_status("second", done=True)
    await asyncio.gather(first_emitter.flush(), second_emitter.flush())
    await asyncio.gather(first_emitter.shutdown(), second_emitter.shutdown())

    assert [cast(dict[str, object], event["data"])["description"] for event in first_events] == [
        "first"
    ]
    assert [cast(dict[str, object], event["data"])["description"] for event in second_events] == [
        "second"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("emitter_class", [EventEmitter, PipeEventEmitter])
@pytest.mark.parametrize("callback_error", [None, RuntimeError, asyncio.CancelledError])
async def test_request_local_emitters_finish_after_callback_success_or_error(
    emitter_class: type[EventEmitter] | type[PipeEventEmitter],
    callback_error: type[BaseException] | None,
) -> None:
    delivered: list[dict[str, object]] = []

    async def callback(event: dict[str, object]) -> None:
        if callback_error is not None:
            raise callback_error()
        delivered.append(event)

    emitter = emitter_class(callback)  # type: ignore[arg-type]
    emitter.emit_status("done", done=True)
    await asyncio.wait_for(emitter.flush(), timeout=1)
    await asyncio.wait_for(emitter.shutdown(), timeout=1)

    if callback_error is None:
        assert len(delivered) == 1


def _completed_event(event_id: str = "completed"):
    return interaction_types.InteractionCompletedEvent(
        event_id=event_id,
        interaction=interaction_types.InteractionSseEventInteraction(
            id="interaction-stream",
            status="completed",
            steps=[
                interaction_types.ModelOutputStep(
                    type="model_output",
                    content=[interaction_types.TextContent(type="text", text="done")],
                )
            ],
            usage=interaction_types.Usage(
                total_input_tokens=2, total_output_tokens=1, total_tokens=3
            ),
        ),
    )


@pytest.mark.asyncio
async def test_interaction_stream_closes_and_emits_done_only_on_completed(
    pipe_instance_fixture,
):
    pipe_instance, _ = pipe_instance_fixture
    stream = _FakeInteractionStream([_completed_event()])
    boundary = MagicMock()
    emitter = MagicMock(spec=EventEmitter)
    emitter.start_time = 0.0
    app = FastAPI()

    chunks = [
        chunk
        async for chunk in pipe_instance._present_interaction_stream(
            stream=cast(AsyncInteractionStream, stream),
            interactions=cast(AsyncInteractionsBoundary, boundary),
            app=app,
            event_emitter=emitter,
            metadata=cast("Metadata", {}),
            selected_service=_selected_service(),
        )
    ]

    assert stream.closed is True
    assert chunks[-1] == "data: [DONE]"
    assert chunks[0] == {"choices": [{"delta": {"content": "done"}}]}


@pytest.mark.asyncio
async def test_interaction_stream_resumes_same_id_and_event_cursor(pipe_instance_fixture):
    pipe_instance, _ = pipe_instance_fixture
    cursor_canary = "SIGNED_EVENT_CURSOR_CANARY"
    created = interaction_types.InteractionCreatedEvent(
        event_id=cursor_canary,
        interaction=interaction_types.InteractionSseEventInteraction(
            id="interaction-stream", status="in_progress"
        ),
    )
    first = _FakeInteractionStream([created], error=ConnectionError("dropped"))
    resumed = _FakeInteractionStream([_completed_event("cursor-2")])
    boundary = MagicMock()
    boundary.get = AsyncMock(return_value=resumed)
    emitter = MagicMock(spec=EventEmitter)
    emitter.start_time = 0.0
    app = FastAPI()

    captured: list[str] = []
    handler_id = gemini_manifold_module.log.add(
        lambda message: captured.append(str(message)), format="{message}"
    )
    try:
        chunks = [
            chunk
            async for chunk in pipe_instance._present_interaction_stream(
                stream=cast(AsyncInteractionStream, first),
                interactions=cast(AsyncInteractionsBoundary, boundary),
                app=app,
                event_emitter=emitter,
                metadata=cast("Metadata", {}),
                selected_service=_selected_service(),
            )
        ]
    finally:
        gemini_manifold_module.log.remove(handler_id)

    boundary.get.assert_awaited_once_with(
        "interaction-stream", stream=True, last_event_id=cursor_canary
    )
    assert cursor_canary not in "\n".join(captured)
    assert first.closed and resumed.closed
    assert chunks[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_interaction_stream_empty_error_and_cancel_never_emit_done(
    pipe_instance_fixture,
):
    pipe_instance, _ = pipe_instance_fixture
    emitter = MagicMock(spec=EventEmitter)
    emitter.start_time = 0.0
    app = FastAPI()
    boundary = MagicMock()
    boundary.cancel = AsyncMock()

    empty = _FakeInteractionStream([])
    with pytest.raises(InteractionExecutionError, match="without a terminal status"):
        _ = [
            chunk
            async for chunk in pipe_instance._present_interaction_stream(
                stream=cast(AsyncInteractionStream, empty),
                interactions=cast(AsyncInteractionsBoundary, boundary),
                app=app,
                event_emitter=emitter,
                metadata=cast("Metadata", {}),
                selected_service=_selected_service(),
            )
        ]
    assert empty.closed

    stream_canary = "STREAM_PROVIDER_ERROR_CANARY"
    failed = _FakeInteractionStream(
        [
            interaction_types.ErrorEvent(
                error=interaction_types.Error(code="500", message=stream_canary)
            )
        ]
    )
    with pytest.raises(
        InteractionExecutionError, match="Interaction error event"
    ) as stream_error_info:
        _ = [
            chunk
            async for chunk in pipe_instance._present_interaction_stream(
                stream=cast(AsyncInteractionStream, failed),
                interactions=cast(AsyncInteractionsBoundary, boundary),
                app=app,
                event_emitter=emitter,
                metadata=cast("Metadata", {}),
                selected_service=_selected_service(),
            )
        ]
    assert stream_canary not in str(stream_error_info.value)
    assert failed.closed

    cancelled = _FakeInteractionStream([], error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        _ = [
            chunk
            async for chunk in pipe_instance._present_interaction_stream(
                stream=cast(AsyncInteractionStream, cancelled),
                interactions=cast(AsyncInteractionsBoundary, boundary),
                app=app,
                event_emitter=emitter,
                metadata=cast("Metadata", {}),
                selected_service=_selected_service(),
            )
        ]
    assert cancelled.closed
    boundary.cancel.assert_not_awaited()


def _authorized_tool(
    function: AsyncMock,
    *,
    direct: bool = False,
    parameters: dict | None = None,
) -> dict[str, object]:
    return {
        "callable": function,
        "direct": direct,
        "spec": {
            "name": "backend_name",
            "description": "Look up a value",
            "parameters": parameters
            or {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


FORBIDDEN_AUTOMATIC_TOOL_REQUEST_FIELDS = {
    "automatic_function_calling",
    "automatic_function_calling_config",
    "cached_content",
    "batch",
    "batch_api",
    "batch_config",
    "video_metadata",
}


def _assert_explicit_function_requests(raw_requests: list[dict[str, object]]) -> None:
    def contains_callable(value: object) -> bool:
        if isinstance(value, dict):
            return any(contains_callable(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_callable(item) for item in value)
        return callable(value)

    assert raw_requests
    for request in raw_requests:
        assert FORBIDDEN_AUTOMATIC_TOOL_REQUEST_FIELDS.isdisjoint(request)
        tools = cast(list[dict[str, object]], request["tools"])
        assert tools
        assert all(tool.get("type") == "function" for tool in tools)
        assert not contains_callable(tools)


def _function_policy(enabled: bool = True) -> CatalogInteractions:
    return _selected_service(custom_function_calling=enabled).policy.interactions


def test_open_webui_tool_registry_uses_authorized_mapping_name():
    function = AsyncMock(return_value={"ok": True})
    registry = Pipe._resolve_open_webui_tools(
        {"catalog_lookup": _authorized_tool(function)},
        model_id="gemini-3.5-flash",
        interactions_policy=_function_policy(),
    )

    assert list(registry) == ["catalog_lookup"]
    assert registry["catalog_lookup"].declaration.name == "catalog_lookup"
    assert registry["catalog_lookup"].declaration.parameters == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }


def test_installed_catalog_advertises_only_implemented_tools_and_functions() -> None:
    catalog = Pipe._validate_app_state_catalog(_installed_model_catalog()).payload.runtime_models()
    implemented_builtins = {
        "google_search",
        "code_execution",
        "url_context",
        "google_maps",
    }
    custom_function_models: set[str] = set()

    for model_id, model in catalog.items():
        developer = model.services.developer
        assert isinstance(developer, CatalogSupportedService), model_id
        assert isinstance(model.services.enterprise, CatalogUnavailableService), model_id
        policy = developer.interactions
        if policy.custom_function_calling:
            custom_function_models.add(model_id)
        advertised = {
            tool_name
            for tool_name in (
                "google_search",
                "code_execution",
                "url_context",
                "google_maps",
            )
            if getattr(policy.tools, tool_name)
        }
        assert advertised <= implemented_builtins, model_id
        assert policy.tools.file_search is False, model_id

    assert custom_function_models == {"gemini-3.5-flash"}


@pytest.mark.parametrize(
    ("tools", "policy", "message"),
    [
        ({"lookup": _authorized_tool(AsyncMock())}, _function_policy(False), "not approved"),
        ({"bad-name": _authorized_tool(AsyncMock())}, _function_policy(), "Invalid"),
        ({"lookup": _authorized_tool(AsyncMock(), direct=True)}, _function_policy(), "direct"),
        ({"lookup": {"spec": {}}}, _function_policy(), "no authorized callable"),
        (
            {
                "lookup": _authorized_tool(
                    AsyncMock(),
                    parameters={
                        "type": "object",
                        "properties": {"__user__": {"type": "string"}},
                    },
                )
            },
            _function_policy(),
            "reserved",
        ),
    ],
)
def test_open_webui_tool_registry_fails_closed(tools, policy, message):
    with pytest.raises(ValueError, match=message):
        Pipe._resolve_open_webui_tools(
            tools,
            model_id="gemini-3.5-flash",
            interactions_policy=policy,
        )


@pytest.mark.asyncio
async def test_interaction_options_add_custom_function_and_allowed_tools(pipe_instance_fixture):
    pipe, _ = pipe_instance_fixture
    function = interaction_types.Function(
        name="lookup", parameters={"type": "object", "properties": {}}
    )
    options = await pipe._build_interaction_request_options(
        {},
        cast(Metadata, {"canonical_model_id": "gemini-test"}),
        pipe.valves,
        _interaction_policy(),
        [function],
    )

    assert function in options.tools
    assert options.generation_config.tool_choice is not None
    assert options.generation_config.tool_choice.allowed_tools is not None
    assert options.generation_config.tool_choice.allowed_tools.mode == "auto"
    assert options.generation_config.tool_choice.allowed_tools.tools == ["lookup"]


@pytest.mark.asyncio
async def test_selected_service_output_limit_is_enforced(pipe_instance_fixture):
    pipe, _ = pipe_instance_fixture
    policy = _selected_service(output_tokens=64)

    with pytest.raises(ValueError, match="selected service limit of 64"):
        await pipe._build_interaction_request_options(
            {"max_tokens": 65},
            cast(Metadata, {"canonical_model_id": "gemini-test"}),
            pipe.valves,
            policy,
        )


def test_app_state_catalog_boundary_narrows_supported_and_unavailable_services() -> None:
    envelope = Pipe._validate_app_state_catalog(_installed_model_catalog())
    catalog = envelope.payload.runtime_models()
    developer = catalog["gemini-2.5-flash"].services.developer
    enterprise = catalog["gemini-2.5-flash"].services.enterprise

    assert isinstance(developer, CatalogSupportedService)
    assert developer.limits.output_tokens == 65_536
    assert isinstance(enterprise, CatalogUnavailableService)
    assert enterprise.availability == "unverified"


@pytest.mark.parametrize("mutation", ["missing", "extra", "digest"])
def test_app_state_catalog_boundary_rejects_malformed_nested_policy(mutation: str) -> None:
    envelope = _installed_model_catalog()
    interactions = envelope["payload"]["product_authorizations"]["gemini-2.5-flash"]["interactions"]
    if mutation == "missing":
        interactions.pop("tools")
    elif mutation == "extra":
        interactions["unexpected"] = True
    else:
        envelope["canonical_digest"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match="catalog protocol 3"):
        Pipe._validate_app_state_catalog(envelope)


def _function_call(call_id: str, arguments: dict | None = None, name: str = "lookup"):
    return interaction_types.FunctionCallStep(
        id=call_id, name=name, arguments=arguments or {"query": "value"}
    )


def _tool_interaction(interaction_id: str, *calls):
    return interaction_types.Interaction(
        id=interaction_id, status="requires_action", steps=list(calls)
    )


def _final_interaction(text: str = "finished"):
    return interaction_types.Interaction(
        id="final",
        status="completed",
        steps=[
            interaction_types.ModelOutputStep(content=[interaction_types.TextContent(text=text)])
        ],
    )


def _tool_stream(
    interaction_id: str,
    *calls,
    argument_fragments: list[str] | None = None,
):
    events: list[interaction_types.InteractionSSEEvent] = [
        interaction_types.InteractionCreatedEvent(
            event_id=f"{interaction_id}-created",
            interaction=interaction_types.InteractionSseEventInteraction(
                id=interaction_id, status="in_progress"
            ),
        )
    ]
    for index, call_step in enumerate(calls):
        events.append(
            interaction_types.StepStart(
                event_id=f"{interaction_id}-start-{index}",
                index=index,
                step=interaction_types.FunctionCallStep(
                    id=call_step.id,
                    name=call_step.name,
                    arguments={},
                ),
            )
        )
        if argument_fragments is not None:
            for fragment_index, fragment in enumerate(argument_fragments):
                events.append(
                    interaction_types.StepDelta(
                        event_id=f"{interaction_id}-delta-{index}-{fragment_index}",
                        index=index,
                        delta=interaction_types.ArgumentsDelta(arguments=fragment),
                    )
                )
        events.append(
            interaction_types.StepStop(event_id=f"{interaction_id}-stop-{index}", index=index)
        )
    events.append(
        interaction_types.InteractionCompletedEvent(
            event_id=f"{interaction_id}-completed",
            interaction=interaction_types.InteractionSseEventInteraction(
                id=interaction_id,
                status="requires_action",
                steps=list(calls),
            ),
        )
    )
    return FakeInteractionStream(events)


def _final_stream(text: str = "finished"):
    return FakeInteractionStream(
        [
            interaction_types.InteractionCompletedEvent(
                event_id="final-completed",
                interaction=interaction_types.InteractionSseEventInteraction(
                    id="final",
                    status="completed",
                    steps=[
                        interaction_types.ModelOutputStep(
                            content=[interaction_types.TextContent(text=text)]
                        )
                    ],
                ),
            )
        ]
    )


def _tool_registry(function: AsyncMock):
    return Pipe._resolve_open_webui_tools(
        {"lookup": _authorized_tool(function)},
        model_id="gemini-3.5-flash",
        interactions_policy=_function_policy(),
    )


@pytest.mark.asyncio
async def test_custom_function_loop_executes_parallel_calls_and_preserves_result_order(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    started: list[str] = []
    both_started = asyncio.Event()

    async def lookup(query: object):
        started.append(cast(str, query))
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        return {"query": query}

    function = AsyncMock(side_effect=lookup)
    boundary = MagicMock()
    boundary.create = AsyncMock(
        side_effect=[
            _tool_interaction(
                "round-1",
                _function_call("call-1", {"query": "first"}),
                _function_call("call-2", {"query": "second"}),
            ),
            _final_interaction(),
        ]
    )

    emissions, reduction = await pipe._run_custom_function_loop(
        interactions=cast(AsyncInteractionsBoundary, boundary),
        common_request={"model": "gemini-3.5-flash", "input": "go", "store": True},
        registry=_tool_registry(function),
    )

    assert started == ["first", "second"]
    assert reduction.status == "completed"
    assert reduction.original_content == "finished"
    assert all(emission.kind != "tool" for emission in emissions)
    continuation = boundary.create.await_args_list[1].kwargs
    assert continuation["previous_interaction_id"] == "round-1"
    assert [item["call_id"] for item in continuation["input"]] == ["call-1", "call-2"]
    assert all(item["type"] == "function_result" for item in continuation["input"])


@pytest.mark.asyncio
async def test_custom_function_loop_streams_partial_arguments_and_closes_each_round(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    function = AsyncMock(return_value={"ok": True})
    first = _tool_stream(
        "round-1",
        _function_call("call-1", {"query": "streamed"}),
        argument_fragments=['{"query":"stream', 'ed"}'],
    )
    final = _final_stream()
    boundary = FakeInteractions([first, final])

    emissions, reduction = await pipe._run_custom_function_loop(
        interactions=cast(AsyncInteractionsBoundary, boundary),
        common_request={"model": "gemini-3.5-flash", "input": "go", "store": True},
        registry=_tool_registry(function),
        stream=True,
    )

    function.assert_awaited_once_with(query="streamed")
    assert reduction.status == "completed"
    assert reduction.original_content == "finished"
    assert all(emission.kind != "tool" for emission in emissions)
    assert [request.stream for request in boundary.requests] == [True, True]
    assert boundary.requests[1].previous_interaction_id == "round-1"
    assert first.close_count == 1
    assert final.close_count == 1
    boundary.assert_exhausted()


@pytest.mark.asyncio
async def test_custom_function_stream_rejects_malformed_arguments_without_execution(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    function = AsyncMock(return_value="must not run")
    boundary = FakeInteractions(
        [
            _tool_stream(
                "round-1",
                _function_call("call-1", {}),
                argument_fragments=['{"query":'],
            )
        ]
    )

    with pytest.raises(InteractionExecutionError, match="malformed streamed arguments"):
        await pipe._run_custom_function_loop(
            interactions=cast(AsyncInteractionsBoundary, boundary),
            common_request={"model": "gemini-3.5-flash", "input": "go", "store": True},
            registry=_tool_registry(function),
            stream=True,
        )

    function.assert_not_awaited()


@pytest.mark.asyncio
async def test_custom_function_parallel_cancellation_propagates(pipe_instance_fixture):
    pipe, _ = pipe_instance_fixture
    started = 0
    both_started = asyncio.Event()
    cleaned_up = 0

    async def blocking(**_kwargs):
        nonlocal started, cleaned_up
        started += 1
        if started == 2:
            both_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up += 1

    boundary = MagicMock()
    boundary.create = AsyncMock(
        return_value=_tool_interaction(
            "round-1",
            _function_call("call-1", {"query": "first"}),
            _function_call("call-2", {"query": "second"}),
        )
    )
    loop_task = asyncio.create_task(
        pipe._run_custom_function_loop(
            interactions=cast(AsyncInteractionsBoundary, boundary),
            common_request={"model": "gemini-3.5-flash", "input": "go", "store": True},
            registry=_tool_registry(AsyncMock(side_effect=blocking)),
        )
    )
    await asyncio.wait_for(both_started.wait(), timeout=1)
    loop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loop_task
    assert cleaned_up == 2
    assert boundary.create.await_count == 1


@pytest.mark.asyncio
async def test_custom_function_duplicate_ids_fail_before_execution(pipe_instance_fixture):
    pipe, _ = pipe_instance_fixture
    function = AsyncMock(return_value="must not run")
    boundary = MagicMock()
    boundary.create = AsyncMock(
        return_value=_tool_interaction(
            "round-1",
            _function_call("duplicate", {"query": "first"}),
            _function_call("duplicate", {"query": "second"}),
        )
    )

    with pytest.raises(InteractionExecutionError, match="duplicate function call IDs"):
        await pipe._run_custom_function_loop(
            interactions=cast(AsyncInteractionsBoundary, boundary),
            common_request={"model": "gemini-3.5-flash", "input": "go", "store": True},
            registry=_tool_registry(function),
        )
    function.assert_not_awaited()


@pytest.mark.asyncio
async def test_custom_function_errors_are_sanitized_and_strict_args_never_invoke(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    function = AsyncMock(side_effect=RuntimeError("secret backend detail"))
    registry = _tool_registry(function)
    records = {}

    invalid = await pipe._execute_custom_function_call(
        _function_call("invalid", {"query": "ok", "extra": "denied"}), registry, records
    )
    failed = await pipe._execute_custom_function_call(_function_call("failed"), registry, records)
    unauthorized = await pipe._execute_custom_function_call(
        _function_call("unknown", name="not_declared"), registry, records
    )

    assert invalid.is_error is True and "Unexpected arguments" in cast(str, invalid.result)
    assert function.await_count == 1
    assert failed.is_error is True
    assert "RuntimeError" in cast(str, failed.result)
    assert "secret backend detail" not in cast(str, failed.result)
    assert unauthorized.is_error is True


@pytest.mark.asyncio
async def test_custom_function_result_serialization_timeout_and_size_bounds(
    pipe_instance_fixture, monkeypatch
):
    pipe, _ = pipe_instance_fixture
    serial = AsyncMock(return_value={"b": 2, "a": 1})
    result = await pipe._execute_custom_function_call(
        _function_call("json"), _tool_registry(serial), {}
    )
    assert result.result == '{"a":1,"b":2}'

    monkeypatch.setattr(gemini_manifold_module, "GEMINI_TOOL_MAX_RESULT_BYTES", 3)
    oversized = await pipe._execute_custom_function_call(
        _function_call("large"), _tool_registry(AsyncMock(return_value="large")), {}
    )
    assert oversized.is_error is True and "exceeded" in cast(str, oversized.result)

    async def slow(**_kwargs):
        await asyncio.sleep(1)

    monkeypatch.setattr(gemini_manifold_module, "GEMINI_TOOL_TIMEOUT_SECONDS", 0.001)
    timed_out = await pipe._execute_custom_function_call(
        _function_call("slow"), _tool_registry(AsyncMock(side_effect=slow)), {}
    )
    assert timed_out.is_error is True and timed_out.result == "Tool execution timed out."


@pytest.mark.asyncio
async def test_custom_function_duplicate_call_is_idempotent_and_conflict_fails(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    function = AsyncMock(return_value="ok")
    registry = _tool_registry(function)
    records = {}
    first = await pipe._execute_custom_function_call(_function_call("same"), registry, records)
    duplicate = await pipe._execute_custom_function_call(_function_call("same"), registry, records)

    assert duplicate is first
    function.assert_awaited_once()
    with pytest.raises(InteractionExecutionError, match="reused with different"):
        await pipe._execute_custom_function_call(
            _function_call("same", {"query": "changed"}), registry, records
        )


@pytest.mark.asyncio
async def test_custom_function_loop_stateless_replays_full_signed_ledger_and_config(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    signature = base64.b64encode(b"opaque-tool-thought").decode("ascii")
    first = interaction_types.Interaction(
        id=None,
        status="requires_action",
        steps=[
            interaction_types.ThoughtStep(signature=signature),
            _function_call("call-1"),
        ],
    )
    boundary = MagicMock()
    boundary.create = AsyncMock(side_effect=[first, _final_interaction()])
    root = cast(
        list[interaction_types.StepParam],
        [{"type": "user_input", "content": [{"type": "text", "text": "go"}]}],
    )
    common_request = {
        "model": "gemini-3.5-flash",
        "input": root,
        "store": False,
        "previous_interaction_id": "must-not-leak",
        "system_instruction": "remain exact",
        "generation_config": {"temperature": 0.25},
        "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
        "response_format": {"type": "text", "mime_type": "text/plain"},
    }

    _emissions, reduction = await pipe._run_custom_function_loop(
        interactions=cast(AsyncInteractionsBoundary, boundary),
        common_request=common_request,
        registry=_tool_registry(AsyncMock(return_value={"ok": True})),
        root_replay_input=root,
    )

    assert reduction.status == "completed"
    assert boundary.create.await_count == 2
    first_request, second_request = [call.kwargs for call in boundary.create.await_args_list]
    for request in (first_request, second_request):
        assert request["store"] is False
        assert "previous_interaction_id" not in request
        assert request["system_instruction"] == "remain exact"
        assert request["generation_config"] == {"temperature": 0.25}
        assert request["tools"] == common_request["tools"]
        assert request["response_format"] == common_request["response_format"]
    assert [step["type"] for step in second_request["input"]] == [
        "user_input",
        "thought",
        "function_call",
        "function_result",
    ]
    assert second_request["input"][1]["signature"] == signature
    assert second_request["input"][2]["id"] == "call-1"
    assert second_request["input"][3]["call_id"] == "call-1"


@pytest.mark.asyncio
async def test_custom_function_loop_stateless_sse_replay_is_cumulative_across_rounds(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    streams = [
        _tool_stream("round-1", _function_call("call-1")),
        _tool_stream("round-2", _function_call("call-2")),
        _final_stream(),
    ]
    boundary = FakeInteractions(streams)
    root = cast(
        list[interaction_types.StepParam],
        [{"type": "user_input", "content": [{"type": "text", "text": "go"}]}],
    )

    _emissions, reduction = await pipe._run_custom_function_loop(
        interactions=cast(AsyncInteractionsBoundary, boundary),
        common_request={"model": "gemini-3.5-flash", "input": root, "store": False},
        registry=_tool_registry(AsyncMock(return_value="ok")),
        root_replay_input=root,
        stream=True,
    )

    assert reduction.status == "completed"
    assert [len(cast(list, request.input)) for request in boundary.requests] == [1, 3, 5]
    assert all(request.previous_interaction_id is None for request in boundary.requests)
    assert all(request.store is False and request.stream is True for request in boundary.requests)
    assert all(stream.close_count == 1 for stream in streams)
    boundary.assert_exhausted()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("root", "message"),
    [
        (
            [{"type": "function_call", "id": "orphan", "name": "lookup", "arguments": {}}],
            "end with",
        ),
        (
            [
                {"type": "function_call", "id": "orphan", "name": "lookup", "arguments": {}},
                {"type": "user_input", "content": [{"type": "text", "text": "go"}]},
            ],
            "incomplete",
        ),
        (
            [
                {
                    "type": "function_result",
                    "call_id": "missing-call",
                    "name": "lookup",
                    "result": "no call",
                },
                {"type": "user_input", "content": [{"type": "text", "text": "go"}]},
            ],
            "orphan",
        ),
        (
            [
                {"type": "function_call", "id": "same", "name": "lookup", "arguments": {}},
                {
                    "type": "function_result",
                    "call_id": "same",
                    "name": "lookup",
                    "result": "done",
                },
                {"type": "function_call", "id": "same", "name": "lookup", "arguments": {}},
                {
                    "type": "function_result",
                    "call_id": "same",
                    "name": "lookup",
                    "result": "again",
                },
                {"type": "user_input", "content": [{"type": "text", "text": "go"}]},
            ],
            "duplicate",
        ),
        ([{"type": "new_future_step"}], "unknown"),
    ],
)
async def test_custom_function_loop_rejects_invalid_stateless_root_before_provider_or_tool(
    pipe_instance_fixture, root, message
):
    pipe, _ = pipe_instance_fixture
    boundary = MagicMock()
    boundary.create = AsyncMock()
    function = AsyncMock(return_value="must not run")
    with pytest.raises(ValueError, match=message):
        await pipe._run_custom_function_loop(
            interactions=cast(AsyncInteractionsBoundary, boundary),
            common_request={"model": "gemini-3.5-flash", "input": root, "store": False},
            registry=_tool_registry(function),
            root_replay_input=cast(list[interaction_types.StepParam], root),
        )
    boundary.create.assert_not_awaited()
    function.assert_not_awaited()


@pytest.mark.asyncio
async def test_custom_function_loop_enforces_round_bounds(pipe_instance_fixture, monkeypatch):
    pipe, _ = pipe_instance_fixture
    function = AsyncMock(return_value="again")
    registry = _tool_registry(function)
    boundary = MagicMock()
    boundary.create = AsyncMock(
        side_effect=[
            _tool_interaction("one", _function_call("one")),
            _tool_interaction("two", _function_call("two")),
        ]
    )
    monkeypatch.setattr(gemini_manifold_module, "GEMINI_TOOL_MAX_ROUNDS", 2)
    with pytest.raises(InteractionExecutionError, match="2-round"):
        await pipe._run_custom_function_loop(
            interactions=cast(AsyncInteractionsBoundary, boundary),
            common_request={"model": "gemini-3.5-flash", "input": "go", "store": True},
            registry=registry,
        )


@pytest.mark.asyncio
async def test_custom_function_loop_enforces_call_bounds_and_nonempty_identity(
    pipe_instance_fixture, monkeypatch
):
    pipe, _ = pipe_instance_fixture
    registry = _tool_registry(AsyncMock(return_value="ok"))
    boundary = MagicMock()
    boundary.create = AsyncMock(
        return_value=_tool_interaction(
            "too-many",
            _function_call("one"),
            _function_call("two"),
        )
    )
    monkeypatch.setattr(gemini_manifold_module, "GEMINI_TOOL_MAX_CALLS_PER_ROUND", 1)
    with pytest.raises(InteractionExecutionError, match="per-round"):
        await pipe._run_custom_function_loop(
            interactions=cast(AsyncInteractionsBoundary, boundary),
            common_request={"model": "gemini-3.5-flash", "input": "go", "store": True},
            registry=registry,
        )

    monkeypatch.setattr(gemini_manifold_module, "GEMINI_TOOL_MAX_CALLS_PER_ROUND", 16)
    monkeypatch.setattr(gemini_manifold_module, "GEMINI_TOOL_MAX_TOTAL_CALLS", 1)
    with pytest.raises(InteractionExecutionError, match="total function"):
        await pipe._run_custom_function_loop(
            interactions=cast(AsyncInteractionsBoundary, boundary),
            common_request={"model": "gemini-3.5-flash", "input": "go", "store": True},
            registry=registry,
        )

    boundary.create = AsyncMock(return_value=_tool_interaction("bad-id", _function_call("")))
    with pytest.raises(InteractionExecutionError, match="non-empty"):
        await pipe._run_custom_function_loop(
            interactions=cast(AsyncInteractionsBoundary, boundary),
            common_request={"model": "gemini-3.5-flash", "input": "go", "store": True},
            registry=registry,
        )


def _interaction_envelope(
    *,
    interaction_id: str | None = "parent-interaction",
    endpoint_scope: str = "scope",
    model_id: str = "gemini-3.5-flash",
    store: bool = True,
    status: Literal[
        "in_progress",
        "requires_action",
        "completed",
        "failed",
        "cancelled",
        "incomplete",
        "budget_exceeded",
    ] = "completed",
    visible_content: str = "parent answer",
    steps: list[dict] | None = None,
) -> dict[str, object]:
    return InteractionEnvelopeV1(
        interaction_id=interaction_id,
        endpoint_scope=endpoint_scope,
        model_id=model_id,
        store=store,
        status=status,
        visible_content=visible_content,
        steps=steps
        or [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": visible_content}],
            }
        ],
    ).model_dump(mode="json", exclude_none=False)


def _continuation_builder(pipe, *, parent_content: str = "parent answer") -> GeminiContentBuilder:
    files_manager = MagicMock()
    files_manager.endpoint_identity = _developer_identity().for_model("gemini-3.5-flash")
    builder = GeminiContentBuilder(
        messages_body=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": parent_content},
            {"role": "user", "content": "next"},
        ],
        metadata_body={"chat_id": "chat"},  # type: ignore
        user_data={"id": "user", "email": "user@example.com"},  # type: ignore
        event_emitter=MagicMock(spec=EventEmitter),
        valves=pipe.valves,
        files_api_manager=files_manager,
        pdf_mitigation_manager=pipe.pdf_mitigation_manager,
        service_policy=_builder_service_policy(),
    )
    builder.messages_body = cast(
        list,
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": parent_content},
            {"role": "user", "content": "next"},
        ],
    )
    builder.messages_db = cast(
        list,
        [
            {"id": "u1", "role": "user", "content": "first"},
            {
                "id": "a1",
                "role": "assistant",
                "content": "parent answer",
                "gemini_interaction": _interaction_envelope(),
            },
            {"id": "u2", "role": "user", "content": "next"},
        ],
    )
    return builder


def test_interaction_envelope_is_strict_and_versioned(pipe_instance_fixture):
    pipe, _ = pipe_instance_fixture
    builder = _continuation_builder(pipe)
    assert builder.messages_db is not None
    valid = builder._parse_interaction_envelope(builder.messages_db[1])
    assert valid is not None and valid.version == 1

    malformed = dict(builder.messages_db[1])
    malformed["gemini_interaction"] = {**_interaction_envelope(), "legacy": True}
    with pytest.raises(ContentBuildError, match="malformed or uses an unsupported version"):
        builder._parse_interaction_envelope(malformed)  # type: ignore
    future = dict(builder.messages_db[1])
    future["gemini_interaction"] = {**_interaction_envelope(), "version": 2}
    with pytest.raises(ContentBuildError, match="unsupported version"):
        builder._parse_interaction_envelope(future)  # type: ignore


def test_continuation_uses_only_current_input_for_same_scope_completed_parent(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    builder = _continuation_builder(pipe)
    full_input = cast(
        list[interaction_types.StepParam],
        [
            {"type": "user_input", "content": [{"type": "text", "text": "first"}]},
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "parent answer"}],
            },
            {"type": "user_input", "content": [{"type": "text", "text": "next"}]},
        ],
    )
    decision = builder.select_continuation(
        full_input, store=True, endpoint_scope="scope", model_id="gemini-3.5-flash"
    )

    assert decision.used_server_state is True
    assert decision.previous_interaction_id == "parent-interaction"
    assert decision.input == [full_input[-1]]


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("endpoint_scope", "foreign"),
        ("model_id", "other-model"),
        ("status", "failed"),
        ("store", False),
        ("interaction_id", None),
    ],
)
def test_continuation_replays_full_ledger_for_ineligible_parent(
    pipe_instance_fixture, change, value
):
    pipe, _ = pipe_instance_fixture
    builder = _continuation_builder(pipe)
    assert builder.messages_db is not None
    builder.messages_db[1]["gemini_interaction"] = _interaction_envelope(**{change: value})
    full_input = cast(
        list[interaction_types.StepParam],
        [
            {"type": "user_input", "content": [{"type": "text", "text": "first"}]},
            {"type": "user_input", "content": [{"type": "text", "text": "next"}]},
        ],
    )

    decision = builder.select_continuation(
        full_input, store=True, endpoint_scope="scope", model_id="gemini-3.5-flash"
    )

    assert decision.used_server_state is False
    assert decision.previous_interaction_id is None
    assert decision.input == full_input


def test_continuation_edit_privacy_and_non_gemini_parent_force_replay(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    full_input = cast(
        list[interaction_types.StepParam],
        [
            {"type": "user_input", "content": [{"type": "text", "text": "first"}]},
            {"type": "user_input", "content": [{"type": "text", "text": "next"}]},
        ],
    )
    edited = _continuation_builder(pipe, parent_content="edited")
    assert edited.messages_db is not None
    assert not edited.select_continuation(
        full_input, store=True, endpoint_scope="scope", model_id="gemini-3.5-flash"
    ).used_server_state
    assert not edited.select_continuation(
        full_input, store=False, endpoint_scope="scope", model_id="gemini-3.5-flash"
    ).used_server_state
    edited.messages_db[1].pop("gemini_interaction")
    assert not edited.select_continuation(
        full_input, store=True, endpoint_scope="scope", model_id="gemini-3.5-flash"
    ).used_server_state


@pytest.mark.asyncio
async def test_edit_detection_never_logs_prompt_or_output_text(pipe_instance_fixture) -> None:
    pipe, _ = pipe_instance_fixture
    canary = "EDITED_CONVERSATION_CANARY"
    builder = _continuation_builder(pipe, parent_content=canary)
    assert builder.messages_db is not None
    message_db = builder.messages_db[1]
    event_emitter = cast(MagicMock, builder.event_emitter)
    builder._interaction_contents_from_text = AsyncMock(
        return_value=[interaction_types.TextContentParam(type="text", text=canary)]
    )
    captured: list[str] = []
    handler_id = gemini_manifold_module.log.add(
        lambda message: captured.append(str(message)), format="{message}"
    )
    try:
        await builder._process_assistant_message(
            1,
            cast("AssistantMessage", {"role": "assistant", "content": canary}),
            cast("ChatMessageTD", message_db),
            None,
            asyncio.Queue(),
        )
    finally:
        gemini_manifold_module.log.remove(handler_id)

    assert canary not in "\n".join(captured)
    assert canary not in str(event_emitter.emit_toast.call_args_list)


@pytest.mark.parametrize(
    ("admin", "user", "temp", "task", "expected"),
    [
        (True, None, False, False, True),
        (True, False, False, False, False),
        (False, True, False, False, False),
        (True, True, True, False, False),
        (True, True, False, True, False),
    ],
)
def test_interaction_storage_policy_is_explicit_and_privacy_monotonic(
    admin, user, temp, task, expected
):
    assert (
        Pipe._resolve_store_policy(
            admin_allows=admin,
            user_preference=user,
            is_temp=temp,
            is_task=task,
        )
        is expected
    )


def test_selected_service_must_explicitly_support_interaction_storage() -> None:
    assert _selected_service(store=True).policy.interactions.store is True
    assert _selected_service(store=False).policy.interactions.store is False


@pytest.mark.asyncio
async def test_completed_reduction_persists_one_envelope_idempotently(pipe_instance_fixture):
    pipe, _ = pipe_instance_fixture
    mock_chats_module.Chats.upsert_message_to_chat_by_id_and_message_id = AsyncMock()
    emitter = MagicMock(spec=EventEmitter)
    emitter.start_time = 0.0
    metadata = cast(
        Metadata,
        {
            "chat_id": "chat",
            "message_id": "message",
            "canonical_model_id": "gemini-3.5-flash",
            "gemini_endpoint_scope": "scope",
            "gemini_effective_store": True,
        },
    )
    reduction = gemini_manifold_module.InteractionReduction(
        interaction_id="stored-id",
        status="completed",
        terminal=True,
        steps=[
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "answer"}],
            }
        ],
        original_content="answer",
        last_event_id="cursor-final",
    )

    for _ in range(2):
        _ = [
            chunk
            async for chunk in pipe._finalize_reduction(
                reduction,
                emitter,
                metadata,
                _selected_service(model_id="gemini-3.5-flash"),
            )
        ]

    mock_chats_module.Chats.upsert_message_to_chat_by_id_and_message_id.assert_awaited_once()
    awaited = mock_chats_module.Chats.upsert_message_to_chat_by_id_and_message_id.await_args
    assert awaited is not None
    payload = awaited.kwargs["message"]
    assert set(payload) == {"gemini_interaction"}
    envelope = InteractionEnvelopeV1.model_validate(payload["gemini_interaction"])
    assert envelope.interaction_id == "stored-id"
    assert envelope.steps == reduction.steps
    assert envelope.visible_content == "answer"
    assert envelope.last_event_id == "cursor-final"


@pytest.mark.asyncio
async def test_stateless_envelope_drops_server_id_and_temp_chat_never_persists(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    mock_chats_module.Chats.upsert_message_to_chat_by_id_and_message_id = AsyncMock()
    emitter = MagicMock(spec=EventEmitter)
    emitter.start_time = 0.0
    reduction = gemini_manifold_module.InteractionReduction(
        interaction_id="ephemeral-server-id",
        status="completed",
        terminal=True,
        steps=[],
        original_content="answer",
    )
    stateless = cast(
        Metadata,
        {
            "chat_id": "persisted-chat",
            "message_id": "stateless-message",
            "canonical_model_id": "gemini-3.5-flash",
            "gemini_endpoint_scope": "scope",
            "gemini_effective_store": False,
        },
    )
    selected = _selected_service(model_id="gemini-3.5-flash")
    _ = [item async for item in pipe._finalize_reduction(reduction, emitter, stateless, selected)]
    awaited = mock_chats_module.Chats.upsert_message_to_chat_by_id_and_message_id.await_args
    assert awaited is not None
    envelope = InteractionEnvelopeV1.model_validate(awaited.kwargs["message"]["gemini_interaction"])
    assert envelope.store is False
    assert envelope.interaction_id is None

    mock_chats_module.Chats.upsert_message_to_chat_by_id_and_message_id.reset_mock()
    temporary = cast(
        Metadata,
        {
            "chat_id": "local",
            "message_id": "temporary-message",
            "canonical_model_id": "gemini-3.5-flash",
            "gemini_endpoint_scope": "scope",
            "gemini_effective_store": False,
        },
    )
    _ = [item async for item in pipe._finalize_reduction(reduction, emitter, temporary, selected)]
    mock_chats_module.Chats.upsert_message_to_chat_by_id_and_message_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_message_generation_guard_blocks_duplicate_create_until_stream_finishes(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    release = asyncio.Event()
    calls = 0

    async def stream():
        await release.wait()
        yield "data: [DONE]"

    async def execute():
        nonlocal calls
        calls += 1
        return stream()

    key = ("chat", "message")
    first = await pipe._execute_message_once(key, execute)
    with pytest.raises(RuntimeError, match="already in progress or completed"):
        await pipe._execute_message_once(key, execute)
    release.set()
    assert not isinstance(first, dict)
    assert [item async for item in first] == ["data: [DONE]"]
    assert calls == 1

    second = await pipe._execute_message_once(key, execute)
    assert not isinstance(second, dict)
    assert [item async for item in second] == ["data: [DONE]"]
    assert calls == 2


@pytest.mark.asyncio
async def test_message_generation_guard_does_not_share_missing_or_local_keys(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture

    async def execute():
        return {"ok": True}

    assert await pipe._execute_message_once(("", ""), execute) == {"ok": True}
    assert await pipe._execute_message_once(("local", "message"), execute) == {"ok": True}
    assert pipe._generation_inflight == set()


@pytest.mark.asyncio
async def test_message_gates_and_completion_memory_are_bounded_and_reclaimed(
    pipe_instance_fixture, monkeypatch
) -> None:
    pipe, _constructor = pipe_instance_fixture
    key = ("chat", "message")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def holder() -> None:
        async with pipe._message_guard(key):
            entered.set()
            await release.wait()

    async def waiter() -> None:
        await entered.wait()
        async with pipe._message_guard(key):
            return

    holder_task = asyncio.create_task(holder())
    await entered.wait()
    waiter_task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    assert pipe._message_locks[key].users == 2
    release.set()
    await asyncio.gather(holder_task, waiter_task)
    assert pipe._message_locks == {}

    monkeypatch.setattr(gemini_manifold_module, "MESSAGE_COMPLETION_MAX_ENTRIES", 2)
    pipe._remember_completed_message(("chat", "one"))
    pipe._remember_completed_message(("chat", "two"))
    pipe._remember_completed_message(("chat", "three"))
    assert list(pipe._persisted_messages) == [("chat", "two"), ("chat", "three")]

    calls = 0

    async def execute() -> dict[str, bool]:
        nonlocal calls
        calls += 1
        return {"ok": True}

    with pytest.raises(RuntimeError, match="already in progress or completed"):
        await pipe._execute_message_once(("chat", "three"), execute)
    assert calls == 0
    for completed_key in pipe._persisted_messages:
        pipe._persisted_messages[completed_key] = 0.0
    assert await pipe._execute_message_once(("chat", "three"), execute) == {"ok": True}
    assert calls == 1
    assert pipe._message_locks == {}


@pytest.mark.asyncio
async def test_pipe_shutdown_is_idempotent_closes_all_clients_and_clears_state(
    pipe_instance_fixture,
) -> None:
    pipe, _constructor = pipe_instance_fixture
    first = MagicMock()
    first.aio.aclose = AsyncMock()
    second = MagicMock()
    second.aio.aclose = AsyncMock()
    Pipe._cached_client_bindings = [
        GenAIClientBinding(cast(genai.Client, first), _developer_identity()),
        GenAIClientBinding(cast(genai.Client, second), _developer_identity()),
    ]
    pipe._remember_completed_message(("chat", "message"))
    await pipe.file_content_cache.set("file", object())

    await pipe.shutdown()
    await pipe.shutdown()

    first.aio.aclose.assert_awaited_once()
    second.aio.aclose.assert_awaited_once()
    assert Pipe._cached_client_bindings == []
    assert pipe._persisted_messages == {}
    assert pipe._message_locks == {}
    assert await pipe.file_content_cache.get("file") is None


@pytest.mark.asyncio
async def test_pipe_shutdown_attempts_all_clients_and_is_final_after_failure(
    pipe_instance_fixture,
) -> None:
    pipe, _constructor = pipe_instance_fixture
    failed = MagicMock()
    failed.aio.aclose = AsyncMock(side_effect=RuntimeError("close failed"))
    closed = MagicMock()
    closed.aio.aclose = AsyncMock()
    Pipe._cached_client_bindings = [
        GenAIClientBinding(cast(genai.Client, failed), _developer_identity()),
        GenAIClientBinding(cast(genai.Client, closed), _developer_identity()),
    ]

    with pytest.raises(ExceptionGroup, match="Failed to close"):
        await pipe.shutdown()
    await pipe.shutdown()

    failed.aio.aclose.assert_awaited_once()
    closed.aio.aclose.assert_awaited_once()
    assert pipe._shutdown_complete is True


@pytest.mark.asyncio
async def test_previous_interaction_404_retries_once_as_full_root(pipe_instance_fixture):
    pipe, _ = pipe_instance_fixture

    class NotFound(Exception):
        status_code = 404

    boundary = MagicMock()
    boundary.create = AsyncMock(side_effect=[NotFound(), _final_interaction("replayed")])
    full_input = cast(
        list[interaction_types.StepParam],
        [
            {"type": "user_input", "content": [{"type": "text", "text": "old"}]},
            {"type": "user_input", "content": [{"type": "text", "text": "new"}]},
        ],
    )

    _emissions, reduction = await pipe._run_custom_function_loop(
        interactions=cast(AsyncInteractionsBoundary, boundary),
        common_request={
            "model": "gemini-3.5-flash",
            "input": [full_input[-1]],
            "store": True,
            "previous_interaction_id": "expired",
        },
        registry=_tool_registry(AsyncMock(return_value="unused")),
        root_replay_input=full_input,
    )

    assert reduction.original_content == "replayed"
    replay_request = boundary.create.await_args_list[1].kwargs
    assert replay_request["input"] == full_input
    assert "previous_interaction_id" not in replay_request
    assert boundary.create.await_count == 2


# region Test USER_MUST_PROVIDE_AUTH_CONFIG=True scenarios
def test_user_must_auth_no_user_key_provided_errors(pipe_instance_fixture):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=True. User provides no keys. Expected: ValueError.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = None

    user_valves_instance = Pipe.UserValves(GEMINI_FREE_API_KEY=None, GEMINI_PAID_API_KEY=None)
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_UNPRIVILEGED
    )

    with pytest.raises(ValueError, match="Please set GEMINI_FREE_API_KEY or GEMINI_PAID_API_KEY"):
        pipe._get_user_client(merged_valves, USER_EMAIL_UNPRIVILEGED)

    MockedGenAIClientConstructor.assert_not_called()
    Pipe._get_or_create_genai_client.cache_clear()


def test_user_must_auth_user_provides_gemini_key_uses_user_creds(pipe_instance_fixture):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=True. User provides their own Gemini key.
    Expected: Gemini client with user's credentials, ignoring admin's.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = None
    pipe.valves.GEMINI_FREE_API_KEY = ADMIN_FREE_KEY
    pipe.valves.GEMINI_PAID_API_KEY = ADMIN_PAID_KEY

    user_valves_instance = Pipe.UserValves(
        GEMINI_FREE_API_KEY=USER_FREE_KEY,
        GEMINI_PAID_API_KEY=None,
        GEMINI_API_BASE_URL=USER_GEMINI_BASE_URL,
    )
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_UNPRIVILEGED
    )

    pipe._get_user_client(merged_valves, USER_EMAIL_UNPRIVILEGED)

    MockedGenAIClientConstructor.assert_called_once_with(
        api_key=USER_FREE_KEY,
        http_options=gemini_types.HttpOptions(api_version="v1", base_url=USER_GEMINI_BASE_URL),
    )
    Pipe._get_or_create_genai_client.cache_clear()


def test_user_must_auth_user_tries_enterprise_no_user_gemini_key_errors(
    pipe_instance_fixture,
):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=True. User tries Enterprise without providing a fallback Gemini key.
    Expected: ValueError because Enterprise usage is denied and no fallback is available.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = None

    user_valves_instance = Pipe.UserValves(
        USE_ENTERPRISE=True,
        ENTERPRISE_PROJECT="user_tries_this_project",
        GEMINI_FREE_API_KEY=None,
        GEMINI_PAID_API_KEY=None,
    )
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_UNPRIVILEGED
    )

    assert merged_valves.USE_ENTERPRISE is False
    assert merged_valves.GEMINI_FREE_API_KEY is None
    assert merged_valves.GEMINI_PAID_API_KEY is None

    with pytest.raises(
        ValueError, match="User must provide their own authentication configuration"
    ):
        pipe._get_user_client(merged_valves, USER_EMAIL_UNPRIVILEGED)

    MockedGenAIClientConstructor.assert_not_called()
    Pipe._get_or_create_genai_client.cache_clear()


def test_user_must_auth_user_tries_enterprise_with_user_gemini_key_falls_back(
    pipe_instance_fixture,
):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=True. User tries Enterprise but provides a Gemini key.
    Expected: Falls back to Gemini Developer API using the user's provided key.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = None

    user_valves_instance = Pipe.UserValves(
        USE_ENTERPRISE=True,
        GEMINI_FREE_API_KEY=USER_FREE_KEY,
        GEMINI_PAID_API_KEY=USER_PAID_KEY,
        GEMINI_API_BASE_URL=USER_GEMINI_BASE_URL,
    )
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_UNPRIVILEGED
    )

    assert merged_valves.USE_ENTERPRISE is False
    assert merged_valves.ENTERPRISE_PROJECT is None
    assert merged_valves.GEMINI_FREE_API_KEY == USER_FREE_KEY
    assert merged_valves.GEMINI_PAID_API_KEY == USER_PAID_KEY

    pipe._get_user_client(merged_valves, USER_EMAIL_UNPRIVILEGED)

    MockedGenAIClientConstructor.assert_called_once_with(
        api_key=USER_FREE_KEY,  # Should prefer user's free key for fallback
        http_options=gemini_types.HttpOptions(api_version="v1", base_url=USER_GEMINI_BASE_URL),
    )
    Pipe._get_or_create_genai_client.cache_clear()


# endregion Test USER_MUST_PROVIDE_AUTH_CONFIG=True scenarios


# region Test USER_MUST_PROVIDE_AUTH_CONFIG=True with whitelisted user
def test_whitelist_user_no_uservalves_uses_admin_gemini_config(pipe_instance_fixture):
    """
    Whitelisted user with no UserValves uses admin's Gemini config.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = USER_EMAIL_WHITELISTED
    pipe.valves.USE_ENTERPRISE = False

    user_valves_instance = Pipe.UserValves()
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_WHITELISTED
    )

    pipe._get_user_client(merged_valves, USER_EMAIL_WHITELISTED)

    MockedGenAIClientConstructor.assert_called_once_with(
        api_key=ADMIN_FREE_KEY,
        http_options=gemini_types.HttpOptions(api_version="v1", base_url=ADMIN_GEMINI_BASE_URL),
    )
    Pipe._get_or_create_genai_client.cache_clear()


def test_whitelist_user_no_uservalves_uses_admin_enterprise_config(pipe_instance_fixture):
    """
    Whitelisted user with no UserValves uses admin's Enterprise config.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = USER_EMAIL_WHITELISTED
    pipe.valves.USE_ENTERPRISE = True
    pipe.valves.ENTERPRISE_PROJECT = ADMIN_ENTERPRISE_PROJECT

    user_valves_instance = Pipe.UserValves()
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_WHITELISTED
    )

    pipe._get_user_client(merged_valves, USER_EMAIL_WHITELISTED)

    MockedGenAIClientConstructor.assert_called_once_with(
        enterprise=True,
        project=ADMIN_ENTERPRISE_PROJECT,
        location=pipe.valves.ENTERPRISE_LOCATION,
        http_options=gemini_types.HttpOptions(
            api_version="v1beta1", base_url=ADMIN_GEMINI_BASE_URL
        ),
    )
    Pipe._get_or_create_genai_client.cache_clear()


def test_whitelist_user_provides_own_gemini_key_overrides_admin(pipe_instance_fixture):
    """
    Whitelisted user provides their own Gemini key, which overrides admin's config.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = USER_EMAIL_WHITELISTED
    pipe.valves.USE_ENTERPRISE = False
    # Explicitly remove the admin's free key for this test case to ensure
    # we are testing the user's paid key override without ambiguity.
    pipe.valves.GEMINI_FREE_API_KEY = None
    pipe.valves.GEMINI_PAID_API_KEY = "admin_paid_key_to_be_overridden"

    user_valves_instance = Pipe.UserValves(GEMINI_PAID_API_KEY=USER_PAID_KEY)
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_WHITELISTED
    )

    pipe._get_user_client(merged_valves, USER_EMAIL_WHITELISTED)

    MockedGenAIClientConstructor.assert_called_once_with(
        api_key=USER_PAID_KEY,
        http_options=gemini_types.HttpOptions(api_version="v1", base_url=ADMIN_GEMINI_BASE_URL),
    )
    Pipe._get_or_create_genai_client.cache_clear()


# endregion Test USER_MUST_PROVIDE_AUTH_CONFIG=True with whitelisted user


# region Test user's ability to override admin's settings
def test_user_opts_out_of_admin_enterprise_to_user_gemini(pipe_instance_fixture):
    """
    Admin uses Enterprise. User opts-out to use their own Gemini key.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = False
    pipe.valves.USE_ENTERPRISE = True
    pipe.valves.ENTERPRISE_PROJECT = ADMIN_ENTERPRISE_PROJECT

    user_valves_instance = Pipe.UserValves(
        USE_ENTERPRISE=False,
        GEMINI_FREE_API_KEY=USER_FREE_KEY,
        GEMINI_API_BASE_URL=USER_GEMINI_BASE_URL,
    )
    merged_valves = pipe._get_merged_valves(pipe.valves, user_valves_instance, USER_EMAIL_REGULAR)

    pipe._get_user_client(merged_valves, USER_EMAIL_REGULAR)

    MockedGenAIClientConstructor.assert_called_once_with(
        api_key=USER_FREE_KEY,
        http_options=gemini_types.HttpOptions(api_version="v1", base_url=USER_GEMINI_BASE_URL),
    )
    Pipe._get_or_create_genai_client.cache_clear()


def test_user_opts_in_to_enterprise_from_admin_gemini(pipe_instance_fixture):
    """
    Admin uses Gemini. User opts-in to use their own Enterprise project.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = False
    pipe.valves.USE_ENTERPRISE = False

    user_valves_instance = Pipe.UserValves(
        USE_ENTERPRISE=True,
        ENTERPRISE_PROJECT=USER_ENTERPRISE_PROJECT,
        ENTERPRISE_LOCATION=USER_ENTERPRISE_LOCATION,
    )
    merged_valves = pipe._get_merged_valves(pipe.valves, user_valves_instance, USER_EMAIL_REGULAR)

    pipe._get_user_client(merged_valves, USER_EMAIL_REGULAR)

    MockedGenAIClientConstructor.assert_called_once_with(
        enterprise=True,
        project=USER_ENTERPRISE_PROJECT,
        location=USER_ENTERPRISE_LOCATION,
        http_options=gemini_types.HttpOptions(
            api_version="v1beta1", base_url=ADMIN_GEMINI_BASE_URL
        ),
    )
    Pipe._get_or_create_genai_client.cache_clear()


# endregion Test user's ability to override admin's settings


# region Test Toggleable Paid API Filter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "toggle_status, expected_api_key",
    [
        (
            (True, True),
            ADMIN_PAID_KEY,
        ),
        (
            (True, False),
            ADMIN_FREE_KEY,
        ),
        (
            (False, False),
            ADMIN_FREE_KEY,
        ),
    ],
)
async def test_paid_api_toggle_selects_correct_key(
    pipe_instance_fixture, toggle_status, expected_api_key
):
    """
    Tests that the 'gemini_paid_api' toggle correctly selects the paid or free key.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    # Provide both the model ID and the canonical ID required by the new pipe logic
    model_id = "gemini-2.5-flash"
    __metadata__ = {"model": {"id": model_id}, "canonical_model_id": model_id}

    # Ensure both keys are present in the initial valves
    pipe.valves.GEMINI_FREE_API_KEY = ADMIN_FREE_KEY
    pipe.valves.GEMINI_PAID_API_KEY = ADMIN_PAID_KEY

    # Mock the request app state which is now required early in pipe()
    mock_request = MagicMock()
    mock_request.app.state._state = {"gemini_model_catalog": _installed_model_catalog()}

    def mock_toggle_side_effect(filter_id, metadata):
        if filter_id == "gemini_paid_api":
            return toggle_status
        return (False, False)

    with (
        patch.object(
            pipe, "_get_toggleable_feature_status", side_effect=mock_toggle_side_effect
        ) as mock_toggle_status,
        patch.object(pipe, "_get_user_client", wraps=pipe._get_user_client),
        patch.object(pipe, "_check_companion_filter_version"),
        patch.object(pipe, "_get_merged_valves", return_value=pipe.valves),
    ):
        try:
            # We expect this to get quite far now with the fixed metadata
            await pipe.pipe(
                body={"model": model_id, "messages": []},
                __user__={"email": "test@test.com"},
                __request__=mock_request,
                __event_emitter__=None,
                __metadata__=__metadata__,
            )
        except Exception:
            # Still catching exceptions as we aren't mocking the full builder/network stack
            pass

        # Update: Pipe now adds 'merged_custom_params' to metadata before checking toggles.
        # We check that the toggle was called with the model ID present.
        paid_toggle_calls = [
            toggle_call
            for toggle_call in mock_toggle_status.call_args_list
            if toggle_call.args and toggle_call.args[0] == "gemini_paid_api"
        ]
        assert paid_toggle_calls
        args, _ = paid_toggle_calls[-1]
        assert args[0] == "gemini_paid_api"
        assert args[1]["model"]["id"] == model_id

        # The partial pipe harness intentionally stops before execution. Exercise
        # the selected endpoint configuration directly.
        if expected_api_key == ADMIN_PAID_KEY:
            pipe.valves.GEMINI_FREE_API_KEY = None
        else:
            pipe.valves.GEMINI_PAID_API_KEY = None
        pipe._get_user_client(pipe.valves, USER_EMAIL_REGULAR)

        # Assert the genai Client was initialized with the routed key.
        MockedGenAIClientConstructor.assert_called()
        # Find the call that matches our expected key
        api_keys_used = [
            call.kwargs.get("api_key") for call in MockedGenAIClientConstructor.call_args_list
        ]
        assert expected_api_key in api_keys_used

    Pipe._get_or_create_genai_client.cache_clear()


# endregion Test Toggleable Paid API Filter


# region Test _get_genai_models
async def _clear_model_discovery_cache(pipe: Pipe) -> None:
    cache = getattr(pipe._get_genai_models, "cache", None)
    if isinstance(cache, BaseCache):
        await cache.clear()


def _developer_discovery_binding() -> GenAIClientBinding:
    return GenAIClientBinding(
        client=MagicMock(),
        identity=_developer_identity(),
    )


@pytest.mark.asyncio
async def test_developer_discovery_uses_catalog_not_legacy_supported_actions(
    pipe_instance_fixture,
) -> None:
    pipe, _ = pipe_instance_fixture
    await _clear_model_discovery_cache(pipe)
    display_canary = "PROVIDER_DISPLAY_NAME_CANARY"
    description_canary = "PROVIDER_DESCRIPTION_CANARY"
    uncatalogued_canary = "PROVIDER_UNCATALOGUED_ID_CANARY"
    listed_models = [
        gemini_types.Model(
            name="models/gemini-2.5-flash",
            display_name=display_canary,
            description=description_canary,
            supported_actions=[],
        ),
        gemini_types.Model(
            name=f"models/{uncatalogued_canary}",
            display_name="Uncatalogued with legacy action",
            supported_actions=["generateContent"],
        ),
    ]
    fetch = AsyncMock(return_value=listed_models)
    captured: list[str] = []
    handler_id = gemini_manifold_module.log.add(
        lambda message: captured.append(str(message)), format="{message}"
    )

    try:
        with (
            patch.object(
                pipe,
                "_get_or_create_genai_client",
                return_value=_developer_discovery_binding(),
            ),
            patch.object(pipe, "_fetch_models_from_client_internal", new=fetch),
        ):
            models = await pipe._get_genai_models(free_api_key="developer-key")
    finally:
        gemini_manifold_module.log.remove(handler_id)

    assert models == [
        {
            "id": "gemini-2.5-flash",
            "name": "gemini-2.5-flash",
            "description": "Catalog-validated Gemini model.",
        }
    ]
    observable = repr(models) + "\n".join(captured)
    assert display_canary not in observable
    assert description_canary not in observable
    assert uncatalogued_canary not in observable
    fetch.assert_awaited_once()
    await _clear_model_discovery_cache(pipe)


@pytest.mark.asyncio
async def test_developer_discovery_surfaces_all_retained_and_no_excluded_targets(
    pipe_instance_fixture,
) -> None:
    pipe, _ = pipe_instance_fixture
    await _clear_model_discovery_cache(pipe)
    provenance_path = (
        Path(__file__).parents[1] / "docs" / "development" / "gemini-model-provenance-v1.yaml"
    )
    provenance = yaml.safe_load(provenance_path.read_text(encoding="utf-8"))
    provider_ids = provenance["interactions_supported_ids"]
    excluded = set(provenance["product_scope_exclusions"])
    removed = {
        model_id
        for model_id, decision in provenance["current_catalog_reconciliation"].items()
        if decision["disposition"] == "remove"
    }
    listed_ids = list(provider_ids["models"]) + list(provider_ids["agents"]) + sorted(removed)
    fetch = AsyncMock(
        return_value=[gemini_types.Model(name=f"models/{model_id}") for model_id in listed_ids]
    )

    with (
        patch.object(
            pipe,
            "_get_or_create_genai_client",
            return_value=_developer_discovery_binding(),
        ),
        patch.object(pipe, "_fetch_models_from_client_internal", new=fetch),
    ):
        models = await pipe._get_genai_models(free_api_key="developer-key-all-models")

    discovered = {model["id"] for model in models}
    assert discovered == CATALOG_MODEL_IDS
    assert discovered.isdisjoint(excluded | removed)
    fetch.assert_awaited_once()
    await _clear_model_discovery_cache(pipe)


@pytest.mark.asyncio
async def test_developer_discovery_applies_whitelist_and_blacklist(
    pipe_instance_fixture,
) -> None:
    pipe, _ = pipe_instance_fixture
    await _clear_model_discovery_cache(pipe)
    listed_models = [
        gemini_types.Model(name="models/gemini-2.5-flash"),
        gemini_types.Model(name="models/gemini-2.5-pro"),
        gemini_types.Model(name="models/gemini-3.5-flash"),
    ]

    with (
        patch.object(
            pipe,
            "_get_or_create_genai_client",
            return_value=_developer_discovery_binding(),
        ),
        patch.object(
            pipe,
            "_fetch_models_from_client_internal",
            new=AsyncMock(return_value=listed_models),
        ),
    ):
        models = await pipe._get_genai_models(
            free_api_key="developer-key",
            whitelist_str="gemini-2.5-*",
            blacklist_str="*-pro",
        )

    assert [model["id"] for model in models] == ["gemini-2.5-flash"]
    await _clear_model_discovery_cache(pipe)


@pytest.mark.asyncio
async def test_combined_discovery_deduplicates_developer_models_before_merge(
    pipe_instance_fixture,
) -> None:
    pipe, _ = pipe_instance_fixture
    await _clear_model_discovery_cache(pipe)
    developer_models = [
        gemini_types.Model(
            name="models/gemini-2.5-flash",
            display_name="First Developer listing",
        ),
        gemini_types.Model(
            name="publishers/google/models/gemini-2.5-flash",
            display_name="Duplicate Developer listing",
        ),
    ]
    enterprise_binding = GenAIClientBinding(
        client=MagicMock(),
        identity=EndpointIdentity(
            service="enterprise",
            credential_fingerprint="credential",
            project="project",
            location="global",
            api_version="v1beta1",
        ),
    )

    with (
        patch.object(
            pipe,
            "_get_or_create_genai_client",
            side_effect=[_developer_discovery_binding(), enterprise_binding],
        ),
        patch.object(
            pipe,
            "_fetch_models_from_client_internal",
            new=AsyncMock(side_effect=[developer_models, []]),
        ),
    ):
        models = await pipe._get_genai_models(
            free_api_key="developer-key",
            use_enterprise=True,
            enterprise_project="project",
            enterprise_location="global",
        )

    assert models == [
        {
            "id": "gemini-2.5-flash",
            "name": "gemini-2.5-flash",
            "description": "Catalog-validated Gemini model.",
        }
    ]
    await _clear_model_discovery_cache(pipe)


@pytest.mark.asyncio
async def test_developer_discovery_cache_avoids_duplicate_provider_listing(
    pipe_instance_fixture,
) -> None:
    pipe, _ = pipe_instance_fixture
    await _clear_model_discovery_cache(pipe)
    fetch = AsyncMock(return_value=[gemini_types.Model(name="models/gemini-2.5-flash")])

    with (
        patch.object(
            pipe,
            "_get_or_create_genai_client",
            return_value=_developer_discovery_binding(),
        ),
        patch.object(pipe, "_fetch_models_from_client_internal", new=fetch),
    ):
        first = await pipe._get_genai_models(free_api_key="developer-cache-key")
        second = await pipe._get_genai_models(free_api_key="developer-cache-key")

    assert first == second
    assert [model["id"] for model in first] == ["gemini-2.5-flash"]
    fetch.assert_awaited_once()
    await _clear_model_discovery_cache(pipe)


@pytest.mark.asyncio
async def test_enterprise_discovery_hides_unverified_catalog_models(pipe_instance_fixture):
    pipe, _ = pipe_instance_fixture
    await _clear_model_discovery_cache(pipe)
    binding = GenAIClientBinding(
        client=MagicMock(),
        identity=EndpointIdentity(
            service="enterprise",
            credential_fingerprint="credential",
            project="project",
            location="global",
            api_version="v1beta1",
        ),
    )
    with (
        patch.object(pipe, "_get_or_create_genai_client", return_value=binding),
        patch.object(
            pipe,
            "_fetch_models_from_client_internal",
            new=AsyncMock(
                return_value=[
                    gemini_types.Model(
                        name=f"publishers/google/models/{model_id}",
                        display_name=model_id,
                    )
                    for model_id in CATALOG_MODEL_IDS
                ]
            ),
        ),
    ):
        models = await pipe._get_genai_models(
            use_enterprise=True,
            enterprise_project="project",
            enterprise_location="global",
        )

    assert models == []
    await _clear_model_discovery_cache(pipe)


@pytest.mark.asyncio
async def test_combined_discovery_does_not_leak_developer_policy_from_enterprise(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    await _clear_model_discovery_cache(pipe)
    developer_binding = GenAIClientBinding(
        client=MagicMock(),
        identity=_developer_identity(),
    )
    enterprise_binding = GenAIClientBinding(
        client=MagicMock(),
        identity=EndpointIdentity(
            service="enterprise",
            credential_fingerprint="credential",
            project="project",
            location="global",
            api_version="v1beta1",
        ),
    )
    enterprise_only_response = gemini_types.Model(
        name="publishers/google/models/gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
    )
    with (
        patch.object(
            pipe,
            "_get_or_create_genai_client",
            side_effect=[developer_binding, enterprise_binding],
        ),
        patch.object(
            pipe,
            "_fetch_models_from_client_internal",
            new=AsyncMock(side_effect=[[], [enterprise_only_response]]),
        ),
    ):
        models = await pipe._get_genai_models(
            free_api_key="developer-key",
            use_enterprise=True,
            enterprise_project="project",
            enterprise_location="global",
        )

    assert models == []
    await _clear_model_discovery_cache(pipe)


# endregion Test _get_genai_models


def _policy_builder(
    pipe: Pipe,
    *,
    inputs: list[str],
    files: bool,
    external_urls: bool,
) -> tuple[GeminiContentBuilder, MagicMock]:
    files_manager = MagicMock()
    files_manager.endpoint_identity = _developer_identity()
    builder = GeminiContentBuilder(
        messages_body=[],
        metadata_body={"chat_id": "chat"},  # type: ignore
        user_data={"id": "user", "email": "user@example.test"},  # type: ignore
        event_emitter=MagicMock(spec=EventEmitter),
        valves=pipe.valves,
        files_api_manager=files_manager,
        pdf_mitigation_manager=pipe.pdf_mitigation_manager,
        service_policy=_builder_service_policy(
            inputs=inputs, files=files, external_urls=external_urls
        ),
    )
    return builder, files_manager


@pytest.mark.asyncio
async def test_builder_enforces_selected_service_media_and_files_policy(pipe_instance_fixture):
    pipe, _ = pipe_instance_fixture
    denied, denied_files = _policy_builder(pipe, inputs=["text"], files=True, external_urls=False)
    with pytest.raises(ContentBuildError, match="does not support image input"):
        await denied._create_interaction_content_from_file_data(
            b"image", "image/png", None, asyncio.Queue()
        )
    denied_files.get_or_upload_file.assert_not_called()

    inline, inline_files = _policy_builder(
        pipe, inputs=["text", "image"], files=False, external_urls=False
    )
    content = await inline._create_interaction_content_from_file_data(
        b"image", "image/png", None, asyncio.Queue()
    )
    assert content["type"] == "image"
    assert content.get("data") == base64.b64encode(b"image").decode("ascii")
    inline_files.get_or_upload_file.assert_not_called()


def test_builder_enforces_selected_service_external_url_policy(pipe_instance_fixture):
    pipe, _ = pipe_instance_fixture
    builder, _ = _policy_builder(pipe, inputs=["text", "video"], files=False, external_urls=False)
    with pytest.raises(ContentBuildError, match="does not support external video URLs"):
        builder._interaction_content_from_youtube_uri("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


# region Test FilesAPIManager
def _mock_files_client():
    client = MagicMock()

    client.aio.files.get = AsyncMock()
    client.aio.files.upload = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_files_api_manager_recovers_when_byte_upload_already_exists():
    client = _mock_files_client()
    existing_file = gemini_types.File(
        name="files/owui-existing",
        uri="https://generativelanguage.googleapis.com/v1beta/files/owui-existing",
        mime_type="application/pdf",
        state=gemini_types.FileState.ACTIVE,
    )
    client.aio.files.get.side_effect = [
        _client_error(403, "PERMISSION_DENIED", "not found"),
        existing_file,
    ]
    client.aio.files.upload.side_effect = _client_error(
        409, "ALREADY_EXISTS", "owui-existing already exists."
    )
    event_emitter = MagicMock(spec=EventEmitter)
    manager = FilesAPIManager(
        client=client,
        endpoint_identity=_developer_identity(),
        file_cache=SimpleMemoryCache(serializer=NullSerializer()),
        id_hash_cache=SimpleMemoryCache(serializer=NullSerializer()),
        event_emitter=event_emitter,
    )

    result = await manager.get_or_upload_file(
        file_bytes=b"%PDF duplicate",
        mime_type="application/pdf",
        owui_file_id="owui-file-id",
    )

    assert result is existing_file
    assert client.aio.files.upload.await_count == 1
    assert client.aio.files.get.await_count == 2
    event_emitter.emit_toast.assert_not_called()


@pytest.mark.asyncio
async def test_files_api_manager_recovers_when_path_upload_already_exists(tmp_path):
    client = _mock_files_client()
    existing_file = gemini_types.File(
        name="files/owui-existing-path",
        uri="https://generativelanguage.googleapis.com/v1beta/files/owui-existing-path",
        mime_type="application/pdf",
        state=gemini_types.FileState.ACTIVE,
    )
    client.aio.files.get.side_effect = [
        _client_error(404, "NOT_FOUND", "not found"),
        existing_file,
    ]
    client.aio.files.upload.side_effect = _client_error(
        409, "ALREADY_EXISTS", "owui-existing-path already exists."
    )
    pdf_path = tmp_path / "duplicate.pdf"
    pdf_path.write_bytes(b"%PDF duplicate from path")
    event_emitter = MagicMock(spec=EventEmitter)
    manager = FilesAPIManager(
        client=client,
        endpoint_identity=_developer_identity(),
        file_cache=SimpleMemoryCache(serializer=NullSerializer()),
        id_hash_cache=SimpleMemoryCache(serializer=NullSerializer()),
        event_emitter=event_emitter,
    )

    result = await manager.get_or_upload_file_from_path(
        file_path=str(pdf_path),
        mime_type="application/pdf",
        owui_file_id="owui-file-id:path",
    )

    assert result is existing_file
    assert client.aio.files.upload.await_count == 1
    assert client.aio.files.get.await_count == 2
    event_emitter.emit_toast.assert_not_called()


@pytest.mark.asyncio
async def test_files_api_manager_retries_upload_conflict_recovery_get():
    client = _mock_files_client()
    existing_file = gemini_types.File(
        name="files/owui-eventual",
        uri="https://generativelanguage.googleapis.com/v1beta/files/owui-eventual",
        mime_type="application/pdf",
        state=gemini_types.FileState.ACTIVE,
    )
    client.aio.files.get.side_effect = [
        _client_error(404, "NOT_FOUND", "not found yet"),
        existing_file,
    ]
    event_emitter = MagicMock(spec=EventEmitter)
    manager = FilesAPIManager(
        client=client,
        endpoint_identity=_developer_identity(),
        file_cache=SimpleMemoryCache(serializer=NullSerializer()),
        id_hash_cache=SimpleMemoryCache(serializer=NullSerializer()),
        event_emitter=event_emitter,
    )

    result = await manager._recover_after_upload_conflict(
        content_hash="eventual-hash",
        deterministic_name="files/owui-eventual",
        owui_file_id="owui-file-id",
        attempts=2,
        retry_delay=0,
    )

    assert result is existing_file
    assert client.aio.files.get.await_count == 2
    event_emitter.emit_toast.assert_not_called()


@pytest.mark.asyncio
async def test_files_cache_never_reuses_uri_across_endpoint_scope():
    shared_cache = SimpleMemoryCache(serializer=NullSerializer())
    shared_id_cache = SimpleMemoryCache(serializer=NullSerializer())
    payload = b"same bytes across services"
    developer = FilesAPIManager(
        client=MagicMock(),
        endpoint_identity=_developer_identity(),
        file_cache=shared_cache,
        id_hash_cache=shared_id_cache,
        event_emitter=MagicMock(spec=EventEmitter),
    )
    enterprise_client = MagicMock()
    enterprise = FilesAPIManager(
        client=enterprise_client,
        endpoint_identity=EndpointIdentity(
            service="enterprise",
            credential_fingerprint="project-credential",
            api_version="v1beta1",
            project="project",
            location="global",
        ),
        file_cache=shared_cache,
        id_hash_cache=shared_id_cache,
        event_emitter=MagicMock(spec=EventEmitter),
    )
    content_hash = await developer._get_content_hash(payload, "owui-id")
    developer_file = gemini_types.File(
        name="files/developer",
        uri="https://files.example/developer",
        mime_type="image/png",
        state=gemini_types.FileState.ACTIVE,
    )
    await shared_cache.set(developer._get_file_cache_key(content_hash), developer_file)
    enterprise_file = gemini_types.File(
        name="files/enterprise",
        uri="https://files.example/enterprise",
        mime_type="image/png",
        state=gemini_types.FileState.ACTIVE,
    )
    enterprise_client.aio.files.get = AsyncMock(return_value=enterprise_file)

    result = await enterprise.get_or_upload_file(
        payload,
        "image/png",
        owui_file_id="owui-id",
    )

    assert developer._get_file_cache_key(content_hash) != enterprise._get_file_cache_key(
        content_hash
    )
    assert result.uri == enterprise_file.uri
    enterprise_client.aio.files.get.assert_awaited()


@pytest.mark.asyncio
async def test_files_public_state_policy_processing_failed_and_expired():
    client = MagicMock()
    emitter = MagicMock(spec=EventEmitter)
    manager = FilesAPIManager(
        client=client,
        endpoint_identity=_developer_identity(),
        file_cache=SimpleMemoryCache(serializer=NullSerializer()),
        id_hash_cache=SimpleMemoryCache(serializer=NullSerializer()),
        event_emitter=emitter,
    )
    processing = gemini_types.File(name="files/stateful", state=gemini_types.FileState.PROCESSING)
    active = gemini_types.File(
        name="files/stateful",
        uri="https://files.example/active",
        state=gemini_types.FileState.ACTIVE,
        expiration_time=datetime.now(UTC) + timedelta(minutes=5),
    )
    client.aio.files.get = AsyncMock(side_effect=[processing, active])

    with patch("plugins.pipes.gemini_manifold.asyncio.sleep", new_callable=AsyncMock):
        assert await manager._poll_for_active_state("files/stateful", "owui-id") is active

    failed = gemini_types.File(
        name="files/failed",
        state=gemini_types.FileState.FAILED,
        error=gemini_types.FileStatus(code=13, message="FILES_PROVIDER_CANARY"),
    )
    client.aio.files.get = AsyncMock(return_value=failed)
    with pytest.raises(FilesAPIError, match="could not process an uploaded file") as error_info:
        await manager._poll_for_active_state("files/failed", "owui-id")
    assert "FILES_PROVIDER_CANARY" not in str(error_info.value)
    assert "FILES_PROVIDER_CANARY" not in str(emitter.emit_toast.call_args_list)
    emitter.emit_toast.assert_called_with("Gemini could not process an uploaded file.", "error")
    assert manager._calculate_ttl(datetime.now(UTC) - timedelta(seconds=1)) == 0


# endregion Test FilesAPIManager


# region Test GeminiContentBuilder
@pytest.mark.asyncio
async def test_get_file_source_resolves_open_webui_storage_path(tmp_path):
    mock_files_module.reset_mock()
    mock_storage_module.reset_mock()
    local_path = tmp_path / "stored.pdf"
    local_path.write_bytes(b"%PDF stored")
    file_model = MagicMock()
    file_model.path = "s3://bucket/uploads/stored.pdf"
    file_model.meta = {"content_type": "application/pdf"}
    mock_files_module.Files.get_file_by_id_and_user_id = AsyncMock(return_value=file_model)
    mock_storage_module.Storage.get_file.return_value = str(local_path)

    source = await GeminiContentBuilder._get_file_source("stored-file-id", "owner-user")

    assert source is not None
    assert source.file_path == str(local_path)
    assert source.file_bytes is None
    assert source.mime_type == "application/pdf"
    mock_files_module.Files.get_file_by_id_and_user_id.assert_awaited_once_with(
        id="stored-file-id", user_id="owner-user"
    )
    mock_storage_module.Storage.get_file.assert_called_once_with("s3://bucket/uploads/stored.pdf")


@pytest.mark.asyncio
async def test_get_file_data_resolves_open_webui_storage_path(tmp_path):
    mock_files_module.reset_mock()
    mock_storage_module.reset_mock()
    local_path = tmp_path / "history.pdf"
    file_bytes = b"%PDF history"
    local_path.write_bytes(file_bytes)
    file_model = MagicMock()
    file_model.path = "https://account.blob.core.windows.net/container/history.pdf"
    file_model.meta = {"content_type": "application/pdf"}
    mock_files_module.Files.get_file_by_id_and_user_id = AsyncMock(return_value=file_model)
    mock_storage_module.Storage.get_file.return_value = str(local_path)

    data, mime_type = await GeminiContentBuilder._get_file_data("history-file-id", "owner-user")

    assert data == file_bytes
    assert mime_type == "application/pdf"
    mock_files_module.Files.get_file_by_id_and_user_id.assert_awaited_once_with(
        id="history-file-id", user_id="owner-user"
    )
    mock_storage_module.Storage.get_file.assert_called_once_with(
        "https://account.blob.core.windows.net/container/history.pdf"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reader_name", ["_get_file_source", "_get_file_data"])
async def test_local_file_read_denies_non_owner_before_storage_without_identifier_leak(
    reader_name: str,
) -> None:
    file_canary = "TRANSPLANTED_FILE_IDENTIFIER_CANARY"
    user_canary = "REQUESTING_USER_IDENTIFIER_CANARY"
    mock_files_module.reset_mock()
    mock_storage_module.reset_mock()
    mock_files_module.Files.get_file_by_id_and_user_id = AsyncMock(return_value=None)
    captured: list[str] = []
    handler_id = gemini_manifold_module.log.add(
        lambda message: captured.append(str(message)), format="{message}"
    )
    try:
        reader = getattr(GeminiContentBuilder, reader_name)
        with pytest.raises(LocalFileAccessError, match="file access was denied"):
            await reader(file_canary, user_canary)
    finally:
        gemini_manifold_module.log.remove(handler_id)

    mock_files_module.Files.get_file_by_id_and_user_id.assert_awaited_once_with(
        id=file_canary, user_id=user_canary
    )
    mock_storage_module.Storage.get_file.assert_not_called()
    emitted = "\n".join(captured)
    assert file_canary not in emitted
    assert user_canary not in emitted


@pytest.mark.asyncio
async def test_builder_build_contents_simple_user_text(pipe_instance_fixture):
    """A simple user message becomes one canonical user-input step."""
    # Reset mocks for test isolation
    mock_chats_module.reset_mock()
    mock_chats_module.Chats.get_chat_by_id_and_user_id.reset_mock()
    mock_misc_module.reset_mock()

    pipe_instance, _ = pipe_instance_fixture
    messages_body = [{"role": "user", "content": "Hello!"}]
    mock_event_emitter = MagicMock(spec=EventEmitter)
    mock_event_emitter.start_time = 1234567890.0  # Mock the required start_time
    mock_user_data = {
        "id": "test_user_id",
        "email": "test@example.com",
        "name": "Test User",
        "role": "user",
    }
    mock_files_api_manager = MagicMock()

    # The builder fetches chat history, mock it to return None for this test
    mock_chats_module.Chats.get_chat_by_id_and_user_id.return_value = None
    # The builder uses pop_system_message, mock its behavior
    mock_misc_module.pop_system_message.return_value = (None, messages_body)

    builder = GeminiContentBuilder(
        messages_body=messages_body,  # type: ignore
        metadata_body={"chat_id": "test_chat_id"},  # type: ignore
        user_data=mock_user_data,  # type: ignore
        event_emitter=mock_event_emitter,
        valves=pipe_instance.valves,
        files_api_manager=mock_files_api_manager,
        pdf_mitigation_manager=pipe_instance.pdf_mitigation_manager,
        service_policy=_builder_service_policy(),
    )

    steps = await builder.build_contents()

    mock_misc_module.pop_system_message.assert_called_once_with(messages_body)
    assert steps == [{"type": "user_input", "content": [{"type": "text", "text": "Hello!"}]}]
    mock_event_emitter.emit_toast.assert_called_once_with(ANY, "warning")


@pytest.mark.asyncio
async def test_builder_separates_system_instruction_and_preserves_multi_turn_steps(
    pipe_instance_fixture,
):
    pipe_instance, _ = pipe_instance_fixture
    messages = [
        {"role": "system", "content": "Be exact."},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "Follow-up"},
    ]
    remaining = messages[1:]
    mock_misc_module.pop_system_message.return_value = (messages[0], remaining)
    mock_chats_module.Chats.get_chat_by_id_and_user_id.return_value = None
    files_manager = MagicMock()
    files_manager.endpoint_identity = _developer_identity()
    builder = GeminiContentBuilder(
        messages_body=messages,  # type: ignore
        metadata_body={"chat_id": "chat"},  # type: ignore
        user_data={"id": "user", "email": "user@example.com"},  # type: ignore
        event_emitter=MagicMock(spec=EventEmitter),
        valves=pipe_instance.valves,
        files_api_manager=files_manager,
        pdf_mitigation_manager=pipe_instance.pdf_mitigation_manager,
        service_policy=_builder_service_policy(),
    )

    steps = await builder.build_contents()

    assert builder.system_prompt == "Be exact."
    assert steps == [
        {"type": "user_input", "content": [{"type": "text", "text": "Question"}]},
        {"type": "model_output", "content": [{"type": "text", "text": "Answer"}]},
        {"type": "user_input", "content": [{"type": "text", "text": "Follow-up"}]},
    ]


@pytest.mark.asyncio
async def test_builder_temp_chat_image_payload_uses_explicit_data(pipe_instance_fixture):
    pipe_instance, _ = pipe_instance_fixture
    image_bytes = b"temporary image"
    image_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]
    mock_misc_module.pop_system_message.return_value = (None, messages)
    files_manager = AsyncMock()
    files_manager.endpoint_identity = _developer_identity()
    builder = GeminiContentBuilder(
        messages_body=messages,  # type: ignore
        metadata_body={"chat_id": "local-temp-chat"},  # type: ignore
        user_data={"id": "user", "email": "user@example.com"},  # type: ignore
        event_emitter=MagicMock(spec=EventEmitter),
        valves=pipe_instance.valves,
        files_api_manager=files_manager,
        pdf_mitigation_manager=pipe_instance.pdf_mitigation_manager,
        service_policy=_builder_service_policy(),
    )

    steps = await builder.build_contents()

    assert steps == [
        {
            "type": "user_input",
            "content": [
                {"type": "text", "text": "Describe"},
                {
                    "type": "image",
                    "mime_type": "image/png",
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                },
            ],
        }
    ]
    files_manager.get_or_upload_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_builder_history_follows_active_parent_branch_and_excludes_sibling(
    pipe_instance_fixture,
):
    pipe_instance, _ = pipe_instance_fixture
    messages = [
        {"role": "user", "content": "root"},
        {"role": "assistant", "content": "edited active answer"},
        {"role": "user", "content": "active follow-up"},
    ]
    mock_misc_module.pop_system_message.return_value = (None, messages)
    chat = MagicMock()
    chat.chat = {
        "history": {
            "currentId": "active-user",
            "messages": {
                "root": {"id": "root", "parentId": None, "role": "user", "content": "root"},
                "active-assistant": {
                    "id": "active-assistant",
                    "parentId": "root",
                    "role": "assistant",
                    "content": "edited active answer",
                },
                "active-user": {
                    "id": "active-user",
                    "parentId": "active-assistant",
                    "role": "user",
                    "content": "active follow-up",
                },
                "sibling": {
                    "id": "sibling",
                    "parentId": "root",
                    "role": "assistant",
                    "content": "wrong branch",
                },
            },
        }
    }
    mock_chats_module.Chats.get_chat_by_id_and_user_id.return_value = chat
    files_manager = MagicMock()
    files_manager.endpoint_identity = _developer_identity()
    builder = GeminiContentBuilder(
        messages_body=messages,  # type: ignore
        metadata_body={"chat_id": "chat"},  # type: ignore
        user_data={"id": "user", "email": "user@example.com"},  # type: ignore
        event_emitter=MagicMock(spec=EventEmitter),
        valves=pipe_instance.valves,
        files_api_manager=files_manager,
        pdf_mitigation_manager=pipe_instance.pdf_mitigation_manager,
        service_policy=_builder_service_policy(),
    )

    steps = await builder.build_contents()

    assert [message["id"] for message in builder.messages_db or []] == [
        "root",
        "active-assistant",
        "active-user",
    ]
    assert steps[1] == {
        "type": "model_output",
        "content": [{"type": "text", "text": "edited active answer"}],
    }
    assert "wrong branch" not in str(steps)


@pytest.mark.asyncio
async def test_builder_build_contents_youtube_link_mixed_with_text(
    pipe_instance_fixture,
):
    """
    Tests that text around a YouTube URL becomes ordered Interactions content.
    """
    # Reset mocks for test isolation
    mock_chats_module.reset_mock()
    mock_misc_module.reset_mock()

    pipe_instance, _ = pipe_instance_fixture
    # Exercise canonical external-video content on the Developer endpoint.
    pipe_instance.valves.USE_ENTERPRISE = False

    # Arrange: Inputs
    youtube_url = "https://www.youtube.com/watch?v=kpwNjdEPz7E"
    text_before_raw = "Look at this: "
    text_after_raw = " it's great!"
    text_before_stripped = text_before_raw.strip()
    text_after_stripped = text_after_raw.strip()
    user_content_string = f"{text_before_raw}{youtube_url}{text_after_raw}"
    messages_body = [{"role": "user", "content": user_content_string}]
    mock_event_emitter = MagicMock(spec=EventEmitter)
    mock_event_emitter.start_time = 1234567890.0  # Mock the required start_time
    mock_user_data = {
        "id": "test_user_id",
        "email": "test@example.com",
        "name": "Test User",
        "role": "user",
    }
    mock_files_api_manager = MagicMock()

    # Mock DB and system prompt extraction
    mock_chats_module.Chats.get_chat_by_id_and_user_id.return_value = None
    mock_misc_module.pop_system_message.return_value = (None, messages_body)

    builder = GeminiContentBuilder(
        messages_body=messages_body,  # type: ignore
        metadata_body={"chat_id": "test_chat_id"},  # type: ignore
        user_data=mock_user_data,  # type: ignore
        event_emitter=mock_event_emitter,
        valves=pipe_instance.valves,
        files_api_manager=mock_files_api_manager,  # Pass the new mock
        pdf_mitigation_manager=pipe_instance.pdf_mitigation_manager,
        service_policy=_builder_service_policy(),
    )

    steps = await builder.build_contents()

    assert steps == [
        {
            "type": "user_input",
            "content": [
                {"type": "text", "text": text_before_stripped},
                {"type": "video", "uri": youtube_url, "mime_type": "video/mp4"},
                {"type": "text", "text": text_after_stripped},
            ],
        }
    ]
    mock_event_emitter.emit_toast.assert_called_once_with(ANY, "warning")


@pytest.mark.asyncio
async def test_builder_build_contents_user_text_with_pdf(pipe_instance_fixture):
    """
    Tests conversion of a user message with text and an attached PDF file.
    """
    # Reset mocks for test isolation
    mock_chats_module.reset_mock()
    mock_chats_module.Chats.get_chat_by_id_and_user_id.reset_mock()
    mock_misc_module.reset_mock()

    pipe_instance, _ = pipe_instance_fixture
    pipe_instance.valves.PDF_LIMIT_MITIGATION = False

    # Arrange: Inputs
    user_text_content = "Please analyze this PDF."
    pdf_file_id = "test-pdf-id-001"
    fake_pdf_bytes = b"%PDF-1.4 fake content..."
    pdf_mime_type = "application/pdf"
    messages_body = [{"role": "user", "content": user_text_content}]
    mock_event_emitter = MagicMock(spec=EventEmitter)
    mock_event_emitter.start_time = 1234567890.0  # Mock the required start_time
    mock_user_data = {
        "id": "test_user_id",
        "email": "test@example.com",
        "name": "Test User",
        "role": "user",
    }

    # Mock the chat object returned by the DB
    mock_chat_from_db = MagicMock()
    mock_chat_from_db.chat = {
        "history": {
            "currentId": "f72886c4-5420-46ce-bb0b-b95286835d51",
            "messages": {
                "f72886c4-5420-46ce-bb0b-b95286835d51": {
                    "id": "742262d1-ea16-41c5-9cf4-2e07006decf1",
                    "parentId": None,
                    "role": "user",
                    "content": user_text_content,
                    "files": [
                        {
                            "id": pdf_file_id,
                            "type": "file",
                            "content_type": pdf_mime_type,
                        }
                    ],
                },
            },
        }
    }

    # Mock the DB call and system prompt extraction
    mock_misc_module.pop_system_message.return_value = (None, messages_body)

    pipe_instance.valves.PDF_LIMIT_MITIGATION = False
    mock_files_api_manager = AsyncMock()
    mock_files_api_manager.endpoint_identity = _developer_identity()
    mock_gemini_file = MagicMock()
    mock_gemini_file.uri = "gs://fake-bucket/fake-file.pdf"
    mock_gemini_file.mime_type = pdf_mime_type
    mock_files_api_manager.get_or_upload_file.return_value = mock_gemini_file

    builder = GeminiContentBuilder(
        messages_body=messages_body,  # type: ignore
        metadata_body={
            "chat_id": "test_chat_id",
            "features": {"upload_documents": True},  # type: ignore
        },
        user_data=mock_user_data,  # type: ignore
        event_emitter=mock_event_emitter,
        valves=pipe_instance.valves,
        files_api_manager=mock_files_api_manager,
        pdf_mitigation_manager=pipe_instance.pdf_mitigation_manager,
        service_policy=_builder_service_policy(),
    )

    # Resolve the OWUI attachment without touching host storage.
    with (
        patch.object(
            mock_chats_module.Chats,
            "get_chat_by_id_and_user_id",
            new=AsyncMock(return_value=mock_chat_from_db),
        ) as mock_get_chat,
        patch(
            "plugins.pipes.gemini_manifold.GeminiContentBuilder._get_file_source",
            new_callable=AsyncMock,
            return_value=LocalFileSource(
                file_bytes=fake_pdf_bytes,
                file_path=None,
                mime_type=pdf_mime_type,
            ),
        ) as mock_get_file_source,
    ):
        # Act
        contents = await builder.build_contents()

        # Assert
        mock_get_chat.assert_called_once_with(id="test_chat_id", user_id="test_user_id")
        mock_get_file_source.assert_awaited_once_with(pdf_file_id, "test_user_id")

        mock_files_api_manager.get_or_upload_file.assert_awaited_once_with(
            file_bytes=fake_pdf_bytes,
            mime_type=pdf_mime_type,
            owui_file_id=pdf_file_id,
            status_queue=ANY,
        )

        # This assertion is no longer needed as we are not mocking `from_text`
        # mock_part_from_text.assert_called_once_with(text=user_text_content)

        assert contents == [
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "document",
                        "uri": mock_gemini_file.uri,
                        "mime_type": pdf_mime_type,
                    },
                    {"type": "text", "text": user_text_content},
                ],
            }
        ]

        mock_event_emitter.assert_not_called()


@pytest.mark.asyncio
async def test_builder_build_contents_with_multiple_pdf_attachments(
    pipe_instance_fixture,
):
    """
    Tests that a user turn with multiple attached PDFs preserves every file part.
    """
    mock_chats_module.reset_mock()
    mock_misc_module.reset_mock()

    pipe_instance, _ = pipe_instance_fixture
    pipe_instance.valves.PDF_LIMIT_MITIGATION = False

    user_text_content = "Compare these PDFs."
    pdf_mime_type = "application/pdf"
    first_pdf_id = "first-pdf-id"
    second_pdf_id = "second-pdf-id"
    messages_body = [{"role": "user", "content": user_text_content}]
    mock_event_emitter = MagicMock(spec=EventEmitter)
    mock_event_emitter.start_time = 1234567890.0
    mock_user_data = {
        "id": "test_user_id",
        "email": "test@example.com",
        "name": "Test User",
        "role": "user",
    }

    mock_chat_from_db = MagicMock()
    mock_chat_from_db.chat = {
        "history": {
            "currentId": "current-user-message",
            "messages": {
                "current-user-message": {
                    "id": "current-user-message",
                    "parentId": None,
                    "role": "user",
                    "content": user_text_content,
                    "files": [
                        {
                            "id": first_pdf_id,
                            "name": "first.pdf",
                            "type": "file",
                            "content_type": pdf_mime_type,
                        },
                        {
                            "id": second_pdf_id,
                            "name": "second.pdf",
                            "type": "file",
                            "content_type": pdf_mime_type,
                        },
                    ],
                },
            },
        }
    }
    mock_chats_module.Chats.get_chat_by_id_and_user_id.return_value = mock_chat_from_db
    mock_misc_module.pop_system_message.return_value = (None, messages_body)

    mock_files_api_manager = AsyncMock()
    mock_files_api_manager.endpoint_identity = _developer_identity()
    first_gemini_file = MagicMock()
    first_gemini_file.uri = "gs://fake-bucket/first.pdf"
    first_gemini_file.mime_type = pdf_mime_type
    second_gemini_file = MagicMock()
    second_gemini_file.uri = "gs://fake-bucket/second.pdf"
    second_gemini_file.mime_type = pdf_mime_type
    mock_files_api_manager.get_or_upload_file.side_effect = [
        first_gemini_file,
        second_gemini_file,
    ]

    builder = GeminiContentBuilder(
        messages_body=messages_body,  # type: ignore
        metadata_body={
            "chat_id": "test_chat_id",
            "features": {"upload_documents": True},  # type: ignore
        },
        user_data=mock_user_data,  # type: ignore
        event_emitter=mock_event_emitter,
        valves=pipe_instance.valves,
        files_api_manager=mock_files_api_manager,
        pdf_mitigation_manager=pipe_instance.pdf_mitigation_manager,
        service_policy=_builder_service_policy(),
    )

    with patch(
        "plugins.pipes.gemini_manifold.GeminiContentBuilder._get_file_source",
        new_callable=AsyncMock,
        side_effect=[
            LocalFileSource(
                file_bytes=b"%PDF first",
                file_path=None,
                mime_type=pdf_mime_type,
            ),
            LocalFileSource(
                file_bytes=b"%PDF second",
                file_path=None,
                mime_type=pdf_mime_type,
            ),
        ],
    ) as mock_get_file_source:
        contents = await builder.build_contents()

    assert contents == [
        {
            "type": "user_input",
            "content": [
                {
                    "type": "document",
                    "uri": first_gemini_file.uri,
                    "mime_type": pdf_mime_type,
                },
                {
                    "type": "document",
                    "uri": second_gemini_file.uri,
                    "mime_type": pdf_mime_type,
                },
                {"type": "text", "text": user_text_content},
            ],
        }
    ]
    assert mock_get_file_source.await_args_list == [
        call(first_pdf_id, "test_user_id"),
        call(second_pdf_id, "test_user_id"),
    ]
    assert mock_files_api_manager.get_or_upload_file.await_count == 2
    assert [
        await_call.kwargs["owui_file_id"]
        for await_call in mock_files_api_manager.get_or_upload_file.await_args_list
    ] == [first_pdf_id, second_pdf_id]


@pytest.mark.asyncio
async def test_create_interaction_content_optimizes_pdf_with_synthetic_id(
    pipe_instance_fixture, tmp_path
):
    """
    Tests that a compressed single-PDF output is uploaded under a synthetic ID,
    avoiding stale original-file ID hash mappings.
    """
    pipe_instance, _ = pipe_instance_fixture
    pdf_file_id = "test-pdf-id-002"
    original_pdf_bytes = b"%PDF original oversized"
    optimized_pdf_bytes = b"%PDF optimized"
    optimized_pdf_path = tmp_path / "optimized.pdf"
    optimized_pdf_path.write_bytes(optimized_pdf_bytes)
    pdf_mime_type = "application/pdf"
    original_hash = "optimized-hash"

    mock_event_emitter = MagicMock(spec=EventEmitter)
    mock_pdf_mitigation_manager = MagicMock(spec=PDFMitigationManager)
    mock_pdf_mitigation_manager.prepare = AsyncMock(
        return_value=PDFMitigationOutcome(
            original_hash=original_hash,
            result=PreparedPDFResult(
                parts=[
                    PreparedPDFPart(
                        path=str(optimized_pdf_path),
                        size=len(optimized_pdf_bytes),
                        start_page=1,
                        end_page=12,
                    )
                ],
                page_count=12,
                was_mitigated=True,
            ),
        )
    )
    mock_files_api_manager = AsyncMock()
    mock_files_api_manager.endpoint_identity = _developer_identity()
    mock_gemini_file = MagicMock()
    mock_gemini_file.uri = "gs://fake-bucket/optimized.pdf"
    mock_gemini_file.mime_type = pdf_mime_type
    mock_files_api_manager.get_or_upload_file_from_path.return_value = mock_gemini_file

    builder = GeminiContentBuilder(
        messages_body=[],
        metadata_body={"chat_id": "test_chat_id", "features": {"upload_documents": True}},  # type: ignore
        user_data={"id": "test_user_id", "email": "test@example.com"},  # type: ignore
        event_emitter=mock_event_emitter,
        valves=pipe_instance.valves,
        files_api_manager=mock_files_api_manager,
        pdf_mitigation_manager=mock_pdf_mitigation_manager,
        service_policy=_builder_service_policy(),
    )

    parts = await builder._create_interaction_contents_from_file_data(
        file_bytes=original_pdf_bytes,
        mime_type=pdf_mime_type,
        owui_file_id=pdf_file_id,
        status_queue=asyncio.Queue(),
        source_name="large.pdf",
    )

    assert len(parts) == 1
    assert parts[0] == {
        "type": "document",
        "uri": mock_gemini_file.uri,
        "mime_type": pdf_mime_type,
    }
    mock_pdf_mitigation_manager.prepare.assert_awaited_once_with(
        file_bytes=original_pdf_bytes,
        file_path=None,
    )
    mock_files_api_manager.get_or_upload_file_from_path.assert_awaited_once()
    upload_kwargs = mock_files_api_manager.get_or_upload_file_from_path.await_args.kwargs
    assert upload_kwargs["file_path"] == str(optimized_pdf_path)
    assert upload_kwargs["mime_type"] == pdf_mime_type
    assert upload_kwargs["owui_file_id"].startswith(f"{pdf_file_id}:pdf:")
    assert upload_kwargs["owui_file_id"].endswith(":optimized")
    mock_event_emitter.emit_status.assert_called_once_with(
        "Optimized PDF to fit Gemini API limits.",
        done=True,
        indent_level=1,
    )


@pytest.mark.asyncio
async def test_create_interaction_content_splits_pdf_in_order(pipe_instance_fixture, tmp_path):
    """
    Tests that split PDFs produce an instruction text part followed by ordered
    file parts uploaded with distinct synthetic IDs.
    """
    pipe_instance, _ = pipe_instance_fixture
    pdf_file_id = "test-pdf-id-003"
    original_pdf_bytes = b"%PDF original huge"
    chunks = [b"%PDF chunk 1", b"%PDF chunk 2", b"%PDF chunk 3"]
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = tmp_path / f"chunk-{i + 1}.pdf"
        chunk_path.write_bytes(chunk)
        chunk_paths.append(chunk_path)
    pdf_mime_type = "application/pdf"
    original_hash = "split-hash"

    mock_event_emitter = MagicMock(spec=EventEmitter)
    mock_pdf_mitigation_manager = MagicMock(spec=PDFMitigationManager)
    mock_pdf_mitigation_manager.prepare = AsyncMock(
        return_value=PDFMitigationOutcome(
            original_hash=original_hash,
            result=PreparedPDFResult(
                parts=[
                    PreparedPDFPart(
                        path=str(path),
                        size=len(chunk),
                        start_page=(i * 46) + 1,
                        end_page=(i + 1) * 46,
                    )
                    for i, (path, chunk) in enumerate(zip(chunk_paths, chunks, strict=True))
                ],
                page_count=2401,
                was_mitigated=True,
            ),
        )
    )
    mock_files_api_manager = AsyncMock()
    mock_files_api_manager.endpoint_identity = _developer_identity()
    mock_gemini_files = []
    for i in range(len(chunks)):
        mock_file = MagicMock()
        mock_file.uri = f"gs://fake-bucket/chunk-{i + 1}.pdf"
        mock_file.mime_type = pdf_mime_type
        mock_gemini_files.append(mock_file)
    mock_files_api_manager.get_or_upload_file_from_path.side_effect = mock_gemini_files

    builder = GeminiContentBuilder(
        messages_body=[],
        metadata_body={"chat_id": "test_chat_id", "features": {"upload_documents": True}},  # type: ignore
        user_data={"id": "test_user_id", "email": "test@example.com"},  # type: ignore
        event_emitter=mock_event_emitter,
        valves=pipe_instance.valves,
        files_api_manager=mock_files_api_manager,
        pdf_mitigation_manager=mock_pdf_mitigation_manager,
        service_policy=_builder_service_policy(),
    )

    parts = await builder._create_interaction_contents_from_file_data(
        file_bytes=original_pdf_bytes,
        mime_type=pdf_mime_type,
        owui_file_id=pdf_file_id,
        status_queue=asyncio.Queue(),
        source_name="very-large.pdf",
    )

    assert len(parts) == 4
    mock_pdf_mitigation_manager.prepare.assert_awaited_once_with(
        file_bytes=original_pdf_bytes,
        file_path=None,
    )
    instruction = parts[0]
    assert instruction["type"] == "text"
    instruction_text = instruction["text"]
    assert "very-large.pdf" in instruction_text
    assert "3 consecutive attachments" in instruction_text
    assert "PDF 'very-large.pdf', attachment 1: original document pages 1-46" in instruction_text
    assert "PDF 'very-large.pdf', attachment 2: original document pages 47-92" in instruction_text
    assert "do not restart page numbering at 1" in instruction_text
    assert [dict(part).get("uri") for part in parts[1:]] == [
        "gs://fake-bucket/chunk-1.pdf",
        "gs://fake-bucket/chunk-2.pdf",
        "gs://fake-bucket/chunk-3.pdf",
    ]
    assert mock_files_api_manager.get_or_upload_file_from_path.await_count == 3
    upload_ids = [
        await_call.kwargs["owui_file_id"]
        for await_call in mock_files_api_manager.get_or_upload_file_from_path.await_args_list
    ]
    assert upload_ids[0].endswith(":part-0001-of-0003")
    assert upload_ids[1].endswith(":part-0002-of-0003")
    assert upload_ids[2].endswith(":part-0003-of-0003")
    mock_event_emitter.emit_status.assert_called_once_with(
        "Optimized and split PDF into 3 parts.",
        done=True,
        indent_level=1,
    )


@pytest.mark.asyncio
async def test_builder_force_raw_media_is_explicit_base64(pipe_instance_fixture):
    pipe_instance, _ = pipe_instance_fixture
    files_manager = AsyncMock()
    files_manager.endpoint_identity = _developer_identity()
    builder = GeminiContentBuilder(
        messages_body=[],
        metadata_body={"chat_id": "chat"},  # type: ignore
        user_data={"id": "user", "email": "user@example.com"},  # type: ignore
        event_emitter=MagicMock(spec=EventEmitter),
        valves=pipe_instance.valves,
        files_api_manager=files_manager,
        pdf_mitigation_manager=pipe_instance.pdf_mitigation_manager,
        service_policy=_builder_service_policy(),
    )
    payload = b"\x89PNG\r\nexplicit bytes"

    content = await builder._create_interaction_content_from_file_data(
        file_bytes=payload,
        mime_type="image/png",
        owui_file_id="image-id",
        status_queue=asyncio.Queue(),
        force_raw=True,
    )

    assert content == {
        "type": "image",
        "mime_type": "image/png",
        "data": base64.b64encode(payload).decode("ascii"),
    }
    files_manager.get_or_upload_file.assert_not_awaited()


@pytest.mark.parametrize(
    ("mime_type", "content_type"),
    [
        ("image/png", "image"),
        ("audio/wav", "audio"),
        ("video/mp4", "video"),
        ("application/pdf", "document"),
    ],
)
def test_interaction_media_factory_supports_data_and_uri(
    mime_type: str,
    content_type: str,
) -> None:
    encoded = base64.b64encode(b"media").decode("ascii")

    assert GeminiContentBuilder._media_content(mime_type=mime_type, data=encoded) == {
        "type": content_type,
        "mime_type": mime_type,
        "data": encoded,
    }
    assert GeminiContentBuilder._media_content(
        mime_type=mime_type, uri="https://files.example/media"
    ) == {
        "type": content_type,
        "mime_type": mime_type,
        "uri": "https://files.example/media",
    }


@pytest.mark.asyncio
async def test_builder_rehydrates_generated_local_image_for_exact_replay(
    pipe_instance_fixture,
):
    pipe_instance, _ = pipe_instance_fixture
    files_manager = MagicMock()
    files_manager.endpoint_identity = _developer_identity()
    builder = GeminiContentBuilder(
        messages_body=[],
        metadata_body={"chat_id": "chat"},  # type: ignore
        user_data={"id": "user", "email": "user@example.com"},  # type: ignore
        event_emitter=MagicMock(spec=EventEmitter),
        valves=pipe_instance.valves,
        files_api_manager=files_manager,
        pdf_mitigation_manager=pipe_instance.pdf_mitigation_manager,
        service_policy=_builder_service_policy(),
    )
    image_bytes = b"generated image bytes"
    with patch.object(
        builder,
        "_get_file_data",
        new_callable=AsyncMock,
        return_value=(image_bytes, "image/png"),
    ) as get_file_data:
        replay = await builder._rehydrate_assistant_steps(
            [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "image",
                            "mime_type": "image/png",
                            "uri": "/api/v1/files/generated-image/content",
                        }
                    ],
                }
            ],
            files_manager.endpoint_identity.scope,
        )

    assert replay == [
        {
            "type": "model_output",
            "content": [
                {
                    "type": "image",
                    "mime_type": "image/png",
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                }
            ],
        }
    ]
    get_file_data.assert_awaited_once_with("generated-image", "user")


def test_builder_rejects_request_over_100_mib_preflight(pipe_instance_fixture):
    pipe_instance, _ = pipe_instance_fixture
    files_manager = MagicMock()
    files_manager.endpoint_identity = _developer_identity()
    builder = GeminiContentBuilder(
        messages_body=[],
        metadata_body={"chat_id": "chat"},  # type: ignore
        user_data={"id": "user", "email": "user@example.com"},  # type: ignore
        event_emitter=MagicMock(spec=EventEmitter),
        valves=pipe_instance.valves,
        files_api_manager=files_manager,
        pdf_mitigation_manager=pipe_instance.pdf_mitigation_manager,
        service_policy=_builder_service_policy(),
    )
    oversized_text = "x" * (100 * 1024 * 1024)

    with pytest.raises(ContentBuildError, match="exceeding the 100 MiB limit"):
        builder._validate_request_size(
            [
                {
                    "type": "user_input",
                    "content": [{"type": "text", "text": oversized_text}],
                }
            ]
        )


@pytest.mark.asyncio
async def test_builder_exact_replay_preserves_signature_and_edit_drops_ledger(
    pipe_instance_fixture,
):
    pipe_instance, _ = pipe_instance_fixture
    files_manager = MagicMock()
    files_manager.endpoint_identity = _developer_identity()
    builder = GeminiContentBuilder(
        messages_body=[],
        metadata_body={"chat_id": "chat"},  # type: ignore
        user_data={"id": "user", "email": "user@example.com"},  # type: ignore
        event_emitter=MagicMock(spec=EventEmitter),
        valves=pipe_instance.valves,
        files_api_manager=files_manager,
        pdf_mitigation_manager=pipe_instance.pdf_mitigation_manager,
        service_policy=_builder_service_policy(),
    )
    signature = base64.b64encode(b"opaque signed thought").decode("ascii")
    stored_steps: list[dict[str, object]] = [
        {"type": "thought", "signature": signature},
        {"type": "model_output", "content": [{"type": "text", "text": "answer"}]},
    ]
    message_db = {
        "gemini_interaction": {
            "version": 1,
            "interaction_id": "stored",
            "endpoint_scope": files_manager.endpoint_identity.scope,
            "model_id": "gemini-test",
            "store": True,
            "status": "completed",
            "steps": stored_steps,
            "visible_content": "answer",
            "usage": {},
            "last_event_id": None,
            "grounding": {
                "protocol_version": 1,
                "visible_content_sha256": "",
                "grounded_text_sha256": "",
            },
        }
    }

    exact = await builder._process_assistant_message(
        0,
        {"role": "assistant", "content": "answer"},  # type: ignore
        message_db,  # type: ignore
        None,
        asyncio.Queue(),
    )
    edited = await builder._process_assistant_message(
        0,
        {"role": "assistant", "content": "edited answer"},  # type: ignore
        message_db,  # type: ignore
        None,
        asyncio.Queue(),
    )

    assert exact == stored_steps
    assert edited == [
        {
            "type": "model_output",
            "content": [{"type": "text", "text": "edited answer"}],
        }
    ]


@pytest.mark.asyncio
async def test_builder_replay_rejects_unknown_and_foreign_endpoint_scope(
    pipe_instance_fixture,
):
    pipe_instance, _ = pipe_instance_fixture
    files_manager = MagicMock()
    files_manager.endpoint_identity = _developer_identity()
    builder = GeminiContentBuilder(
        messages_body=[],
        metadata_body={"chat_id": "chat"},  # type: ignore
        user_data={"id": "user", "email": "user@example.com"},  # type: ignore
        event_emitter=MagicMock(spec=EventEmitter),
        valves=pipe_instance.valves,
        files_api_manager=files_manager,
        pdf_mitigation_manager=pipe_instance.pdf_mitigation_manager,
        service_policy=_builder_service_policy(),
    )

    with pytest.raises(ValueError, match="unknown step or content"):
        await builder._rehydrate_assistant_steps(
            [{"type": "future_step", "payload": "unsafe"}],
            files_manager.endpoint_identity.scope,
        )

    with pytest.raises(ValueError, match="different endpoint identity"):
        await builder._rehydrate_assistant_steps(
            [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "document",
                            "mime_type": "application/pdf",
                            "uri": "https://files.example/foreign.pdf",
                        }
                    ],
                }
            ],
            "developer:other-credential:v1:model",
        )


@pytest.mark.asyncio
async def test_pdf_mitigation_manager_reuses_cached_result(tmp_path):
    """
    Tests that repeated processing of the same oversized PDF skips expensive
    compression/splitting and avoids writing another byte-backed temp source.
    """
    manager = PDFMitigationManager()
    original_pdf_bytes = b"%PDF original huge repeated"
    chunks = [b"%PDF cached chunk 1", b"%PDF cached chunk 2"]
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = tmp_path / f"cached-chunk-{i + 1}.pdf"
        chunk_path.write_bytes(chunk)
        chunk_paths.append(chunk_path)

    with (
        patch.object(
            GeminiPDFProcessor,
            "prepare_to_directory",
            return_value=PreparedPDFResult(
                parts=[
                    PreparedPDFPart(
                        path=str(path),
                        size=len(chunk),
                        start_page=(i * 600) + 1,
                        end_page=(i + 1) * 600,
                    )
                    for i, (path, chunk) in enumerate(zip(chunk_paths, chunks, strict=True))
                ],
                page_count=1200,
                was_mitigated=True,
            ),
        ) as mock_prepare,
        patch.object(
            manager,
            "_write_temp_source",
            wraps=manager._write_temp_source,
        ) as mock_write_temp_source,
    ):
        first_outcome = await manager.prepare(
            file_bytes=original_pdf_bytes,
            file_path=None,
        )
        second_outcome = await manager.prepare(
            file_bytes=original_pdf_bytes,
            file_path=None,
        )

    assert first_outcome is not None
    assert second_outcome is not None
    assert mock_prepare.call_count == 1
    assert mock_write_temp_source.call_count == 1
    assert first_outcome.result is second_outcome.result
    assert [part.path for part in first_outcome.result.parts] == [str(path) for path in chunk_paths]


def test_pdf_processor_rejects_single_page_over_limit(monkeypatch):
    """
    Tests the dynamic split edge case where even one page cannot fit.
    """
    processor = GeminiPDFProcessor(max_bytes=10, target_bytes=8, max_pages=1000)

    class FakePages:
        def __len__(self):
            return 1

        def __getitem__(self, item):
            return ["fake-page"]

    fake_pdf = MagicMock()
    fake_pdf.pages = FakePages()

    class FakePdfContext:
        def __enter__(self):
            return fake_pdf

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(
        processor,
        "_open_pdf",
        MagicMock(return_value=FakePdfContext()),
    )
    monkeypatch.setattr(
        processor,
        "_save_page_range",
        MagicMock(return_value=b"this-page-is-too-large"),
    )

    with pytest.raises(PDFProcessingError, match="single PDF page"):
        processor._split_pdf(MagicMock(), b"%PDF fake")


def test_pdf_processor_splits_real_pdf_by_page_limit(tmp_path):
    """
    Tests page-count mitigation against real PDF bytes.
    """
    pikepdf = pytest.importorskip("pikepdf")
    source_pdf = pikepdf.Pdf.new()
    for _ in range(5):
        source_pdf.add_blank_page(page_size=(72, 72))
    source_path = tmp_path / "source.pdf"
    source_pdf.save(source_path)

    processor = GeminiPDFProcessor(
        max_bytes=1024 * 1024,
        target_bytes=1024 * 1024,
        max_pages=2,
    )
    with patch.object(
        processor,
        "_optimize_pdf_to_path",
        side_effect=AssertionError("page-count-only split should not optimize first"),
    ):
        result = processor.prepare_to_directory(str(source_path), str(tmp_path / "out"))

    assert result.page_count == 5
    assert result.was_mitigated is True
    assert len(result.parts) == 3
    page_counts = [processor._count_pages_from_path(pikepdf, part.path) for part in result.parts]
    assert page_counts == [2, 2, 1]
    assert [(part.start_page, part.end_page) for part in result.parts] == [
        (1, 2),
        (3, 4),
        (5, 5),
    ]
    assert "pages-000001-000002" in result.parts[0].path


@pytest.mark.asyncio
async def test_build_contents_raises_content_build_error(pipe_instance_fixture):
    """
    Tests that concurrent content-building failures are surfaced to the pipe
    instead of being silently dropped.
    """
    pipe_instance, _ = pipe_instance_fixture
    mock_event_emitter = MagicMock(spec=EventEmitter)
    mock_files_api_manager = AsyncMock()
    mock_files_api_manager.endpoint_identity = _developer_identity()

    builder = GeminiContentBuilder(
        messages_body=[{"role": "user", "content": "hello"}],  # type: ignore
        metadata_body={"chat_id": "local", "features": {"upload_documents": True}},  # type: ignore
        user_data={"id": "test_user_id", "email": "test@example.com"},  # type: ignore
        event_emitter=mock_event_emitter,
        valves=pipe_instance.valves,
        files_api_manager=mock_files_api_manager,
        pdf_mitigation_manager=pipe_instance.pdf_mitigation_manager,
        service_policy=_builder_service_policy(),
    )

    with (
        patch.object(
            builder,
            "_process_message_turn",
            new_callable=AsyncMock,
            side_effect=PDFProcessingError("failed to process PDF"),
        ),
        pytest.raises(ContentBuildError, match="failed to process PDF"),
    ):
        await builder.build_contents()


# endregion Test GeminiContentBuilder


def _installed_model_catalog() -> dict:
    catalog_path = Path(__file__).parents[1] / "plugins" / "pipes" / "gemini_models.yaml"
    catalog = CompanionModelCatalog.model_validate(
        yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    )
    return CompanionCatalogEnvelope.from_catalog(catalog).model_dump(
        mode="json", exclude_none=False
    )


def _public_pipe_harness(
    pipe: Pipe, scripted: FakeInteractions
) -> tuple[Body, dict[str, object], MagicMock, GenAIClientBinding]:
    model_id = "gemini-2.5-flash"
    app = FastAPI()
    app.state._state["gemini_model_catalog"] = _installed_model_catalog()
    request = MagicMock()
    request.app = app
    body = cast(
        Body,
        {
            "model": f"gemini_manifold_google_genai.{model_id}",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    metadata = {
        "chat_id": "local-harness",
        "message_id": "message-harness",
        "features": {"gemini_manifold_companion_version": "3.0.0"},
    }
    fake_client = FakeGenAIClient(scripted)
    binding = GenAIClientBinding(
        client=cast(genai.Client, fake_client),
        identity=_developer_identity(),
    )
    pipe.valves.GEMINI_FREE_API_KEY = "sanitized-test-key"
    pipe.valves.GEMINI_PAID_API_KEY = None
    return body, metadata, request, binding


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", sorted(CATALOG_MODEL_IDS))
async def test_public_pipe_builds_canonical_request_for_every_developer_model(
    pipe_instance_fixture,
    model_id: str,
) -> None:
    pipe, _constructor = pipe_instance_fixture
    scripted = FakeInteractions([completed_interaction("catalog request")])
    body, metadata, request, binding = _public_pipe_harness(pipe, scripted)
    body["model"] = f"gemini_manifold_google_genai.{model_id}"
    body["stream"] = False
    pipe.valves.GEMINI_PAID_API_KEY = "sanitized-paid-key"

    toggle_patch, client_patch = _public_pipe_patches(pipe, binding)
    with toggle_patch, client_patch:
        result = await pipe.pipe(
            body=body,
            __user__={"id": "user-harness", "email": "user@example.test"},
            __request__=request,
            __event_emitter__=None,
            __metadata__=cast(Metadata, metadata),
        )

    assert isinstance(result, dict)
    assert result["choices"][0]["message"]["content"] == "catalog request"
    assert len(scripted.requests) == 1
    created = scripted.requests[0]
    assert created.model == model_id
    assert created.store is False
    assert created.previous_interaction_id is None
    assert created.tools == []
    image_output = model_id in {"gemini-3-pro-image", "gemini-3.1-flash-image"}
    assert created.response_format is not None
    expected_format = (
        interaction_types.ImageResponseFormat
        if image_output
        else interaction_types.TextResponseFormat
    )
    assert isinstance(created.response_format, expected_format)
    scripted.assert_exhausted()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_failure", "expected_tiers", "expect_success"),
    [
        (
            _client_error(429, "RESOURCE_EXHAUSTED", "free quota exhausted"),
            ["free", "paid"],
            True,
        ),
        (
            _client_error(400, "INVALID_ARGUMENT", "invalid request"),
            ["free"],
            False,
        ),
    ],
    ids=["retryable-advances-to-paid", "nonretryable-stops"],
)
async def test_public_pipe_retries_only_retryable_free_failures(
    pipe_instance_fixture,
    first_failure: Exception,
    expected_tiers: list[str],
    expect_success: bool,
) -> None:
    pipe, _constructor = pipe_instance_fixture
    body, metadata, request, _binding = _public_pipe_harness(pipe, FakeInteractions([]))
    body["stream"] = False
    pipe.valves.GEMINI_PAID_API_KEY = "sanitized-paid-key"
    pipe.valves.ENABLE_FREE_TIER_FALLBACK = True
    success = {
        "choices": [{"message": {"role": "assistant", "content": "paid success"}}],
        "usage": {},
    }
    attempt = AsyncMock(side_effect=[first_failure, success])

    with (
        patch.object(
            pipe,
            "_get_toggleable_feature_status",
            AsyncMock(return_value=(False, False)),
        ),
        patch.object(pipe, "_execute_generation_attempt", new=attempt),
    ):
        result = await pipe.pipe(
            body=body,
            __user__={"id": "user-harness", "email": "user@example.test"},
            __request__=request,
            __event_emitter__=None,
            __metadata__=cast(Metadata, metadata),
        )

    assert isinstance(result, dict)
    assert [call.kwargs["tier"] for call in attempt.await_args_list] == expected_tiers
    content = result["choices"][0]["message"]["content"]
    if expect_success:
        assert content == "paid success"
    else:
        assert content == "Gemini rejected the request configuration."
        assert "PUBLIC_PROVIDER_CANARY" not in content


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["unary", "sse"])
async def test_public_pipe_reaches_strict_developer_interactions_only(
    pipe_instance_fixture, stream: bool
) -> None:
    pipe, _constructor = pipe_instance_fixture
    fake_stream = FakeInteractionStream(completed_stream())
    scripted = FakeInteractions([fake_stream if stream else completed_interaction()])
    body, metadata, request, binding = _public_pipe_harness(pipe, scripted)
    body["stream"] = stream

    def split_system_message(messages):
        if messages and messages[0].get("role") == "system":
            return messages[0], messages[1:]
        return None, messages

    mock_misc_module.pop_system_message.side_effect = split_system_message
    toggles_off = AsyncMock(return_value=(False, False))
    with (
        patch.object(pipe, "_get_toggleable_feature_status", toggles_off),
        patch.object(pipe, "_get_user_client", return_value=binding) as get_client,
    ):
        result = await pipe.pipe(
            body=body,
            __user__={"id": "user-harness", "email": "user@example.test"},
            __request__=request,
            __event_emitter__=None,
            __metadata__=cast(Metadata, metadata),
        )

    if stream:
        assert not isinstance(result, dict)
        chunks = [chunk async for chunk in result]
        assert "Hello from Gemini." in str(chunks)
        assert chunks[-1] == "data: [DONE]"
        assert fake_stream.enter_count == 1
        assert fake_stream.close_count == 1
    else:
        assert isinstance(result, dict)
        assert result["choices"][0]["message"]["content"] == "Hello from Gemini."

    get_client.assert_called_once()
    scripted.assert_exhausted()
    assert len(scripted.requests) == 1
    request_model = scripted.requests[0]
    assert request_model.model == "gemini-2.5-flash"
    assert request_model.stream is stream
    assert request_model.store is False
    assert request_model.previous_interaction_id is None
    assert isinstance(request_model.input, list)
    assert isinstance(request_model.input[0], interaction_types.UserInputStep)
    assert request_model.input[0].type == "user_input"
    assert not hasattr(binding.client, "models")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field_name",
    [
        "automatic_function_calling",
        "automatic_function_calling_config",
        "cached_content",
        "batch",
        "batch_api",
        "batch_config",
        "video_metadata",
    ],
)
async def test_public_pipe_rejects_unsupported_interactions_fields_before_create(
    pipe_instance_fixture,
    field_name: str,
) -> None:
    pipe, _constructor = pipe_instance_fixture
    scripted = FakeInteractions([])
    body, metadata, request, binding = _public_pipe_harness(pipe, scripted)
    cast(dict[str, object], body)[field_name] = {"enabled": True}
    toggle_patch, client_patch = _public_pipe_patches(pipe, binding)

    with toggle_patch, client_patch:
        result = await pipe.pipe(
            body=body,
            __user__={"id": "user-harness", "email": "user@example.test"},
            __request__=request,
            __event_emitter__=None,
            __metadata__=cast(Metadata, metadata),
        )

    assert isinstance(result, dict)
    assert result["choices"][0]["message"]["content"] == "Gemini request failed unexpectedly."
    assert scripted.raw_requests == []


@pytest.mark.asyncio
async def test_public_pipe_persists_grounded_search_and_companion_renders_it(
    pipe_instance_fixture,
) -> None:
    pipe, _constructor = pipe_instance_fixture
    annotation = interaction_types.URLCitation(
        type="url_citation",
        url="https://example.test/source",
        title="Example source",
        start_index=0,
        end_index=6,
    )
    interaction = interaction_types.Interaction(
        id="grounded-interaction",
        model="gemini-2.5-flash",
        status="completed",
        steps=[
            interaction_types.GoogleSearchCallStep(
                type="google_search_call",
                id="search-call",
                arguments=interaction_types.GoogleSearchCallArguments(
                    queries=["sanitized public query"]
                ),
                search_type="web_search",
                signature="c2ln",
            ),
            interaction_types.GoogleSearchResultStep(
                type="google_search_result",
                call_id="search-call",
                result=[
                    interaction_types.GoogleSearchResult(
                        search_suggestions="sanitized refined query"
                    )
                ],
                signature="c2ln",
            ),
            interaction_types.ModelOutputStep(
                type="model_output",
                content=[
                    interaction_types.TextContent(
                        type="text", text="Source", annotations=[annotation]
                    )
                ],
            ),
        ],
        usage=interaction_types.Usage(total_input_tokens=3, total_output_tokens=2, total_tokens=5),
    )
    scripted = FakeInteractions([interaction])
    body, metadata, request, binding = _public_pipe_harness(pipe, scripted)
    body["stream"] = False
    pipe.valves.GEMINI_PAID_API_KEY = "sanitized-paid-key"
    metadata.update(
        {
            "chat_id": "chat-grounded",
            "message_id": "assistant-grounded",
            "features": {
                "gemini_manifold_companion_version": "3.0.0",
                "google_search_tool": True,
            },
        }
    )
    upsert = _install_public_chat_history(
        [{"id": "user-grounded", "parentId": None, "role": "user", "content": "Hello"}],
        current_message_id="assistant-grounded",
    )

    toggle_patch, client_patch = _public_pipe_patches(pipe, binding)
    with toggle_patch, client_patch:
        result = await pipe.pipe(
            body=body,
            __user__={"id": "user-harness", "email": "user@example.test"},
            __request__=request,
            __event_emitter__=None,
            __metadata__=cast(Metadata, metadata),
        )

    assert isinstance(result, dict)
    assert result["choices"][0]["message"]["content"] == "Source"
    assert len(scripted.requests) == 1
    assert [tool.type for tool in scripted.requests[0].tools or []] == ["google_search"]
    upsert.assert_awaited_once()
    awaited_upsert = upsert.await_args
    assert awaited_upsert is not None
    persisted = awaited_upsert.kwargs["message"]["gemini_interaction"]
    envelope = InteractionEnvelopeV1.model_validate(persisted)
    assert envelope.grounding.queries == ["sanitized public query"]
    assert [source.uri for source in envelope.grounding.sources] == ["https://example.test/source"]
    assert len(envelope.grounding.citations) == 1

    companion_body = cast(
        Body,
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Source",
                    "gemini_interaction": persisted,
                }
            ]
        },
    )
    events: list[Event] = []

    async def collect(event: Event) -> None:
        events.append(event)

    companion = CompanionFilter()
    companion_result = await companion.outlet(
        companion_body,
        request,
        {},
        collect,
    )

    assert companion_result["messages"][-1]["content"] == "Source[1]"
    assert any(event["type"] == "source" for event in events)
    assert any(
        event["type"] == "status" and event["data"].get("action") == "web_search_queries_generated"
        for event in events
    )
    scripted.assert_exhausted()


@pytest.mark.asyncio
async def test_public_pipe_multimodal_storage_input_and_generated_image_output(
    pipe_instance_fixture, tmp_path: Path
) -> None:
    mock_files_module.reset_mock()
    mock_storage_module.reset_mock()
    pipe, _constructor = pipe_instance_fixture
    input_image = b"sanitized-input-image"
    generated_image = b"sanitized-generated-image"
    local_image = tmp_path / "input.png"
    local_image.write_bytes(input_image)
    interaction = interaction_types.Interaction(
        id="media-interaction",
        status="completed",
        steps=[
            interaction_types.ModelOutputStep(
                type="model_output",
                content=[
                    interaction_types.TextContent(type="text", text="Created "),
                    interaction_types.ImageContent(
                        type="image",
                        mime_type="image/png",
                        data=base64.b64encode(generated_image).decode("ascii"),
                    ),
                ],
            )
        ],
    )
    scripted = FakeInteractions([interaction])
    body, metadata, request, binding = _public_pipe_harness(pipe, scripted)
    body["model"] = "gemini_manifold_google_genai.gemini-3.1-flash-image"
    body["stream"] = False
    pipe.valves.GEMINI_PAID_API_KEY = "sanitized-paid-key"
    pipe.valves.USE_FILES_API = False
    metadata.update(
        {
            "chat_id": "chat-media",
            "message_id": "assistant-media",
            "features": {
                "gemini_manifold_companion_version": "3.0.0",
                "upload_documents": True,
            },
        }
    )
    _install_public_chat_history(
        [
            {
                "id": "user-media",
                "parentId": None,
                "role": "user",
                "content": "Describe and transform this image",
                "files": [{"id": "input-file", "name": "input.png", "content_type": "image/png"}],
            }
        ],
        current_message_id="assistant-media",
    )
    file_model = MagicMock(path="s3://sanitized/input.png", meta={"content_type": "image/png"})
    mock_files_module.Files.get_file_by_id_and_user_id = AsyncMock(return_value=file_model)
    mock_storage_module.Storage.get_file.reset_mock()
    mock_storage_module.Storage.get_file.return_value = str(local_image)
    mock_storage_module.Storage.upload_file.return_value = (
        generated_image,
        "s3://sanitized/generated.png",
    )
    mock_files_module.Files.insert_new_file = AsyncMock(return_value=MagicMock(id="generated-file"))
    request.app.url_path_for = MagicMock(return_value="/api/v1/files/generated-file/content")

    toggle_patch, client_patch = _public_pipe_patches(pipe, binding)
    with toggle_patch, client_patch:
        result = await pipe.pipe(
            body=body,
            __user__={"id": "user-harness", "email": "user@example.test"},
            __request__=request,
            __event_emitter__=None,
            __metadata__=cast(Metadata, metadata),
        )

    assert isinstance(result, dict)
    assert result["choices"][0]["message"]["content"] == (
        "Created ![Generated Image](/api/v1/files/generated-file/content)"
    )
    assert len(scripted.requests) == 1
    request_input = scripted.requests[0].input
    assert isinstance(request_input, list)
    assert isinstance(request_input[0], interaction_types.UserInputStep)
    image_parts = [
        part
        for part in request_input[0].content or []
        if isinstance(part, interaction_types.ImageContent)
    ]
    assert len(image_parts) == 1
    assert base64.b64decode(image_parts[0].data or "") == input_image
    assert image_parts[0].uri is None
    mock_files_module.Files.get_file_by_id_and_user_id.assert_awaited_once_with(
        id="input-file", user_id="user-harness"
    )
    mock_storage_module.Storage.get_file.assert_called_once_with("s3://sanitized/input.png")
    mock_storage_module.Storage.upload_file.assert_called_once()
    mock_files_module.Files.insert_new_file.assert_awaited_once()
    assert not hasattr(binding.client, "models")
    scripted.assert_exhausted()


@pytest.mark.asyncio
async def test_public_pipe_rejects_foreign_endpoint_uri_before_create(
    pipe_instance_fixture,
) -> None:
    pipe, _constructor = pipe_instance_fixture
    scripted = FakeInteractions([])
    body, metadata, request, binding = _public_pipe_harness(pipe, scripted)
    body["stream"] = False
    body["messages"] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "parent answer"},
        {"role": "user", "content": "next"},
    ]
    metadata.update({"chat_id": "chat-foreign", "message_id": "assistant-foreign"})
    _install_public_chat_history(
        [
            {"id": "user-first", "parentId": None, "role": "user", "content": "first"},
            {
                "id": "assistant-parent",
                "parentId": "user-first",
                "role": "assistant",
                "content": "parent answer",
                "gemini_interaction": _interaction_envelope(
                    endpoint_scope="foreign-endpoint-scope",
                    model_id="gemini-2.5-flash",
                    steps=[
                        {
                            "type": "model_output",
                            "content": [
                                {"type": "text", "text": "parent answer"},
                                {
                                    "type": "image",
                                    "uri": "https://files.example.test/foreign-image",
                                    "mime_type": "image/png",
                                },
                            ],
                        }
                    ],
                ),
            },
            {
                "id": "user-next",
                "parentId": "assistant-parent",
                "role": "user",
                "content": "next",
            },
        ],
        current_message_id="assistant-foreign",
    )

    toggle_patch, client_patch = _public_pipe_patches(pipe, binding)
    with toggle_patch, client_patch:
        result = await pipe.pipe(
            body=body,
            __user__={"id": "user-harness", "email": "user@example.test"},
            __request__=request,
            __event_emitter__=None,
            __metadata__=cast(Metadata, metadata),
        )

    assert isinstance(result, dict)
    assert result["choices"][0]["message"]["content"] == "Gemini request failed unexpectedly."
    assert scripted.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_path", ["user_attachment", "assistant_history"])
async def test_public_pipe_denies_transplanted_local_file_before_storage_or_provider(
    pipe_instance_fixture,
    replay_path: str,
) -> None:
    pipe, _constructor = pipe_instance_fixture
    scripted = FakeInteractions([])
    body, metadata, request, binding = _public_pipe_harness(pipe, scripted)
    body["stream"] = False
    file_canary = "TRANSPLANTED_PUBLIC_FILE_CANARY"
    user_canary = "REQUESTING_PUBLIC_USER_CANARY"
    metadata.update(
        {
            "chat_id": "chat-owned-file",
            "message_id": "assistant-local-file-owner",
            "features": {
                "gemini_manifold_companion_version": "3.0.0",
                "upload_documents": True,
            },
        }
    )
    if replay_path == "user_attachment":
        body["messages"] = [{"role": "user", "content": "inspect"}]
        history = [
            {
                "id": "user-file",
                "parentId": None,
                "role": "user",
                "content": "inspect",
                "files": [
                    {
                        "id": file_canary,
                        "name": "attachment.png",
                        "content_type": "image/png",
                    }
                ],
            }
        ]
    else:
        body["messages"] = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "parent answer"},
            {"role": "user", "content": "next"},
        ]
        history = [
            {"id": "user-first", "parentId": None, "role": "user", "content": "first"},
            {
                "id": "assistant-parent",
                "parentId": "user-first",
                "role": "assistant",
                "content": "parent answer",
                "gemini_interaction": _interaction_envelope(
                    endpoint_scope=binding.identity.for_model("gemini-2.5-flash").scope,
                    model_id="gemini-2.5-flash",
                    steps=[
                        {
                            "type": "model_output",
                            "content": [
                                {"type": "text", "text": "parent answer"},
                                {
                                    "type": "image",
                                    "uri": f"/api/v1/files/{file_canary}/content",
                                    "mime_type": "image/png",
                                },
                            ],
                        }
                    ],
                ),
            },
            {
                "id": "user-next",
                "parentId": "assistant-parent",
                "role": "user",
                "content": "next",
            },
        ]
    _install_public_chat_history(history, current_message_id="assistant-local-file-owner")
    mock_files_module.Files.get_file_by_id_and_user_id = AsyncMock(return_value=None)
    mock_storage_module.Storage.get_file.reset_mock()
    captured: list[str] = []
    handler_id = gemini_manifold_module.log.add(
        lambda message: captured.append(str(message)), format="{message}"
    )
    toggle_patch, client_patch = _public_pipe_patches(pipe, binding)
    try:
        with toggle_patch, client_patch:
            result = await pipe.pipe(
                body=body,
                __user__={"id": user_canary, "email": "user@example.test"},
                __request__=request,
                __event_emitter__=None,
                __metadata__=cast(Metadata, metadata),
            )
    finally:
        gemini_manifold_module.log.remove(handler_id)

    assert isinstance(result, dict)
    assert result["choices"][0]["message"]["content"] == "Gemini request failed unexpectedly."
    assert scripted.requests == []
    mock_files_module.Files.get_file_by_id_and_user_id.assert_awaited_once_with(
        id=file_canary, user_id=user_canary
    )
    mock_storage_module.Storage.get_file.assert_not_called()
    emitted = "\n".join(captured)
    assert file_canary not in emitted
    assert user_canary not in emitted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_type", "mime_type"),
    [
        ("image", "image/png"),
        ("audio", "audio/wav"),
        ("video", "video/mp4"),
        ("document", "application/pdf"),
    ],
)
@pytest.mark.parametrize(
    ("identity_case", "prior_model_id"),
    [
        ("cross_model", "gemini-3.1-flash-image"),
        ("cross_service", "gemini-2.5-flash"),
    ],
)
async def test_public_pipe_rejects_incompatible_cross_identity_output_before_create(
    pipe_instance_fixture,
    content_type: str,
    mime_type: str,
    identity_case: str,
    prior_model_id: str,
) -> None:
    pipe, _constructor = pipe_instance_fixture
    scripted = FakeInteractions([])
    body, metadata, request, binding = _public_pipe_harness(pipe, scripted)
    body["stream"] = False
    body["messages"] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "parent answer"},
        {"role": "user", "content": "next"},
    ]
    metadata.update({"chat_id": "chat-modality", "message_id": "assistant-modality"})
    envelope = cast(dict, request.app.state._state["gemini_model_catalog"])
    claim = envelope["payload"]["provider_claims"]["gemini-2.5-flash"]
    claim["content"]["inputs"] = ["text"]
    claim["pricing"]["input"] = {"text": claim["pricing"]["input"]["text"]}
    claim["pricing"]["cached_input"] = {"text": claim["pricing"]["cached_input"]["text"]}
    validated_payload = AppStateModelCatalog.model_validate(envelope["payload"])
    envelope["canonical_digest"] = (
        "sha256:" + hashlib.sha256(pipe_canonical_catalog_bytes(validated_payload)).hexdigest()
    )
    _install_public_chat_history(
        [
            {"id": "user-first", "parentId": None, "role": "user", "content": "first"},
            {
                "id": "assistant-parent",
                "parentId": "user-first",
                "role": "assistant",
                "content": "parent answer",
                "gemini_interaction": _interaction_envelope(
                    endpoint_scope=f"prior-{identity_case}-scope",
                    model_id=prior_model_id,
                    steps=[
                        {
                            "type": "model_output",
                            "content": [
                                {"type": "text", "text": "parent answer"},
                                {
                                    "type": content_type,
                                    "data": base64.b64encode(b"prior output").decode("ascii"),
                                    "mime_type": mime_type,
                                },
                            ],
                        }
                    ],
                ),
            },
            {
                "id": "user-next",
                "parentId": "assistant-parent",
                "role": "user",
                "content": "next",
            },
        ],
        current_message_id="assistant-modality",
    )

    toggle_patch, client_patch = _public_pipe_patches(pipe, binding)
    with toggle_patch, client_patch:
        result = await pipe.pipe(
            body=body,
            __user__={"id": "user-harness", "email": "user@example.test"},
            __request__=request,
            __event_emitter__=None,
            __metadata__=cast(Metadata, metadata),
        )

    assert isinstance(result, dict)
    assert result["choices"][0]["message"]["content"] == "Gemini request failed unexpectedly."
    assert scripted.requests == []


def _install_public_chat_history(
    messages: list[dict[str, object]], *, current_message_id: str
) -> AsyncMock:
    placeholder = {
        "id": current_message_id,
        "parentId": messages[-1]["id"],
        "role": "assistant",
        "content": "",
    }
    chat = MagicMock()
    chat.chat = {
        "history": {
            "messages": {str(message["id"]): message for message in [*messages, placeholder]},
            "currentId": current_message_id,
        }
    }
    mock_chats_module.Chats.get_chat_by_id_and_user_id = AsyncMock(return_value=chat)
    upsert = AsyncMock()
    mock_chats_module.Chats.upsert_message_to_chat_by_id_and_message_id = upsert
    return upsert


def _public_pipe_patches(pipe: Pipe, binding: GenAIClientBinding):
    def split_system_message(messages):
        if messages and messages[0].get("role") == "system":
            return messages[0], messages[1:]
        return None, messages

    mock_misc_module.pop_system_message.side_effect = split_system_message
    return (
        patch.object(
            pipe, "_get_toggleable_feature_status", AsyncMock(return_value=(False, False))
        ),
        patch.object(pipe, "_get_user_client", return_value=binding),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_state", ["expired", "deleted"])
async def test_public_pipe_same_scope_continuation_404_replays_and_persists_once(
    pipe_instance_fixture,
    missing_state: str,
) -> None:
    pipe, _constructor = pipe_instance_fixture
    scripted = FakeInteractions(
        [
            _client_error(404, "NOT_FOUND", f"Interaction is {missing_state}"),
            completed_interaction(interaction_id="replayed-id"),
        ]
    )
    body, metadata, request, binding = _public_pipe_harness(pipe, scripted)
    model_id = "gemini-2.5-flash"
    endpoint_scope = binding.identity.for_model(model_id).scope
    body["stream"] = False
    body["messages"] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "parent answer"},
        {"role": "user", "content": "next"},
    ]
    metadata.update({"chat_id": "chat-continuation", "message_id": "assistant-current"})
    upsert = _install_public_chat_history(
        [
            {"id": "user-first", "parentId": None, "role": "user", "content": "first"},
            {
                "id": "assistant-parent",
                "parentId": "user-first",
                "role": "assistant",
                "content": "parent answer",
                "usage": {"cumulative_token_count": 7, "cumulative_total_cost": 0.01},
                "gemini_interaction": _interaction_envelope(
                    interaction_id="parent-interaction",
                    endpoint_scope=endpoint_scope,
                    model_id=model_id,
                ),
            },
            {
                "id": "user-next",
                "parentId": "assistant-parent",
                "role": "user",
                "content": "next",
            },
        ],
        current_message_id="assistant-current",
    )

    toggle_patch, client_patch = _public_pipe_patches(pipe, binding)
    with toggle_patch, client_patch:
        result = await pipe.pipe(
            body=body,
            __user__={"id": "user-harness", "email": "user@example.test"},
            __request__=request,
            __event_emitter__=None,
            __metadata__=cast(Metadata, metadata),
        )

    assert isinstance(result, dict)
    assert result["choices"][0]["message"]["content"] == "Hello from Gemini."
    assert result["usage"]["input_tokens"] == 3
    assert result["usage"]["cumulative_token_count"] == 14
    assert len(scripted.requests) == 2
    continued, replayed = scripted.requests
    assert continued.previous_interaction_id == "parent-interaction"
    assert isinstance(continued.input, list) and len(continued.input) == 1
    assert replayed.previous_interaction_id is None
    assert isinstance(replayed.input, list) and len(replayed.input) == 3
    assert continued.store is True and replayed.store is True
    assert continued.model == replayed.model
    assert continued.tools == replayed.tools
    assert continued.system_instruction == replayed.system_instruction
    assert continued.generation_config == replayed.generation_config
    assert continued.response_format == replayed.response_format
    assert all(raw["store"] is True for raw in scripted.raw_requests)
    assert all("background" not in raw for raw in scripted.raw_requests)
    upsert.assert_awaited_once()
    awaited_upsert = upsert.await_args
    assert awaited_upsert is not None
    persisted = awaited_upsert.kwargs["message"]["gemini_interaction"]
    envelope = InteractionEnvelopeV1.model_validate(persisted)
    assert envelope.interaction_id == "replayed-id"
    assert envelope.usage.total_tokens == 7
    scripted.assert_exhausted()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_streaming", [False, True])
async def test_public_pipe_custom_function_round_trip_uses_result_continuation_order(
    pipe_instance_fixture,
    is_streaming,
) -> None:
    pipe, _constructor = pipe_instance_fixture
    scripted = FakeInteractions(
        [
            _tool_stream("tool-round-1", _function_call("call-1")),
            _final_stream("tool finished"),
        ]
        if is_streaming
        else [
            _tool_interaction("tool-round-1", _function_call("call-1")),
            completed_interaction("tool finished", interaction_id="tool-final"),
        ]
    )
    body, metadata, request, binding = _public_pipe_harness(pipe, scripted)
    body["model"] = "gemini_manifold_google_genai.gemini-3.5-flash"
    body["stream"] = is_streaming
    metadata.update({"chat_id": "chat-tool", "message_id": "assistant-tool"})
    _install_public_chat_history(
        [{"id": "user-tool", "parentId": None, "role": "user", "content": "Hello"}],
        current_message_id="assistant-tool",
    )
    function = AsyncMock(return_value={"value": "sanitized"})

    toggle_patch, client_patch = _public_pipe_patches(pipe, binding)
    with (
        toggle_patch,
        client_patch,
        patch.object(
            pipe,
            "_run_custom_function_loop",
            wraps=pipe._run_custom_function_loop,
        ) as explicit_loop,
    ):
        result = await pipe.pipe(
            body=body,
            __user__={"id": "user-harness", "email": "user@example.test"},
            __request__=request,
            __event_emitter__=None,
            __metadata__=cast(Metadata, metadata),
            __tools__={"lookup": _authorized_tool(function)},
        )

    if is_streaming:
        assert not isinstance(result, dict)
        chunks = [chunk async for chunk in result]
        assert _stream_text(chunks) == "tool finished"
    else:
        assert isinstance(result, dict)
        assert result["choices"][0]["message"]["content"] == "tool finished"
    function.assert_awaited_once_with(query="value")
    explicit_loop.assert_awaited_once()
    assert len(scripted.requests) == 2
    _assert_explicit_function_requests(scripted.raw_requests)
    first, second = scripted.requests
    assert first.previous_interaction_id is None
    assert [tool.type for tool in first.tools or []] == ["function"]
    assert second.previous_interaction_id == "tool-round-1"
    assert isinstance(second.input, list)
    assert len(second.input) == 1
    assert isinstance(second.input[0], interaction_types.FunctionResultStep)
    assert second.input[0].call_id == "call-1"
    assert second.input[0].result == '{"value":"sanitized"}'
    scripted.assert_exhausted()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_streaming", [False, True])
@pytest.mark.parametrize("stateless_mode", ["temporary", "task", "privacy_opt_out"])
async def test_public_pipe_custom_function_round_trip_uses_full_stateless_replay(
    pipe_instance_fixture,
    is_streaming,
    stateless_mode,
) -> None:
    pipe, _constructor = pipe_instance_fixture
    scripted = FakeInteractions(
        [
            _tool_stream("tool-round-1", _function_call("call-1")),
            _final_stream("stateless tool finished"),
        ]
        if is_streaming
        else [
            _tool_interaction("tool-round-1", _function_call("call-1")),
            completed_interaction("stateless tool finished", interaction_id="tool-final"),
        ]
    )
    body, metadata, request, binding = _public_pipe_harness(pipe, scripted)
    body["model"] = "gemini_manifold_google_genai.gemini-3.5-flash"
    body["stream"] = is_streaming
    cast(dict[str, object], body)["background"] = True
    metadata.update({"chat_id": "chat-stateless-tool", "message_id": "assistant-tool"})
    if stateless_mode == "temporary":
        metadata["chat_id"] = "local:stateless-tool"
    elif stateless_mode == "task":
        metadata["task"] = "title_generation"
    _install_public_chat_history(
        [{"id": "user-tool", "parentId": None, "role": "user", "content": "Hello"}],
        current_message_id="assistant-tool",
    )
    function = AsyncMock(return_value={"value": "private"})
    user: dict[str, object] = {"id": "user-harness", "email": "user@example.test"}
    if stateless_mode == "privacy_opt_out":
        user["valves"] = Pipe.UserValves(STORE_INTERACTIONS=False)

    toggle_patch, client_patch = _public_pipe_patches(pipe, binding)
    with toggle_patch, client_patch:
        result = await pipe.pipe(
            body=body,
            __user__=cast(UserData, user),
            __request__=request,
            __event_emitter__=None,
            __metadata__=cast(Metadata, metadata),
            __tools__={"lookup": _authorized_tool(function)},
        )

    if is_streaming:
        assert not isinstance(result, dict)
        chunks = [chunk async for chunk in result]
        assert _stream_text(chunks) == "stateless tool finished"
    else:
        assert isinstance(result, dict)
        assert result["choices"][0]["message"]["content"] == "stateless tool finished"
    function.assert_awaited_once_with(query="value")
    assert len(scripted.requests) == 2
    _assert_explicit_function_requests(scripted.raw_requests)
    first, second = scripted.requests
    assert first.store is False and second.store is False
    assert first.previous_interaction_id is None and second.previous_interaction_id is None
    assert all(raw["store"] is False for raw in scripted.raw_requests)
    assert all("previous_interaction_id" not in raw for raw in scripted.raw_requests)
    assert all("background" not in raw for raw in scripted.raw_requests)
    assert [step.type for step in cast(list, first.input)] == ["user_input"]
    assert [step.type for step in cast(list, second.input)] == [
        "user_input",
        "function_call",
        "function_result",
    ]
    assert first.tools == second.tools
    assert first.system_instruction == second.system_instruction
    assert first.generation_config == second.generation_config
    assert first.response_format == second.response_format
    scripted.assert_exhausted()


@pytest.mark.asyncio
async def test_public_sse_persists_reasoning_signature_and_closes_stream(
    pipe_instance_fixture,
) -> None:
    pipe, _constructor = pipe_instance_fixture
    signature = base64.b64encode(b"opaque-reasoning-signature").decode("ascii")
    fake_stream = FakeInteractionStream(completed_reasoning_stream(signature=signature))
    scripted = FakeInteractions([fake_stream])
    body, metadata, request, binding = _public_pipe_harness(pipe, scripted)
    body["stream"] = True
    metadata.update({"chat_id": "chat-reasoning", "message_id": "assistant-reasoning"})
    upsert = _install_public_chat_history(
        [
            {
                "id": "user-reasoning",
                "parentId": None,
                "role": "user",
                "content": "Hello",
            }
        ],
        current_message_id="assistant-reasoning",
    )

    toggle_patch, client_patch = _public_pipe_patches(pipe, binding)
    with toggle_patch, client_patch:
        result = await pipe.pipe(
            body=body,
            __user__={"id": "user-harness", "email": "user@example.test"},
            __request__=request,
            __event_emitter__=None,
            __metadata__=cast(Metadata, metadata),
        )
    assert not isinstance(result, dict)
    chunks = [chunk async for chunk in result]

    assert "brief rationale" in str(chunks)
    assert "reasoned answer" in str(chunks)
    assert fake_stream.close_count == 1
    upsert.assert_awaited_once()
    awaited_upsert = upsert.await_args
    assert awaited_upsert is not None
    envelope = InteractionEnvelopeV1.model_validate(
        awaited_upsert.kwargs["message"]["gemini_interaction"]
    )
    thought_steps = [step for step in envelope.steps if step.get("type") == "thought"]
    assert thought_steps == [
        {
            "type": "thought",
            "signature": signature,
            "summary": [{"text": "brief rationale", "type": "text"}],
        }
    ]


@pytest.mark.asyncio
async def test_public_sse_early_close_releases_strict_stream(pipe_instance_fixture) -> None:
    pipe, _constructor = pipe_instance_fixture
    signature = base64.b64encode(b"opaque-early-close").decode("ascii")
    fake_stream = FakeInteractionStream(completed_reasoning_stream(signature=signature))
    scripted = FakeInteractions([fake_stream])
    body, metadata, request, binding = _public_pipe_harness(pipe, scripted)
    body["stream"] = True

    toggle_patch, client_patch = _public_pipe_patches(pipe, binding)
    with toggle_patch, client_patch:
        result = await pipe.pipe(
            body=body,
            __user__={"id": "user-harness", "email": "user@example.test"},
            __request__=request,
            __event_emitter__=None,
            __metadata__=cast(Metadata, metadata),
        )
    assert not isinstance(result, dict)
    first_chunk = await anext(result)
    assert "brief rationale" in str(first_chunk)
    await result.aclose()

    assert fake_stream.close_count == 1
    assert fake_stream.closed
    scripted.assert_exhausted()


@pytest.mark.asyncio
async def test_public_pipe_denies_unverified_enterprise_before_create(
    pipe_instance_fixture,
) -> None:
    pipe, _constructor = pipe_instance_fixture
    scripted = FakeInteractions([])
    body, metadata, request, binding = _public_pipe_harness(pipe, scripted)
    pipe.valves.GEMINI_FREE_API_KEY = None
    pipe.valves.ENTERPRISE_PROJECT = "sanitized-project"
    enterprise_identity = EndpointIdentity(
        service="enterprise",
        credential_fingerprint="sanitized-credential",
        project="sanitized-project",
        location="global",
        api_version="v1beta1",
    )
    enterprise_binding = GenAIClientBinding(
        client=binding.client,
        identity=enterprise_identity,
    )

    async def toggle_status(filter_id: str, _metadata: Metadata) -> tuple[bool, bool]:
        return (True, True) if filter_id == "gemini_enterprise_toggle" else (False, False)

    with (
        patch.object(pipe, "_get_toggleable_feature_status", side_effect=toggle_status),
        patch.object(pipe, "_get_user_client", return_value=enterprise_binding) as get_client,
        pytest.raises(ValueError, match="unverified.*enterprise"),
    ):
        await pipe.pipe(
            body=body,
            __user__={"id": "user-harness", "email": "user@example.test"},
            __request__=request,
            __event_emitter__=None,
            __metadata__=cast(Metadata, metadata),
        )

    get_client.assert_not_called()
    assert scripted.requests == []


def teardown_module(module):
    """Cleans up sys.modules after tests in this file are done."""
    del sys.modules["open_webui.models.chats"]
    del sys.modules["open_webui.models.files"]
    del sys.modules["open_webui.models.functions"]
    del sys.modules["open_webui.storage.provider"]
    del sys.modules["open_webui.utils.misc"]
