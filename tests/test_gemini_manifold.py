from typing import cast
from aiocache.base import BaseCache
from pathlib import Path
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock, call, ANY
import sys
import asyncio
import io
import yaml

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


from plugins.pipes.gemini_manifold import (
    Pipe,
    GeminiContentBuilder,
    GeminiPDFProcessor,
    PDFMitigationManager,
    PDFMitigationOutcome,
    PreparedPDFPart,
    PreparedPDFResult,
    LocalFileSource,
    PDFProcessingError,
    ContentBuildError,
    types as gemini_types,
)  # gemini_types is google.genai.types
from plugins.filters.gemini_manifold_companion import EventEmitter

# region Test Constants
# General Users
USER_EMAIL_REGULAR = "regular_user@example.com"
USER_EMAIL_UNPRIVILEGED = "unprivileged_user@example.com"
USER_EMAIL_WHITELISTED = "whitelisted_user@example.com"

# Admin Credentials
ADMIN_FREE_KEY = "admin_default_free_key"
ADMIN_PAID_KEY = "admin_default_paid_key"
ADMIN_GEMINI_BASE_URL = "https://admin.default.gemini.api.com"
ADMIN_VERTEX_PROJECT = "admin_default_vertex_project"
ADMIN_VERTEX_LOCATION = "admin_default_vertex_location"

# User Credentials
USER_FREE_KEY = "user_specific_free_key"
USER_PAID_KEY = "user_specific_paid_key"
USER_GEMINI_BASE_URL = "https://user.specific.gemini.api.com"
USER_VERTEX_PROJECT = "user_specific_vertex_project"
USER_VERTEX_LOCATION = "user_specific_vertex_location"
# endregion Test Constants


def test_gemini_models_yaml_omits_promoted_flash_image_preview():
    config_path = Path(__file__).resolve().parents[1] / "plugins/pipes/gemini_models.yaml"
    model_config = yaml.safe_load(config_path.read_text())

    assert "gemini-3.1-flash-image" in model_config
    assert "gemini-3.1-flash-image-preview" not in model_config


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
        "USE_VERTEX_AI": False,
        "VERTEX_PROJECT": None,
        "VERTEX_LOCATION": "global",
        "MODEL_WHITELIST": "*",
        "MODEL_BLACKLIST": None,
        "CACHE_MODELS": True,
        "THINKING_BUDGET": 8192,
        "SHOW_THINKING_SUMMARY": True,
        "USE_FILES_API": True,
        "PDF_LIMIT_MITIGATION": True,
        "THINKING_MODEL_PATTERN": r"gemini-2.5",
        "LOG_LEVEL": "INFO",
        "ENABLE_URL_CONTEXT_TOOL": False,
    }


@pytest_asyncio.fixture
async def pipe_instance_fixture(mock_pipe_valves_data):
    """
    Helper fixture to setup a Pipe instance with mocked genai.Client constructor
    and yields both the pipe instance and the mock constructor.
    """
    mock_gemini_client_actual_instance = MagicMock()

    with patch(
        "plugins.pipes.gemini_manifold.genai.Client",
        return_value=mock_gemini_client_actual_instance,
    ) as MockedGenAIClientConstructor, patch.object(
        Pipe, "_add_log_handler", MagicMock()
    ), patch(
        "sys.stdout", MagicMock()
    ):
        pipe = Pipe()
        # Initialize with base data from mock_pipe_valves_data
        pipe.valves = Pipe.Valves(**mock_pipe_valves_data)
        # Yield both the pipe instance and the mock for genai.Client constructor
        yield pipe, MockedGenAIClientConstructor

    # Teardown: Clear caches to ensure clean state for subsequent tests
    Pipe._get_or_create_genai_client.cache_clear()
    if hasattr(pipe._get_genai_models, "cache"):
        cache_instance = getattr(pipe._get_genai_models, "cache")
        if cache_instance:
            await cast(BaseCache, cache_instance).clear()


# endregion Fixtures


# region Test _get_or_create_genai_client
def test_pipe_initialization_with_api_key_prefers_free(mock_pipe_valves_data):
    """
    Tests that when both free and paid API keys are available, the free key is preferred by default.
    """
    mock_gemini_client_instance = MagicMock()

    with patch(
        "plugins.pipes.gemini_manifold.genai.Client",
        return_value=mock_gemini_client_instance,
    ) as MockedGenAIClientConstructor, patch.object(
        Pipe, "_add_log_handler", MagicMock()
    ), patch(
        "sys.stdout", MagicMock()
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
                http_options=gemini_types.HttpOptions(base_url=ADMIN_GEMINI_BASE_URL),
            )
        finally:
            Pipe._get_or_create_genai_client.cache_clear()


def test_get_user_client_no_auth_provided_raises_error(mock_pipe_valves_data):
    """
    Tests that genai.Client is NOT called and an error is raised when no API keys
    or Vertex project are provided.
    """
    # Configure valves to have no API keys and no Vertex AI project
    mock_pipe_valves_data["GEMINI_FREE_API_KEY"] = None
    mock_pipe_valves_data["GEMINI_PAID_API_KEY"] = None
    mock_pipe_valves_data["USE_VERTEX_AI"] = False
    mock_pipe_valves_data["VERTEX_PROJECT"] = None

    with patch(
        "plugins.pipes.gemini_manifold.genai.Client"
    ) as MockedGenAIClientConstructor, patch.object(
        Pipe, "_add_log_handler", MagicMock()
    ), patch(
        "sys.stdout", MagicMock()
    ):
        pipe_instance = Pipe()
        pipe_instance.valves = Pipe.Valves(**mock_pipe_valves_data)

        with pytest.raises(
            ValueError, match="Neither VERTEX_PROJECT nor a Gemini API key"
        ):
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
    pipe.valves.USE_VERTEX_AI = False

    pipe._get_user_client(pipe.valves, USER_EMAIL_REGULAR)

    MockedGenAIClientConstructor.assert_called_once_with(
        api_key=expected_key,
        http_options=gemini_types.HttpOptions(base_url=ADMIN_GEMINI_BASE_URL),
    )
    Pipe._get_or_create_genai_client.cache_clear()


def test_client_creation_uses_vertex_ai_when_configured(pipe_instance_fixture):
    """
    Tests that Vertex AI client is created when USE_VERTEX_AI is True and VERTEX_PROJECT is provided.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USE_VERTEX_AI = True
    pipe.valves.VERTEX_PROJECT = "test_vertex_project_id"
    pipe.valves.VERTEX_LOCATION = "europe-west4"

    pipe._get_user_client(pipe.valves, USER_EMAIL_REGULAR)

    MockedGenAIClientConstructor.assert_called_once_with(
        vertexai=True, project="test_vertex_project_id", location="europe-west4"
    )
    Pipe._get_or_create_genai_client.cache_clear()


def test_client_creation_falls_back_to_gemini_api_with_warning(pipe_instance_fixture):
    """
    Tests fallback to Gemini Developer API with a warning when USE_VERTEX_AI is True,
    but VERTEX_PROJECT is not provided.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USE_VERTEX_AI = True
    pipe.valves.VERTEX_PROJECT = None
    pipe.valves.GEMINI_FREE_API_KEY = "fallback_free_key"
    pipe.valves.GEMINI_PAID_API_KEY = "fallback_paid_key"  # Should not be used

    with patch("plugins.pipes.gemini_manifold.log.warning") as mock_log_warning:
        pipe._get_user_client(pipe.valves, USER_EMAIL_REGULAR)
        mock_log_warning.assert_called_once_with(
            "Vertex AI is enabled but no project is set. Using Gemini Developer API."
        )

    MockedGenAIClientConstructor.assert_called_once_with(
        api_key="fallback_free_key",  # Should fall back to the free key
        http_options=gemini_types.HttpOptions(base_url=ADMIN_GEMINI_BASE_URL),
    )
    Pipe._get_or_create_genai_client.cache_clear()


# endregion Test _get_or_create_genai_client


# region Test USER_MUST_PROVIDE_AUTH_CONFIG=True scenarios
def test_user_must_auth_no_user_key_provided_errors(pipe_instance_fixture):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=True. User provides no keys. Expected: ValueError.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = None

    user_valves_instance = Pipe.UserValves(
        GEMINI_FREE_API_KEY=None, GEMINI_PAID_API_KEY=None
    )
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_UNPRIVILEGED
    )

    with pytest.raises(
        ValueError, match="Please set GEMINI_FREE_API_KEY or GEMINI_PAID_API_KEY"
    ):
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
        http_options=gemini_types.HttpOptions(base_url=USER_GEMINI_BASE_URL),
    )
    Pipe._get_or_create_genai_client.cache_clear()


def test_user_must_auth_user_tries_vertex_no_user_gemini_key_errors(
    pipe_instance_fixture,
):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=True. User tries Vertex without providing a fallback Gemini key.
    Expected: ValueError because Vertex usage is denied and no fallback is available.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = None

    user_valves_instance = Pipe.UserValves(
        USE_VERTEX_AI=True,
        VERTEX_PROJECT="user_tries_this_project",
        GEMINI_FREE_API_KEY=None,
        GEMINI_PAID_API_KEY=None,
    )
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_UNPRIVILEGED
    )

    assert merged_valves.USE_VERTEX_AI is False
    assert merged_valves.GEMINI_FREE_API_KEY is None
    assert merged_valves.GEMINI_PAID_API_KEY is None

    with pytest.raises(
        ValueError, match="User must provide their own authentication configuration"
    ):
        pipe._get_user_client(merged_valves, USER_EMAIL_UNPRIVILEGED)

    MockedGenAIClientConstructor.assert_not_called()
    Pipe._get_or_create_genai_client.cache_clear()


def test_user_must_auth_user_tries_vertex_with_user_gemini_key_falls_back(
    pipe_instance_fixture,
):
    """
    USER_MUST_PROVIDE_AUTH_CONFIG=True. User tries Vertex but provides a Gemini key.
    Expected: Falls back to Gemini Developer API using the user's provided key.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = None

    user_valves_instance = Pipe.UserValves(
        USE_VERTEX_AI=True,
        GEMINI_FREE_API_KEY=USER_FREE_KEY,
        GEMINI_PAID_API_KEY=USER_PAID_KEY,
        GEMINI_API_BASE_URL=USER_GEMINI_BASE_URL,
    )
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_UNPRIVILEGED
    )

    assert merged_valves.USE_VERTEX_AI is False
    assert merged_valves.VERTEX_PROJECT is None
    assert merged_valves.GEMINI_FREE_API_KEY == USER_FREE_KEY
    assert merged_valves.GEMINI_PAID_API_KEY == USER_PAID_KEY

    pipe._get_user_client(merged_valves, USER_EMAIL_UNPRIVILEGED)

    MockedGenAIClientConstructor.assert_called_once_with(
        api_key=USER_FREE_KEY,  # Should prefer user's free key for fallback
        http_options=gemini_types.HttpOptions(base_url=USER_GEMINI_BASE_URL),
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
    pipe.valves.USE_VERTEX_AI = False

    user_valves_instance = Pipe.UserValves()
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_WHITELISTED
    )

    pipe._get_user_client(merged_valves, USER_EMAIL_WHITELISTED)

    MockedGenAIClientConstructor.assert_called_once_with(
        api_key=ADMIN_FREE_KEY,
        http_options=gemini_types.HttpOptions(base_url=ADMIN_GEMINI_BASE_URL),
    )
    Pipe._get_or_create_genai_client.cache_clear()


def test_whitelist_user_no_uservalves_uses_admin_vertex_config(pipe_instance_fixture):
    """
    Whitelisted user with no UserValves uses admin's Vertex config.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = True
    pipe.valves.AUTH_WHITELIST = USER_EMAIL_WHITELISTED
    pipe.valves.USE_VERTEX_AI = True
    pipe.valves.VERTEX_PROJECT = ADMIN_VERTEX_PROJECT

    user_valves_instance = Pipe.UserValves()
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_WHITELISTED
    )

    pipe._get_user_client(merged_valves, USER_EMAIL_WHITELISTED)

    MockedGenAIClientConstructor.assert_called_once_with(
        vertexai=True,
        project=ADMIN_VERTEX_PROJECT,
        location=pipe.valves.VERTEX_LOCATION,
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
    pipe.valves.USE_VERTEX_AI = False
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
        http_options=gemini_types.HttpOptions(base_url=ADMIN_GEMINI_BASE_URL),
    )
    Pipe._get_or_create_genai_client.cache_clear()


# endregion Test USER_MUST_PROVIDE_AUTH_CONFIG=True with whitelisted user


# region Test user's ability to override admin's settings
def test_user_opts_out_of_admin_vertex_to_user_gemini(pipe_instance_fixture):
    """
    Admin uses Vertex. User opts-out to use their own Gemini key.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = False
    pipe.valves.USE_VERTEX_AI = True
    pipe.valves.VERTEX_PROJECT = ADMIN_VERTEX_PROJECT

    user_valves_instance = Pipe.UserValves(
        USE_VERTEX_AI=False,
        GEMINI_FREE_API_KEY=USER_FREE_KEY,
        GEMINI_API_BASE_URL=USER_GEMINI_BASE_URL,
    )
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_REGULAR
    )

    pipe._get_user_client(merged_valves, USER_EMAIL_REGULAR)

    MockedGenAIClientConstructor.assert_called_once_with(
        api_key=USER_FREE_KEY,
        http_options=gemini_types.HttpOptions(base_url=USER_GEMINI_BASE_URL),
    )
    Pipe._get_or_create_genai_client.cache_clear()


def test_user_opts_in_to_vertex_from_admin_gemini(pipe_instance_fixture):
    """
    Admin uses Gemini. User opts-in to use their own Vertex project.
    """
    pipe, MockedGenAIClientConstructor = pipe_instance_fixture
    Pipe._get_or_create_genai_client.cache_clear()

    pipe.valves.USER_MUST_PROVIDE_AUTH_CONFIG = False
    pipe.valves.USE_VERTEX_AI = False

    user_valves_instance = Pipe.UserValves(
        USE_VERTEX_AI=True,
        VERTEX_PROJECT=USER_VERTEX_PROJECT,
        VERTEX_LOCATION=USER_VERTEX_LOCATION,
    )
    merged_valves = pipe._get_merged_valves(
        pipe.valves, user_valves_instance, USER_EMAIL_REGULAR
    )

    pipe._get_user_client(merged_valves, USER_EMAIL_REGULAR)

    MockedGenAIClientConstructor.assert_called_once_with(
        vertexai=True, project=USER_VERTEX_PROJECT, location=USER_VERTEX_LOCATION
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
    model_id = "gemini-pro"
    __metadata__ = {
        "model": {"id": model_id},
        "canonical_model_id": model_id 
    }

    # Ensure both keys are present in the initial valves
    pipe.valves.GEMINI_FREE_API_KEY = ADMIN_FREE_KEY
    pipe.valves.GEMINI_PAID_API_KEY = ADMIN_PAID_KEY

    # Mock the request app state which is now required early in pipe()
    mock_request = MagicMock()
    mock_request.app.state._state = {
        "gemini_model_config": {model_id: {"pricing": {"free_tier": True}}}
    }

    def mock_toggle_side_effect(filter_id, metadata):
        if filter_id == "gemini_paid_api":
            return toggle_status
        return (False, False)

    with patch.object(
        pipe, "_get_toggleable_feature_status", side_effect=mock_toggle_side_effect
    ) as mock_toggle_status, patch.object(
        pipe, "_get_user_client", wraps=pipe._get_user_client
    ), patch.object(
        pipe, "_check_companion_filter_version"
    ), patch.object(
        pipe, "_get_merged_valves", return_value=pipe.valves
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
        assert mock_toggle_status.called
        args, _ = mock_toggle_status.call_args
        assert args[0] == "gemini_paid_api"
        assert args[1]["model"]["id"] == model_id

        # Assert the genai Client was initialized with the correct key
        MockedGenAIClientConstructor.assert_called()
        # Find the call that matches our expected key
        api_keys_used = [
            call.kwargs.get("api_key") 
            for call in MockedGenAIClientConstructor.call_args_list
        ]
        assert expected_api_key in api_keys_used

    Pipe._get_or_create_genai_client.cache_clear()

# endregion Test Toggleable Paid API Filter


# region Test _get_genai_models
# TODO: Add tests for _get_genai_models
# endregion Test _get_genai_models


# region Test GenerateContentConfig assembly
@pytest.mark.asyncio
async def test_build_config_level_thinking_model_does_not_send_dynamic_budget(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    model_id = "gemini-3.1-flash-image"
    pipe.valves.THINKING_BUDGET = -1
    pipe.valves.THINKING_LEVEL = "model_default"
    pipe.valves.IMAGE_RESOLUTION = "512"
    pipe.valves.IMAGE_ASPECT_RATIO = "4:5"

    model_config = {
        model_id: {
            "capabilities": {"thinking": True, "image_generation": True},
            "thinking_config": {
                "mode": "level",
                "default_level": "minimal",
                "supported_levels": ["minimal", "high"],
            },
            "image_config": {
                "supported_resolutions": ["512", "1K", "2K", "4K"],
                "supported_aspect_ratios": ["1:1", "4:5", "16:9"],
            },
        }
    }
    metadata = {
        "canonical_model_id": model_id,
        "features": {},
        "merged_custom_params": {},
    }

    with patch.object(
        pipe,
        "_get_toggleable_feature_status",
        AsyncMock(return_value=(False, False)),
    ):
        config = await pipe._build_gen_content_config(
            body={},
            __metadata__=metadata,
            valves=pipe.valves,
            config=model_config,
        )

    assert config.response_modalities == ["TEXT", "IMAGE"]
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_budget is None
    assert config.thinking_config.thinking_level is None
    assert config.image_config is not None
    assert config.image_config.image_size == "512"
    assert config.image_config.aspect_ratio == "4:5"
    assert metadata["image_generation_config_status"] == (
        "Requesting image output with aspect ratio 4:5 and resolution 512."
    )

    json_dump = config.model_dump(exclude_none=True, mode="json", by_alias=True)
    assert json_dump["responseModalities"] == ["TEXT", "IMAGE"]
    assert json_dump["imageConfig"]["imageSize"] == "512"
    assert json_dump["imageConfig"]["aspectRatio"] == "4:5"


@pytest.mark.asyncio
async def test_build_config_image_only_and_thinking_level_for_lite_image_model(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    model_id = "gemini-3.1-flash-lite-image"
    pipe.valves.IMAGE_OUTPUT_FORMAT = "Images only"
    pipe.valves.THINKING_LEVEL = "high"
    pipe.valves.THINKING_BUDGET = 8192
    pipe.valves.IMAGE_RESOLUTION = "4K"
    pipe.valves.IMAGE_ASPECT_RATIO = "16:9"

    model_config = {
        model_id: {
            "capabilities": {"thinking": True, "image_generation": True},
            "thinking_config": {
                "mode": "level",
                "default_level": "minimal",
                "supported_levels": ["minimal", "high"],
            },
            "image_config": {
                "supported_resolutions": ["1K"],
                "supported_aspect_ratios": ["1:1", "16:9"],
            },
        }
    }
    metadata = {
        "canonical_model_id": model_id,
        "features": {},
        "merged_custom_params": {},
    }

    with patch.object(
        pipe,
        "_get_toggleable_feature_status",
        AsyncMock(return_value=(False, False)),
    ):
        config = await pipe._build_gen_content_config(
            body={},
            __metadata__=metadata,
            valves=pipe.valves,
            config=model_config,
        )

    assert config.response_modalities == ["IMAGE"]
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_budget is None
    assert config.thinking_config.thinking_level == gemini_types.ThinkingLevel.HIGH
    assert config.image_config is not None
    assert config.image_config.image_size is None
    assert config.image_config.aspect_ratio == "16:9"

    json_dump = config.model_dump(exclude_none=True, mode="json", by_alias=True)
    assert json_dump["responseModalities"] == ["IMAGE"]
    assert json_dump["thinkingConfig"]["thinkingLevel"] == "HIGH"
    assert json_dump["imageConfig"]["aspectRatio"] == "16:9"


@pytest.mark.asyncio
async def test_build_config_uses_nearest_supported_image_aspect_ratio(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    model_id = "gemini-3-pro-image"
    pipe.valves.IMAGE_ASPECT_RATIO = "1:4"

    model_config = {
        model_id: {
            "capabilities": {"thinking": True, "image_generation": True},
            "thinking_config": {
                "mode": "level",
                "default_level": "high",
                "supported_levels": ["low", "high"],
            },
            "image_config": {
                "supported_resolutions": ["1K", "2K", "4K"],
                "supported_aspect_ratios": [
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
                ],
            },
        }
    }
    metadata = {
        "canonical_model_id": model_id,
        "features": {},
        "merged_custom_params": {},
    }

    with patch.object(
        pipe,
        "_get_toggleable_feature_status",
        AsyncMock(return_value=(False, False)),
    ):
        config = await pipe._build_gen_content_config(
            body={},
            __metadata__=metadata,
            valves=pipe.valves,
            config=model_config,
        )

    assert config.image_config is not None
    assert config.image_config.aspect_ratio == "9:16"
    assert metadata["image_aspect_ratio_fallback_status"] == (
        "Aspect ratio 1:4 is not supported by gemini-3-pro-image; "
        "using nearest supported ratio 9:16."
    )


def test_nearest_supported_aspect_ratio_preserves_orientation(pipe_instance_fixture):
    pipe, _ = pipe_instance_fixture
    supported_ratios = [
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

    assert pipe._nearest_supported_aspect_ratio("1:8", supported_ratios) == "9:16"
    assert pipe._nearest_supported_aspect_ratio("8:1", supported_ratios) == "21:9"


@pytest.mark.asyncio
async def test_build_config_budget_thinking_model_keeps_budget_path(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    model_id = "gemini-2.5-flash"
    pipe.valves.THINKING_BUDGET = -1
    pipe.valves.THINKING_LEVEL = "high"

    model_config = {
        model_id: {
            "capabilities": {"thinking": True, "image_generation": False},
        }
    }
    metadata = {
        "canonical_model_id": model_id,
        "features": {},
        "merged_custom_params": {},
    }

    with patch.object(
        pipe,
        "_get_toggleable_feature_status",
        AsyncMock(return_value=(False, False)),
    ):
        config = await pipe._build_gen_content_config(
            body={},
            __metadata__=metadata,
            valves=pipe.valves,
            config=model_config,
        )

    assert config.response_modalities == ["TEXT"]
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_budget == -1
    assert config.thinking_config.thinking_level is None


def test_high_resolution_image_generation_forces_non_streaming(pipe_instance_fixture):
    pipe, _ = pipe_instance_fixture
    pipe.valves.IMAGE_RESOLUTION = "2K"
    model_id = "gemini-3.1-flash-image"
    model_config = {
        model_id: {
            "capabilities": {"image_generation": True},
        }
    }

    assert pipe._should_force_non_streaming_for_image_generation(
        use_streaming_api=True,
        valves=pipe.valves,
        model_id=model_id,
        model_config=model_config,
    )
    assert not pipe._should_force_non_streaming_for_image_generation(
        use_streaming_api=False,
        valves=pipe.valves,
        model_id=model_id,
        model_config=model_config,
    )

    pipe.valves.IMAGE_RESOLUTION = "1K"
    assert not pipe._should_force_non_streaming_for_image_generation(
        use_streaming_api=True,
        valves=pipe.valves,
        model_id=model_id,
        model_config=model_config,
    )


# endregion Test GenerateContentConfig assembly


# region Test GeminiContentBuilder
@pytest.mark.asyncio
async def test_builder_build_contents_simple_user_text(pipe_instance_fixture):
    """
    Tests conversion of a simple user text message into genai.types.Content.
    """
    # Reset mocks for test isolation
    mock_chats_module.reset_mock()
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
    # Create a mock for the new dependency
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
        files_api_manager=mock_files_api_manager,  # Pass the new mock
        pdf_mitigation_manager=pipe_instance.pdf_mitigation_manager,
    )

    with patch(
        "plugins.pipes.gemini_manifold.types.Part.from_text"
    ) as mock_part_from_text:
        # The mock Part object needs a 'text' attribute for the new check in `build_contents`.
        mock_text_part = MagicMock(spec=gemini_types.Part)
        mock_text_part.text = "Hello!"
        mock_part_from_text.return_value = mock_text_part

        contents = await builder.build_contents()

        mock_misc_module.pop_system_message.assert_called_once_with(messages_body)
        mock_part_from_text.assert_called_once_with(text="Hello!")
        assert len(contents) == 1
        content_item = contents[0]
        assert content_item.role == "user"
        assert content_item.parts is not None
        assert len(content_item.parts) == 1
        assert content_item.parts[0] == mock_text_part
        # A warning toast is emitted when messages_db is not found
        # The call uses positional arguments (msg, toastType), so we match that here.
        mock_event_emitter.emit_toast.assert_called_once_with(ANY, "warning")


@pytest.mark.asyncio
async def test_builder_build_contents_youtube_link_mixed_with_text(
    pipe_instance_fixture,
):
    """
    Tests that a user message with text and a YouTube URL is correctly
    parsed into interleaved text and file_data parts.
    """
    # Reset mocks for test isolation
    mock_chats_module.reset_mock()
    mock_misc_module.reset_mock()

    pipe_instance, _ = pipe_instance_fixture
    # Ensure we test the non-Vertex AI path for file_data creation
    pipe_instance.valves.USE_VERTEX_AI = False

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
    # Create a mock for the new dependency
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
    )

    # Mock part objects need 'text' attribute for new checks
    mock_text_part_before_obj = MagicMock(spec=gemini_types.Part, name="TextPartBefore")
    mock_text_part_before_obj.text = text_before_stripped
    mock_text_part_after_obj = MagicMock(spec=gemini_types.Part, name="TextPartAfter")
    mock_text_part_after_obj.text = text_after_stripped

    with patch(
        "plugins.pipes.gemini_manifold.types.Part.from_text"
    ) as mock_part_from_text:

        def from_text_side_effect(text):
            if text == text_before_stripped:
                return mock_text_part_before_obj
            if text == text_after_stripped:
                return mock_text_part_after_obj
            generic_mock = MagicMock(
                spec=gemini_types.Part, name=f"GenericTextPart_{text[:10]}"
            )
            generic_mock.text = text
            return generic_mock

        mock_part_from_text.side_effect = from_text_side_effect

        # Act
        contents = await builder.build_contents()

        # Assert
        assert len(contents) == 1
        content_item = contents[0]
        assert content_item.role == "user"
        assert content_item.parts is not None
        assert len(content_item.parts) == 3

        # Assertions for correct part segmentation and ordering
        assert mock_part_from_text.call_count == 2
        expected_calls = [
            call(text=text_before_stripped),
            call(text=text_after_stripped),
        ]
        mock_part_from_text.assert_has_calls(expected_calls, any_order=True)

        # Part 1: Text before the YouTube link
        assert content_item.parts[0] is mock_text_part_before_obj

        # Part 2: The YouTube link
        youtube_part = content_item.parts[1]
        assert isinstance(youtube_part, gemini_types.Part)
        assert hasattr(youtube_part, "file_data")
        assert youtube_part.file_data is not None
        assert youtube_part.file_data.file_uri == youtube_url
        # A real file part should have a falsy .text attribute
        assert not youtube_part.text

        # Part 3: Text after the YouTube link
        assert content_item.parts[2] is mock_text_part_after_obj

        # A warning toast is emitted when messages_db is not found
        # The call uses positional arguments (msg, toastType), so we match that here.
        mock_event_emitter.emit_toast.assert_called_once_with(ANY, "warning")


@pytest.mark.asyncio
async def test_builder_build_contents_user_text_with_pdf(pipe_instance_fixture):
    """
    Tests conversion of a user message with text and an attached PDF file.
    """
    # Reset mocks for test isolation
    mock_chats_module.reset_mock()
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
                    "files": [{"id": pdf_file_id, "type": "file"}],
                },
            },
        }
    }

    # Mock the DB call and system prompt extraction
    mock_chats_module.Chats.get_chat_by_id_and_user_id.return_value = mock_chat_from_db
    mock_misc_module.pop_system_message.return_value = (None, messages_body)

    # Create a mock for the new dependency and its methods
    pipe_instance.valves.PDF_LIMIT_MITIGATION = False
    mock_files_api_manager = AsyncMock()
    mock_files_api_manager.client.vertexai = False
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
    )

    # REMOVED: No longer need to create a mock Part object for text.
    # mock_text_part_obj = MagicMock(spec=gemini_types.Part, name="TextPart")

    # Patch _get_file_source to be async.
    # REMOVED: The patch for `types.Part.from_text` is removed to let the real method run.
    with patch(
        "plugins.pipes.gemini_manifold.GeminiContentBuilder._get_file_source",
        new_callable=AsyncMock,
        return_value=LocalFileSource(
            file_bytes=fake_pdf_bytes,
            file_path=None,
            mime_type=pdf_mime_type,
        ),
    ) as mock_get_file_source:
        # Act
        contents = await builder.build_contents()

        # Assert
        mock_chats_module.Chats.get_chat_by_id_and_user_id.assert_called_once_with(
            id="test_chat_id", user_id="test_user_id"
        )
        mock_get_file_source.assert_awaited_once_with(pdf_file_id)

        mock_files_api_manager.get_or_upload_file.assert_awaited_once_with(
            file_bytes=fake_pdf_bytes,
            mime_type=pdf_mime_type,
            owui_file_id=pdf_file_id,
            status_queue=ANY,
        )

        # This assertion is no longer needed as we are not mocking `from_text`
        # mock_part_from_text.assert_called_once_with(text=user_text_content)

        assert len(contents) == 1
        user_content_obj = contents[0]
        assert user_content_obj.role == "user"
        assert user_content_obj.parts is not None
        assert len(user_content_obj.parts) == 2

        # Assert the PDF part was created correctly
        pdf_part = user_content_obj.parts[0]
        assert isinstance(pdf_part, gemini_types.Part)
        assert pdf_part.file_data is not None
        assert pdf_part.file_data.file_uri == mock_gemini_file.uri
        assert pdf_part.file_data.mime_type == mock_gemini_file.mime_type

        # CHANGED: Assert the text part by inspecting its properties, not its identity
        text_part = user_content_obj.parts[1]
        assert isinstance(text_part, gemini_types.Part)
        assert text_part.text == user_text_content

        mock_event_emitter.assert_not_called()


@pytest.mark.asyncio
async def test_create_genai_parts_optimizes_pdf_with_synthetic_id(
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
    mock_files_api_manager.client.vertexai = False
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
    )

    parts = await builder._create_genai_parts_from_file_data(
        file_bytes=original_pdf_bytes,
        mime_type=pdf_mime_type,
        owui_file_id=pdf_file_id,
        status_queue=asyncio.Queue(),
        source_name="large.pdf",
    )

    assert len(parts) == 1
    assert parts[0].file_data is not None
    mock_pdf_mitigation_manager.prepare.assert_awaited_once_with(
        file_bytes=original_pdf_bytes,
        file_path=None,
    )
    mock_files_api_manager.get_or_upload_file_from_path.assert_awaited_once()
    upload_kwargs = (
        mock_files_api_manager.get_or_upload_file_from_path.await_args.kwargs
    )
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
async def test_create_genai_parts_splits_pdf_in_order(pipe_instance_fixture, tmp_path):
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
                    for i, (path, chunk) in enumerate(
                        zip(chunk_paths, chunks, strict=True)
                    )
                ],
                page_count=2401,
                was_mitigated=True,
            ),
        )
    )
    mock_files_api_manager = AsyncMock()
    mock_files_api_manager.client.vertexai = False
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
    )

    parts = await builder._create_genai_parts_from_file_data(
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
    assert parts[0].text is not None
    assert "very-large.pdf" in parts[0].text
    assert "3 consecutive attachments" in parts[0].text
    assert (
        "PDF 'very-large.pdf', attachment 1: original document pages 1-46"
        in parts[0].text
    )
    assert (
        "PDF 'very-large.pdf', attachment 2: original document pages 47-92"
        in parts[0].text
    )
    assert "do not restart page numbering at 1" in parts[0].text
    assert [part.file_data.file_uri for part in parts[1:]] == [  # type: ignore[union-attr]
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

    with patch.object(
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
                for i, (path, chunk) in enumerate(
                    zip(chunk_paths, chunks, strict=True)
                )
            ],
            page_count=1200,
            was_mitigated=True,
        ),
    ) as mock_prepare, patch.object(
        manager,
        "_write_temp_source",
        wraps=manager._write_temp_source,
    ) as mock_write_temp_source:
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
    assert [part.path for part in first_outcome.result.parts] == [
        str(path) for path in chunk_paths
    ]


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
    page_counts = [
        processor._count_pages_from_path(pikepdf, part.path)
        for part in result.parts
    ]
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
    mock_files_api_manager.client.vertexai = False

    builder = GeminiContentBuilder(
        messages_body=[{"role": "user", "content": "hello"}],  # type: ignore
        metadata_body={"chat_id": "local", "features": {"upload_documents": True}},  # type: ignore
        user_data={"id": "test_user_id", "email": "test@example.com"},  # type: ignore
        event_emitter=mock_event_emitter,
        valves=pipe_instance.valves,
        files_api_manager=mock_files_api_manager,
        pdf_mitigation_manager=pipe_instance.pdf_mitigation_manager,
    )

    with patch.object(
        builder,
        "_process_message_turn",
        new_callable=AsyncMock,
        side_effect=PDFProcessingError("failed to process PDF"),
    ):
        with pytest.raises(ContentBuildError, match="failed to process PDF"):
            await builder.build_contents()


# endregion Test GeminiContentBuilder


# region Test usage data
def test_usage_data_counts_image_tokens_with_genai_enum_modalities(
    pipe_instance_fixture,
):
    pipe, _ = pipe_instance_fixture
    model_id = "gemini-3.1-flash-image"
    response = gemini_types.GenerateContentResponse(
        usage_metadata=gemini_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=30,
            candidates_tokens_details=[
                gemini_types.ModalityTokenCount(
                    modality=gemini_types.MediaModality.TEXT,
                    token_count=10,
                ),
                gemini_types.ModalityTokenCount(
                    modality=gemini_types.MediaModality.IMAGE,
                    token_count=20,
                ),
            ],
            total_token_count=130,
        )
    )
    app_state = MagicMock()
    app_state._state = {
        "gemini_model_config": {
            model_id: {
                "pricing": {
                    "input": [{"up_to_tokens": None, "price_per_million": 0.50}],
                    "output": [{"up_to_tokens": None, "price_per_million": 3.00}],
                    "image_output": [
                        {"up_to_tokens": None, "price_per_million": 60.00}
                    ],
                }
            }
        }
    }

    usage = pipe._get_usage_data(
        response=response,
        app_state=app_state,
        metadata={"is_paid_api": True, "canonical_model_id": model_id},
        start_time=0,
    )

    assert usage is not None
    assert usage["token_details"]["candidates_tokens_details"] == [
        {"modality": "TEXT", "token_count": 10},
        {"modality": "IMAGE", "token_count": 20},
    ]
    assert usage["cost_details"]["input_cost"] == 0.00005
    assert usage["cost_details"]["output_cost"] == 0.00003
    assert usage["cost_details"]["image_output_cost"] == 0.0012
    assert usage["cost_details"]["total_cost"] == 0.00128


# endregion Test usage data


def teardown_module(module):
    """Cleans up sys.modules after tests in this file are done."""
    del sys.modules["open_webui.models.chats"]
    del sys.modules["open_webui.models.files"]
    del sys.modules["open_webui.models.functions"]
    del sys.modules["open_webui.storage.provider"]
    del sys.modules["open_webui.utils.misc"]
