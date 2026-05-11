"""Unit tests for Google Vertex AI providers (Gemini and Anthropic on Vertex)."""

import json
from unittest.mock import patch

import pytest

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.google_vertex import GoogleVertex, GoogleVertexAnthropic

from .utils import generate_service_account_json_string


@pytest.fixture
def gemini_provider_config(tmpdir):
    """Return provider configuration for Vertex Gemini."""
    credentials_json = generate_service_account_json_string()
    p = tmpdir.mkdir("sub").join("service-account.json")
    p.write(credentials_json)
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "google_vertex",
            "credentials_path": p.strpath,
            "project_id": "my-gcp-project",
            "models": [
                {
                    "name": "gemini-2.5-flash",
                }
            ],
        }
    )


@pytest.fixture
def gemini_provider_config_with_specific_parameters(tmpdir):
    """Return Gemini Vertex config with explicit google_vertex_config."""
    credentials_json = generate_service_account_json_string()
    p = tmpdir.mkdir("sub").join("service-account.json")
    p.write(credentials_json)
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "google_vertex",
            "url": "https://us-central1-aiplatform.googleapis.com",
            "credentials_path": p.strpath,
            "google_vertex_config": {
                "project": "my-specific-project",
                "location": "us-central1",
            },
            "models": [
                {
                    "name": "gemini-2.5-flash",
                }
            ],
        }
    )


@pytest.fixture
def gemini_provider_config_authorized_user(tmpdir):
    """Return Vertex Gemini config with authorized_user credentials JSON."""
    credentials_json = json.dumps(
        {
            "type": "authorized_user",
            "client_id": "authorized-user-client-id",
            "client_secret": "authorized-user-client-secret",
            "refresh_token": "authorized-user-refresh-token",
        }
    )
    p = tmpdir.mkdir("sub").join("authorized-user.json")
    p.write(credentials_json)
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "google_vertex",
            "url": "https://us-central1-aiplatform.googleapis.com",
            "credentials_path": p.strpath,
            "google_vertex_config": {
                "project": "my-specific-project",
                "location": "us-central1",
            },
            "models": [
                {
                    "name": "gemini-2.5-flash",
                }
            ],
        }
    )


@pytest.fixture
def anthropic_provider_config(tmpdir):
    """Return provider configuration for Vertex Anthropic."""
    credentials_json = generate_service_account_json_string()
    p = tmpdir.mkdir("sub").join("service-account.json")
    p.write(credentials_json)
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "google_vertex_anthropic",
            "url": "https://us-east5-aiplatform.googleapis.com",
            "credentials_path": p.strpath,
            "google_vertex_anthropic_config": {
                "project": "my-specific-project",
                "location": "us-east5",
            },
            "models": [
                {
                    "name": "claude-opus-4-6",
                }
            ],
        }
    )


@pytest.fixture
def anthropic_provider_config_with_specific_parameters(tmpdir):
    """Return Vertex Anthropic config with alternate region."""
    credentials_json = generate_service_account_json_string()
    p = tmpdir.mkdir("sub").join("service-account.json")
    p.write(credentials_json)
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "google_vertex_anthropic",
            "url": "https://europe-west1-aiplatform.googleapis.com",
            "credentials_path": p.strpath,
            "google_vertex_anthropic_config": {
                "project": "my-specific-project",
                "location": "europe-west1",
            },
            "models": [
                {
                    "name": "claude-opus-4-6",
                }
            ],
        }
    )


@pytest.fixture
def anthropic_provider_config_authorized_user(tmpdir):
    """Return Vertex Anthropic config with authorized_user credentials JSON."""
    credentials_json = json.dumps(
        {
            "type": "authorized_user",
            "client_id": "authorized-user-client-id",
            "client_secret": "authorized-user-client-secret",
            "refresh_token": "authorized-user-refresh-token",
        }
    )
    p = tmpdir.mkdir("sub").join("authorized-user.json")
    p.write(credentials_json)
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "google_vertex_anthropic",
            "url": "https://us-east5-aiplatform.googleapis.com",
            "credentials_path": p.strpath,
            "google_vertex_anthropic_config": {
                "project": "my-specific-project",
                "location": "us-east5",
            },
            "models": [
                {
                    "name": "claude-opus-4-6",
                }
            ],
        }
    )


@patch(
    "ols.src.llms.providers.google_vertex.ChatGoogleGenerativeAI",
    autospec=True,
)
def test_gemini_basic_interface(mock_chat, gemini_provider_config):
    """Test Gemini Vertex basic interface."""
    vertex = GoogleVertex(
        model="gemini-2.5-flash", params={}, provider_config=gemini_provider_config
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params
    assert "model" in vertex.default_params
    assert "project" in vertex.default_params
    assert "location" in vertex.default_params
    assert "max_output_tokens" in vertex.default_params
    assert vertex.default_params["project"] == "my-gcp-project"
    assert vertex.default_params["location"] == "global"
    assert vertex.default_params["vertexai"] is True

    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["project"] == "my-gcp-project"
    assert call_kwargs["location"] == "global"
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert call_kwargs["vertexai"] is True
    assert "base_url" not in call_kwargs


@patch(
    "ols.src.llms.providers.google_vertex.ChatGoogleGenerativeAI",
    autospec=True,
)
def test_gemini_params_handling(mock_chat, gemini_provider_config):
    """Test Gemini Vertex strips disallowed parameters before model init."""
    params = {
        "unknown_parameter": "foo",
        "min_new_tokens": 1,
        "max_new_tokens": 10,
        "temperature": 0.3,
    }

    vertex = GoogleVertex(
        model="gemini-2.5-flash", params=params, provider_config=gemini_provider_config
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params
    assert vertex.params

    assert "temperature" in vertex.params
    assert vertex.params["temperature"] == 0.3

    assert "min_new_tokens" not in vertex.params
    assert "max_new_tokens" not in vertex.params
    assert "unknown_parameter" not in vertex.params


@patch(
    "ols.src.llms.providers.google_vertex.ChatGoogleGenerativeAI",
    autospec=True,
)
def test_gemini_loading_provider_specific_parameters(
    mock_chat, gemini_provider_config_with_specific_parameters
):
    """Test Gemini Vertex google_vertex_config overrides project and location."""
    vertex = GoogleVertex(
        model="gemini-2.5-flash",
        params={},
        provider_config=gemini_provider_config_with_specific_parameters,
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params
    assert vertex.params

    assert vertex.project == "my-specific-project"
    assert vertex.location == "us-central1"
    assert vertex.default_params["project"] == "my-specific-project"
    assert vertex.default_params["location"] == "us-central1"

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["project"] == "my-specific-project"
    assert call_kwargs["location"] == "us-central1"
    assert call_kwargs["base_url"] == "https://us-central1-aiplatform.googleapis.com"


@patch(
    "ols.src.llms.providers.google_vertex.ChatGoogleGenerativeAI",
    autospec=True,
)
def test_gemini_authorized_user_credentials(
    mock_chat, gemini_provider_config_authorized_user
):
    """Test Gemini on Vertex accepts authorized_user credentials JSON."""
    vertex = GoogleVertex(
        model="gemini-2.5-flash",
        params={},
        provider_config=gemini_provider_config_authorized_user,
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params

    payload = json.loads(gemini_provider_config_authorized_user.credentials)
    expected_refresh_token = payload["refresh_token"]
    default_credentials = vertex.default_params["credentials"]
    assert default_credentials.refresh_token == expected_refresh_token

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["project"] == "my-specific-project"
    assert call_kwargs["location"] == "us-central1"
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert call_kwargs["vertexai"] is True
    assert call_kwargs["base_url"] == "https://us-central1-aiplatform.googleapis.com"


@patch(
    "ols.src.llms.providers.google_vertex.ChatAnthropicVertex",
    autospec=True,
)
def test_anthropic_basic_interface(mock_chat, anthropic_provider_config):
    """Test Anthropic on Vertex basic interface."""
    vertex = GoogleVertexAnthropic(
        model="claude-opus-4-6", params={}, provider_config=anthropic_provider_config
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params
    assert "model_name" in vertex.default_params
    assert "project" in vertex.default_params
    assert "location" in vertex.default_params
    assert "max_output_tokens" in vertex.default_params
    assert vertex.default_params["project"] == "my-specific-project"
    assert vertex.default_params["location"] == "us-east5"

    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["project"] == "my-specific-project"
    assert call_kwargs["location"] == "us-east5"
    assert call_kwargs["model_name"] == "claude-opus-4-6"


@patch(
    "ols.src.llms.providers.google_vertex.ChatAnthropicVertex",
    autospec=True,
)
def test_anthropic_params_handling(mock_chat, anthropic_provider_config):
    """Test Anthropic on Vertex strips disallowed parameters before model init."""
    params = {
        "unknown_parameter": "foo",
        "min_new_tokens": 1,
        "max_new_tokens": 10,
        "temperature": 0.3,
    }

    vertex = GoogleVertexAnthropic(
        model="claude-opus-4-6",
        params=params,
        provider_config=anthropic_provider_config,
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params
    assert vertex.params

    assert "temperature" in vertex.params
    assert vertex.params["temperature"] == 0.3

    assert "min_new_tokens" not in vertex.params
    assert "max_new_tokens" not in vertex.params
    assert "unknown_parameter" not in vertex.params


@patch(
    "ols.src.llms.providers.google_vertex.ChatAnthropicVertex",
    autospec=True,
)
def test_anthropic_loading_provider_specific_parameters(
    mock_chat, anthropic_provider_config_with_specific_parameters
):
    """Test Anthropic on Vertex config overrides region."""
    vertex = GoogleVertexAnthropic(
        model="claude-opus-4-6",
        params={},
        provider_config=anthropic_provider_config_with_specific_parameters,
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params
    assert vertex.params

    assert vertex.project == "my-specific-project"
    assert vertex.location == "europe-west1"
    assert vertex.default_params["project"] == "my-specific-project"
    assert vertex.default_params["location"] == "europe-west1"

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["project"] == "my-specific-project"
    assert call_kwargs["location"] == "europe-west1"


@patch(
    "ols.src.llms.providers.google_vertex.ChatAnthropicVertex",
    autospec=True,
)
def test_anthropic_authorized_user_credentials(
    mock_chat, anthropic_provider_config_authorized_user
):
    """Test Anthropic on Vertex accepts authorized_user credentials JSON."""
    vertex = GoogleVertexAnthropic(
        model="claude-opus-4-6",
        params={},
        provider_config=anthropic_provider_config_authorized_user,
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params

    payload = json.loads(anthropic_provider_config_authorized_user.credentials)
    expected_refresh_token = payload["refresh_token"]
    default_credentials = vertex.default_params["credentials"]
    assert default_credentials.refresh_token == expected_refresh_token

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["project"] == "my-specific-project"
    assert call_kwargs["location"] == "us-east5"
