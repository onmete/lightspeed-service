"""Utility functions for LLM providers configuration."""

import json

from google.auth.credentials import Credentials as GoogleCredentials
from google.oauth2 import credentials as oauth2_credentials
from google.oauth2 import service_account

# Vertex AI service account requires this scope to access the API
# https://github.com/googleapis/python-genai/issues/2#issuecomment-2537279484
VERTEX_AI_OAUTH_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/cloud-platform",
)


def load_vertex_credentials(credentials_json: str) -> GoogleCredentials:
    """Build Vertex-scoped Google credentials from JSON file contents.

    Supports ``type: service_account`` (key file) and ``type: authorized_user``
    (refresh-token / user OAuth JSON as produced by application-default login).
    The JSON object must include a non-empty ``type`` field.

    Args:
        credentials_json: Raw JSON string read from the credentials file.

    Returns:
        Credentials instance scoped with ``VERTEX_AI_OAUTH_SCOPES``.

    Raises:
        TypeError: If JSON does not decode to an object.
        ValueError: If type is missing/unsupported or required fields are absent.
    """
    parsed = json.loads(credentials_json)
    if not isinstance(parsed, dict):
        msg = "credentials must be a JSON object"
        raise TypeError(msg)
    cred_type = parsed.get("type")
    if not cred_type:
        msg = 'Google credentials JSON must include a non-empty string "type" field'
        raise ValueError(msg)
    scopes = list(VERTEX_AI_OAUTH_SCOPES)
    if cred_type == "service_account":
        return service_account.Credentials.from_service_account_info(
            parsed,
            scopes=scopes,
        )  # type: ignore[no-untyped-call]
    if cred_type == "authorized_user":
        return oauth2_credentials.Credentials.from_authorized_user_info(
            parsed,
            scopes=scopes,
        )  # type: ignore[no-untyped-call]
    msg = f"Unsupported Google credential type for Vertex: {cred_type!r}"
    raise ValueError(msg)
