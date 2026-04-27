"""Utility function for retrieving SSL version and list of ciphers for TLS secutiry profile."""

import logging
import ssl
from typing import Optional

from ols import constants
from ols.app.models.config import TLSSecurityProfile
from ols.utils import tls

logger = logging.getLogger(__name__)


def get_ssl_version(sec_profile: Optional[TLSSecurityProfile]) -> int:
    """Get SSL protocol constant for TLS context creation."""
    logger.info("Using SSL protocol version: %s", ssl.PROTOCOL_TLS_SERVER)
    return ssl.PROTOCOL_TLS_SERVER


def get_min_tls_version(
    sec_profile: Optional[TLSSecurityProfile],
) -> Optional[ssl.TLSVersion]:
    """Get minimum TLS version to enforce on the SSL context."""
    if sec_profile is None or sec_profile.profile_type is None:
        return None

    min_tls_version = tls.min_tls_version(
        sec_profile.min_tls_version, sec_profile.profile_type
    )
    logger.info("min TLS version: %s", min_tls_version)

    resolved_min_tls_version = tls.ssl_tls_version(min_tls_version)
    logger.info("Using minimum TLS version: %s", resolved_min_tls_version)
    return resolved_min_tls_version


def get_ciphers(sec_profile: Optional[TLSSecurityProfile]) -> str:
    """Get allowed ciphers to be used. It can be configured in tls_security_profile section."""
    # if security profile is not set, use default ciphers
    # as specified in SSL library
    if sec_profile is None or sec_profile.profile_type is None:
        logger.info("Allowing default ciphers: %s", constants.DEFAULT_SSL_CIPHERS)
        return constants.DEFAULT_SSL_CIPHERS

    # security profile is set -> we need to retrieve ciphers to be allowed
    ciphers = tls.ciphers_as_string(sec_profile.ciphers, sec_profile.profile_type)
    logger.info("Allowing following ciphers: %s", ciphers)
    return ciphers
