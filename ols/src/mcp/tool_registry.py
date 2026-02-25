"""MCP tool metadata registry for tracking tools with metadata."""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from ols import config

logger = logging.getLogger(__name__)


@dataclass
class ToolMetadata:
    """Metadata for a tool from MCP server.

    Stores the server name and the full metadata dictionary from the tool definition.
    Use helper functions to extract specific metadata like UI resource URIs or visibility.
    """

    server_name: str
    meta: Optional[dict[str, Any]] = None


_tool_meta_registry: dict[str, ToolMetadata] = {}
_resource_uri_to_config_name: dict[str, str] = {}


def get_tool_metadata(tool_name: str) -> Optional[ToolMetadata]:
    """Get metadata for a tool.

    Args:
        tool_name: The name of the tool.

    Returns:
        ToolMetadata if the tool has metadata, None otherwise.
    """
    return _tool_meta_registry.get(tool_name)


def get_ui_resource_uri(metadata: ToolMetadata) -> Optional[str]:
    """Extract the resource URI from tool metadata.

    Args:
        metadata: The tool metadata.

    Returns:
        The resource URI if present, None otherwise.
    """
    if not metadata.meta:
        return None

    # Check both formats: _meta.ui.resourceUri and _meta["ui/resourceUri"]
    ui_meta = metadata.meta.get("ui", {}) if isinstance(metadata.meta, dict) else {}
    resource_uri = ui_meta.get("resourceUri") if isinstance(ui_meta, dict) else None
    if not resource_uri and isinstance(metadata.meta, dict):
        resource_uri = metadata.meta.get("ui/resourceUri")
    return resource_uri


def get_visibility(metadata: ToolMetadata) -> Optional[list[str]]:
    """Extract the visibility list from tool metadata.

    Args:
        metadata: The tool metadata.

    Returns:
        The visibility list if present, None otherwise.
    """
    if not metadata.meta:
        return None

    ui_meta = metadata.meta.get("ui", {}) if isinstance(metadata.meta, dict) else {}
    return ui_meta.get("visibility") if isinstance(ui_meta, dict) else None


def is_model_visible(tool_name: str) -> bool:
    """Check whether a tool should be visible to the LLM.

    Tools with _meta.ui.visibility that does not include "model" are app-only
    and should not be bound to the LLM. Tools without UI metadata or without
    an explicit visibility constraint default to model-visible.

    Args:
        tool_name: The name of the tool.

    Returns:
        True if the tool should be included in LLM tool binding.
    """
    metadata = get_tool_metadata(tool_name)
    if not metadata:
        return True
    visibility = get_visibility(metadata)
    if not visibility:
        return True
    return "model" in visibility


def get_config_name_for_resource_uri(resource_uri: str) -> Optional[str]:
    """Get the config server name for a ui:// resource URI.

    The full URI is treated as an opaque identifier; no internal structure
    (authority, path) is assumed.

    Args:
        resource_uri: The full ui:// resource URI.

    Returns:
        The config server name if the URI was discovered, None otherwise.
    """
    return _resource_uri_to_config_name.get(resource_uri)


def register_tool_metadata(
    tool_name: str,
    server_name: str,
    meta: Optional[dict[str, Any]] = None,
) -> None:
    """Register a tool's metadata.

    Args:
        tool_name: The name of the tool.
        server_name: The MCP server name.
        meta: The full metadata dictionary from the tool definition.
    """
    logger.info(
        "Registered metadata for tool '%s' from server '%s'",
        tool_name,
        server_name,
    )
    metadata = ToolMetadata(server_name=server_name, meta=meta)
    _tool_meta_registry[tool_name] = metadata

    resource_uri = get_ui_resource_uri(metadata)

    # Register resource URI mapping if present
    if resource_uri and resource_uri.startswith("ui://"):
        existing = _resource_uri_to_config_name.get(resource_uri)
        if existing and existing != server_name:
            logger.error(
                "Resource URI '%s' already mapped to server '%s', "
                "overwriting with '%s' - one server's UI resources "
                "will be unreachable",
                resource_uri,
                existing,
                server_name,
            )
        _resource_uri_to_config_name[resource_uri] = server_name
        logger.info(
            "Mapped resource URI '%s' to config server '%s'",
            resource_uri,
            server_name,
        )


async def discover_tool_metadata() -> None:
    """Discover and register metadata for all configured MCP servers.

    This queries each MCP server for its tools and checks for metadata.
    """
    if not config.mcp_servers or not config.mcp_servers.servers:
        logger.debug("No MCP servers configured, skipping metadata discovery")
        return

    for server in config.mcp_servers.servers:
        try:
            await _discover_server_tools(server.name, server.url, server.headers)
        except Exception as e:
            logger.warning(
                "Failed to discover metadata from MCP server '%s': %s",
                server.name,
                e,
            )


# Backward compatibility alias
async def discover_tool_ui_metadata() -> None:
    """Discover and register metadata for all configured MCP servers (backward compatibility)."""
    await discover_tool_metadata()


async def _discover_server_tools(
    server_name: str,
    server_url: str,
    headers: Optional[dict[str, str]] = None,
) -> None:
    """Discover tools with metadata from a single MCP server.

    Args:
        server_name: The name of the MCP server.
        server_url: The URL of the MCP server.
        headers: Optional authentication headers.
    """
    logger.debug(
        "Discovering metadata from MCP server '%s' at %s", server_name, server_url
    )

    resolved_headers = {}
    if headers:
        for key, value in headers.items():
            if value.startswith("/"):
                try:
                    with open(value) as f:
                        resolved_headers[key] = f.read().strip()
                except Exception:
                    resolved_headers[key] = value
            elif value in ("kubernetes", "client"):
                continue
            else:
                resolved_headers[key] = value

    try:
        async with streamablehttp_client(
            url=server_url,
            headers=resolved_headers,
            timeout=30,
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                tools_result = await session.list_tools()

                for tool in tools_result.tools:
                    # MCP SDK uses 'meta' as Python attr (aliased from '_meta' in JSON)
                    meta = getattr(tool, "meta", None) or {}

                    # Register all tools with metadata if they have any
                    if meta and isinstance(meta, dict):
                        register_tool_metadata(
                            tool.name,
                            server_name,
                            meta=meta,
                        )
                    else:
                        logger.debug(
                            "Tool '%s' has no metadata (meta=%s)",
                            tool.name,
                            meta,
                        )

    except Exception as e:
        logger.warning("Error connecting to MCP server '%s': %s", server_name, e)
        raise
