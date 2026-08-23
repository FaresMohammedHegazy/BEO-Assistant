import os
from contextlib import asynccontextmanager

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@asynccontextmanager
async def open_mcp_session():
    """Yield a live ClientSession connected to the real mcp_server, over
    the shared Streamable HTTP server (see mcp_server/server.py's
    MCP_TRANSPORT=http mode) rather than spawning a private stdio
    subprocess per call."""
    mcp_url = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8765/mcp")
    async with streamable_http_client(mcp_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session