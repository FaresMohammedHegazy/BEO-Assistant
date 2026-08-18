"""
Shared LangGraph checkpointer for every state graph agent 
"""
import os
from contextlib import asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(REPO_ROOT, "db", "aurelia.db")


@asynccontextmanager
async def get_checkpointer(db_path: str | None = None):
    """Yield an AsyncSqliteSaver connected to the shared Aurelia database.

    Every state graph agent should wrap its `.compile(checkpointer=...)`
    call in this context manager rather than opening its own connection,
    so all three graphs checkpoint to the same aurelia.db that
    mcp_server/ and db/setup_db.py already manage.

    Args:
        db_path: Optional override, used by tests to point at a throwaway
            SQLite file instead of the real db/aurelia.db.
    """
    resolved_path = db_path or DEFAULT_DB_PATH
    async with AsyncSqliteSaver.from_conn_string(resolved_path) as checkpointer:
        yield checkpointer