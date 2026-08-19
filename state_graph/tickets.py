import os
import sqlite3
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(REPO_ROOT, "db", "aurelia.db")


def raise_ticket(graph_id: str, thread_id: str, error_message: str,
                  state_snapshot: str, db_path: str | None = None) -> str:
    ticket_id = f"TICKET_{uuid.uuid4().hex[:8]}"
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO admin_tickets
            (ticket_id, graph_id, thread_id, status, state_snapshot, error_message)
        VALUES (?, ?, ?, 'open', ?, ?)
        """,
        (ticket_id, graph_id, thread_id, state_snapshot, error_message),
    )
    conn.commit()
    conn.close()
    return ticket_id