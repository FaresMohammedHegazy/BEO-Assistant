import datetime
import os
import sqlite3
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(REPO_ROOT, "db", "aurelia.db")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def raise_ticket(graph_id: str, thread_id: str, error_message: str,
                  state_snapshot: str, db_path: str | None = None) -> str:
    """Open an 'open' ticket for an unrecoverable node failure.

    This is unchanged for the failure-recovery path -- see open_hitl_ticket
    below for the HITL pending_admin path.
    """
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


def open_hitl_ticket(graph_id: str, thread_id: str, reason: str,
                      state_snapshot: str, checkpoint_ns: str = "",
                      db_path: str | None = None) -> str:
    """Idempotently open a 'pending_admin' ticket for a graph pausing on a
    HITL node, and return its ticket_id.

    LangGraph replays a node's function body on resume when it uses the
    dynamic `interrupt()` primitive, so a node that opens a ticket right
    before interrupting can end up calling this more than once for the same
    pause. To stay safe under that replay (and under any accidental
    double-invocation from a static `interrupt_before` pause), this looks
    for an existing 'pending_admin' ticket for the same (graph_id,
    thread_id) first and reuses it instead of inserting a duplicate.
    """
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT ticket_id FROM admin_tickets
        WHERE graph_id = ? AND thread_id = ? AND status = 'pending_admin'
        """,
        (graph_id, thread_id),
    )
    existing = cursor.fetchone()
    if existing is not None:
        conn.close()
        return existing["ticket_id"]

    ticket_id = f"TICKET_{uuid.uuid4().hex[:8]}"
    cursor.execute(
        """
        INSERT INTO admin_tickets
            (ticket_id, graph_id, thread_id, status, state_snapshot,
             error_message, checkpoint_ns, created_at)
        VALUES (?, ?, ?, 'pending_admin', ?, ?, ?, ?)
        """,
        (ticket_id, graph_id, thread_id, state_snapshot, reason, checkpoint_ns, _now()),
    )
    conn.commit()
    conn.close()
    return ticket_id


def get_ticket(ticket_id: str, db_path: str | None = None) -> dict | None:
    """Fetch a single ticket by id, or None if it doesn't exist."""
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admin_tickets WHERE ticket_id = ?", (ticket_id,))
    row = cursor.fetchone()
    conn.close()
    return _row_to_dict(row)


def list_tickets(status: str | None = None, db_path: str | None = None) -> list[dict]:
    """List tickets, optionally filtered by status (e.g. 'pending_admin')."""
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if status is not None:
        cursor.execute(
            "SELECT * FROM admin_tickets WHERE status = ? ORDER BY created_at DESC",
            (status,),
        )
    else:
        cursor.execute("SELECT * FROM admin_tickets ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def resolve_ticket(ticket_id: str, decision: str, decision_payload: str | None = None,
                    db_path: str | None = None) -> dict | None:
    """Mark a 'pending_admin' ticket 'resolved' and record the admin's
    decision. Returns the updated ticket row, or None if it doesn't exist.
    """
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE admin_tickets
        SET status = 'resolved', decision = ?, decision_payload = ?, resolved_at = ?
        WHERE ticket_id = ?
        """,
        (decision, decision_payload, _now(), ticket_id),
    )
    conn.commit()
    cursor.execute("SELECT * FROM admin_tickets WHERE ticket_id = ?", (ticket_id,))
    row = cursor.fetchone()
    conn.close()
    return _row_to_dict(row)