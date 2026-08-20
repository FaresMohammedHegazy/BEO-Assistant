import sqlite3
import os
from langgraph.checkpoint.sqlite import SqliteSaver 

def setup_database():
    # Ensure the db folder exists and create the SQLite file
    db_path = os.path.join(os.path.dirname(__file__), 'aurelia.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Rooms Table (Triggers Defensive Design)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rooms (
            room_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            max_capacity INTEGER NOT NULL,
            fire_code_status TEXT NOT NULL
        )
    ''')

    # 2. Guests Table (Triggers Sampling for Allergies)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS guests (
            guest_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            vip_status BOOLEAN NOT NULL,
            dietary_restrictions TEXT
        )
    ''')

    # 3. Events Table (Triggers Elicitation for Deposit)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            guest_id TEXT,
            room_id TEXT,
            status TEXT NOT NULL,
            headcount INTEGER NOT NULL,
            deposit_required REAL,
            FOREIGN KEY(guest_id) REFERENCES guests(guest_id),
            FOREIGN KEY(room_id) REFERENCES rooms(room_id)
        )
    ''')

    # 4. Ingredients Table (Raw facts for the Sampling tool)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS safe_ingredients (
            ingredient_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            is_nut_free BOOLEAN NOT NULL,
            is_vegan BOOLEAN NOT NULL,
            stock_quantity INTEGER NOT NULL DEFAULT 0
        )
    ''')

    # 5. Agent Tools Table (For Dynamic Tool Assignment)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agent_tools (
            agent_name TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT 1,
            PRIMARY KEY (agent_name, tool_name)
        )
    ''')

    # 6. Admin Tickets Table (For HITL and Failure Recovery)
    #
    # status lifecycle:
    #   'open'          -- an unrecoverable node failure (see state_graph/tickets.raise_ticket)
    #   'pending_admin' -- a graph is paused on a HITL node waiting on an admin decision
    #   'resolved'      -- an admin has submitted a decision for a 'pending_admin' ticket
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_tickets (
            ticket_id TEXT PRIMARY KEY,
            graph_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            status TEXT NOT NULL,
            state_snapshot TEXT,
            error_message TEXT,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            decision TEXT,
            decision_payload TEXT,
            created_at TEXT,
            resolved_at TEXT
        )
    ''')

    # Backfill columns for an aurelia.db created before this change, since
    # `CREATE TABLE IF NOT EXISTS` above is a no-op against an existing table.
    existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(admin_tickets)")}
    for col_name, alter_sql in [
        ("checkpoint_ns", "ALTER TABLE admin_tickets ADD COLUMN checkpoint_ns TEXT NOT NULL DEFAULT ''"),
        ("decision", "ALTER TABLE admin_tickets ADD COLUMN decision TEXT"),
        ("decision_payload", "ALTER TABLE admin_tickets ADD COLUMN decision_payload TEXT"),
        ("created_at", "ALTER TABLE admin_tickets ADD COLUMN created_at TEXT"),
        ("resolved_at", "ALTER TABLE admin_tickets ADD COLUMN resolved_at TEXT"),
    ]:
        if col_name not in existing_cols:
            cursor.execute(alter_sql)


    # --- SEED DATA (THE TRAPS) ---

    # Trap 1: The Ballroom capped at 300 (For Defensive Design)
    cursor.execute('''
        INSERT OR IGNORE INTO rooms (room_id, name, max_capacity, fire_code_status) 
        VALUES ('ROOM_101', 'Grand Magnolia Ballroom', 300, 'STRICT_ENFORCEMENT')
    ''')
    # Extra rooms for the "Progress Tracking" batch lookup tool
    for i in range(2, 151):
        cursor.execute(f'''
            INSERT OR IGNORE INTO rooms (room_id, name, max_capacity, fire_code_status) 
            VALUES ('ROOM_{100+i}', 'Standard Conference Room {i}', 50, 'COMPLIANT')
        ''')

    # Trap 2: The VIP Guest with severe allergies (For Sampling)
    cursor.execute('''
        INSERT OR IGNORE INTO guests (guest_id, name, vip_status, dietary_restrictions) 
        VALUES ('GUEST_VIP_1', 'Eleanor Vance', 1, 'SEVERE NUT ALLERGY, VEGAN')
    ''')

    # Trap 3: The Event requiring a $20,000 deposit (For Elicitation)
    cursor.execute('''
        INSERT OR IGNORE INTO events (event_id, guest_id, room_id, status, headcount, deposit_required) 
        VALUES ('EVT_999', 'GUEST_VIP_1', 'ROOM_101', 'PENDING_DEPOSIT', 250, 20000.00)
    ''')

    # Seed Ingredients for the LLM to reason over during Sampling
    ingredients = [
        ('ING_1', 'Quinoa', 1, 1),
        ('ING_2', 'Almonds', 0, 1), # Contains nuts
        ('ING_3', 'Macadamia Oil', 0, 1), # Contains nuts
        ('ING_4', 'Roasted Chickpeas', 1, 1),
        ('ING_5', 'Tofu', 1, 1),
        ('ING_6', 'Butter', 1, 0) # Not vegan
    ]
    cursor.executemany('''
        INSERT OR IGNORE INTO safe_ingredients (ingredient_id, name, is_nut_free, is_vegan) 
        VALUES (?, ?, ?, ?)
    ''', ingredients)


    # --- SEED AGENT TOOLS (DYNAMIC ASSIGNMENTS) ---
    # Assigns the existing MCP server tools to specific agents.
    agent_tools_seed = [
        ('planning_agent', 'audit_chain_wide_availability', 1),
        ('planning_agent', 'book_event_room', 1),
        ('planning_agent', 'view_event_deposit_status', 1),
        ('memory_agent', 'draft_custom_menu', 1),
        ('state_graph_agent', 'confirm_event_booking', 1),
        ('state_graph_agent', 'authenticate_director', 1),
        ('vip_dietary_agent', 'check_ingredient_stock', 1)
    ]
    cursor.executemany('''
        INSERT OR IGNORE INTO agent_tools (agent_name, tool_name, is_active) 
        VALUES (?, ?, ?)
    ''', agent_tools_seed)


    conn.commit()
    conn.close()

    # LangGraph Checkpoint Tables 
    # Creates the checkpoints / checkpoint_blobs / checkpoint_writes tables
    # that every state graph agent will use, inside the SAME aurelia.db
    # file — not a separate database — using the official
    # langgraph-checkpoint-sqlite library instead of hand-rolled SQL.
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        checkpointer.setup()

    print(f"Database successfully created and seeded at {db_path}")
    print("LangGraph checkpoint tables initialized in the same database.")

if __name__ == "__main__":
    setup_database()