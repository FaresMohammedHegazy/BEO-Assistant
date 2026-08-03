import sqlite3
import os

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
            is_vegan BOOLEAN NOT NULL
        )
    ''')

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

    conn.commit()
    conn.close()
    print(f"Database successfully created and seeded at {db_path}")

if __name__ == "__main__":
    setup_database()