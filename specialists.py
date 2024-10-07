import sqlite3

# Database setup function
def setup_database():
    conn = sqlite3.connect('specialists.db')
    cursor = conn.cursor()

    # Create a table for medical specialists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS specialists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            specialty TEXT NOT NULL,
            phone TEXT NOT NULL,
            availability TEXT
        )
    ''')

    # Insert sample data into the table (run this only if needed)
    specialists = [
    ('Dr. Mohammed', 'Cardiology', '+212611062446', 'Available'),
    ('Dr. Malak', 'Neurology', '+212611062446', 'Available'),
    ]

    cursor.executemany('''
        INSERT INTO specialists (name, specialty, phone, availability)
        VALUES (?, ?, ?, ?)
    ''', specialists)

    conn.commit()
    conn.close()
    print("Database setup completed successfully.")
setup_database()