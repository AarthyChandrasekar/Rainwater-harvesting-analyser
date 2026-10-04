import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

if os.environ.get("VERCEL"):
    DATABASE = "/tmp/database.db"
else:
    DATABASE = "database.db"

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def create_user(name, email, phone, password):
    conn = get_db_connection()

    hashed_password = generate_password_hash(password)

    try:
        conn.execute(
            """
            INSERT INTO users (name, email, phone, password)
            VALUES (?, ?, ?, ?)
            """,
            (name, email, phone, hashed_password)
        )

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()


def verify_user(email, password):
    conn = get_db_connection()

    user = conn.execute(
        "SELECT * FROM users WHERE email = ?",
        (email,)
    ).fetchone()

    conn.close()

    if user and check_password_hash(user["password"], password):
        return user

    return None