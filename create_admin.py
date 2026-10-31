
import os
import sqlite3
from werkzeug.security import generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "marinevision.db")

name = input("Enter admin name: ").strip()
email = input("Enter admin email: ").strip().lower()
username = input("Create admin username: ").strip()
password = input("Create admin password (minimum 8 characters): ")

if not name or not email or not username:
    print("Name, email, and username are required.")
elif len(password) < 8:
    print("Password must contain at least 8 characters.")
else:
    connection = sqlite3.connect(DATABASE)

    try:
        connection.execute("""
            INSERT INTO users (name, email, username, password, role)
            VALUES (?, ?, ?, ?, 'admin')
        """, (
            name,
            email,
            username,
            generate_password_hash(password)
        ))

        connection.commit()
        print("Admin account created successfully.")

    except sqlite3.IntegrityError:
        print("That username or email already exists.")

    finally:
        connection.close()
