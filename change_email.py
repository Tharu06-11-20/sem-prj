
import sqlite3

conn = sqlite3.connect("marinevision.db")
cursor = conn.cursor()

username = input("Enter the existing username: ")
new_email = input("Enter your real email address: ")

cursor.execute(
    "SELECT id, username, role, email FROM users WHERE username = ?",
    (username,)
)
user = cursor.fetchone()

if user:
    cursor.execute(
        "UPDATE users SET email = ? WHERE username = ?",
        (new_email, username)
    )
    conn.commit()
    print("Email updated successfully!")
    print("Username:", user[1])
    print("Role:", user[2])
    print("New email:", new_email)
else:
    print("Username not found.")

conn.close()
