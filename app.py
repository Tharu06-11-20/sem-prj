import os
import re
import io
import textwrap
import uuid
import sqlite3
import threading
import gc
import time
from datetime import datetime, timedelta
import smtplib
import ssl
from email.message import EmailMessage
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

# Configure environment BEFORE importing PyTorch / Ultralytics / OpenCV
# Render free tier provides 512MB RAM and fractional CPU.
# Forcing single-thread CPU execution prevents thread stack exhaustion (saving ~250MB RAM)
# and eliminates CPU lock contention across multi-core container hosts.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["YOLO_VERBOSE"] = "False"

import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.set_grad_enabled(False)

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, render_template_string, send_file, abort
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
PIL_IMAGE_OPEN = Image.open
import numpy as np
from ultralytics import YOLO


app = Flask(__name__)
app.secret_key = os.environ.get(
    "MARINEVISION_SECRET_KEY",
    "change_this_secret_before_deployment"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "marinevision.db")

MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
POLLUTION_MODEL_PATH = os.path.join(BASE_DIR, "model", "pollution_best.pt")
POLLUTION_RESULT_FOLDER = os.path.join(BASE_DIR, "static", "pollution_results")

# Predictions below this confidence are excluded by YOLO.
MODEL_CONFIDENCE_THRESHOLD = 0.25
UNUSUAL_CONFIDENCE_THRESHOLD = 0.45
MAX_IMAGE_DIMENSION = 1280

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
PLANS = {1: 500, 3: 1200, 6: 2200, 12: 4000}
PAYMENT_METHODS = ["UPI", "Card", "Net Banking"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(POLLUTION_RESULT_FOLDER, exist_ok=True)

# Thread-safe model managers with single-load per worker and warmup
_model_lock = threading.Lock()
_organism_model = None
_pollution_model = None


def get_organism_model():
    """Return the marine organism YOLO model, loading and warming it up if needed."""
    global _organism_model, model
    if _organism_model is None:
        with _model_lock:
            if _organism_model is None:
                if not os.path.isfile(MODEL_PATH):
                    app.logger.error("Organism model file not found at: %s", MODEL_PATH)
                    return None
                app.logger.info("Loading marine organism model from: %s", MODEL_PATH)
                loaded = YOLO(MODEL_PATH)
                # Warm up predictor to avoid cold-start timeout/OOM during user requests
                try:
                    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                    _ = loaded.predict(source=dummy, imgsz=320, device="cpu", verbose=False)
                    del dummy
                except Exception as warmup_err:
                    app.logger.warning("Organism model warmup note: %s", warmup_err)
                gc.collect()
                _organism_model = loaded
                model = _organism_model
                app.logger.info("Organism model ready.")
    return _organism_model


def get_pollution_model():
    """Return the marine pollution YOLO model, loading and warming it up if needed."""
    global _pollution_model, pollution_model
    if _pollution_model is None:
        with _model_lock:
            if _pollution_model is None:
                if not os.path.isfile(POLLUTION_MODEL_PATH):
                    app.logger.warning("Pollution model file not found at: %s", POLLUTION_MODEL_PATH)
                    return None
                app.logger.info("Loading marine pollution model from: %s", POLLUTION_MODEL_PATH)
                loaded = YOLO(POLLUTION_MODEL_PATH)
                try:
                    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                    _ = loaded.predict(source=dummy, imgsz=320, device="cpu", verbose=False)
                    del dummy
                except Exception as warmup_err:
                    app.logger.warning("Pollution model warmup note: %s", warmup_err)
                gc.collect()
                _pollution_model = loaded
                pollution_model = _pollution_model
                app.logger.info("Pollution model ready.")
    return _pollution_model


# Initialize module-level references and preload models on startup
model = None
pollution_model = None

try:
    model = get_organism_model()
except Exception as init_err:
    app.logger.warning("Initial organism model preload deferred: %s", init_err)

try:
    pollution_model = get_pollution_model()
except Exception as init_err:
    app.logger.warning("Initial pollution model preload deferred: %s", init_err)


def preprocess_and_save_upload(file_storage, target_path, max_dim=MAX_IMAGE_DIMENSION):
    """
    Saves and validates an uploaded image file safely.
    - Verifies file integrity using Pillow.
    - Transposes EXIF orientation.
    - Converts RGBA, LA, or Palette modes to RGB (handling transparency with white bg).
    - If either dimension exceeds max_dim, resizes proportionally using high-quality downsampling.
    - Saves clean JPEG to target_path with quality=90.
    Returns (True, None) on success or (False, error_reason) on failure.
    """
    temp_path = target_path + ".tmp"
    try:
        file_storage.save(temp_path)
        with Image.open(temp_path) as raw:
            raw.verify()

        with Image.open(temp_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                img = img.convert("RGBA")
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

            if max(img.size) > max_dim:
                app.logger.info("Downsampling large image from %s to max_dim %d", img.size, max_dim)
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            img.save(target_path, format="JPEG", quality=90)
        return True, None
    except Exception as err:
        app.logger.warning("Image validation/preprocessing failed: %s", err)
        return False, str(err)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def get_db_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def create_table():
    connection = get_db_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    # Existing databases are upgraded without deleting current accounts.
    user_columns = {row["name"] for row in connection.execute("PRAGMA table_info(users)").fetchall()}
    if "approval_status" not in user_columns:
        connection.execute("ALTER TABLE users ADD COLUMN approval_status TEXT NOT NULL DEFAULT 'Approved'")
    if "organization_id" not in user_columns:
        connection.execute("ALTER TABLE users ADD COLUMN organization_id INTEGER")
    if "created_at" not in user_columns:
        connection.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
    if "last_login_at" not in user_columns:
        connection.execute("ALTER TABLE users ADD COLUMN last_login_at TEXT")
    # Optional professional profile fields; NULL keeps legacy accounts valid.
    for column in ("contact_number", "address", "profession", "profession_other",
                   "designation", "highest_qualification", "areas_of_interest",
                   "organization_name"):
        if column not in user_columns:
            connection.execute(f"ALTER TABLE users ADD COLUMN {column} TEXT")

    connection.execute("""
        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            organization_name TEXT NOT NULL,
            duration_months INTEGER NOT NULL,
            amount REAL NOT NULL,
            payment_status TEXT NOT NULL DEFAULT 'Pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    subscription_columns = {row["name"] for row in connection.execute("PRAGMA table_info(subscriptions)").fetchall()}
    if "expires_at" not in subscription_columns:
        connection.execute("ALTER TABLE subscriptions ADD COLUMN expires_at TEXT")
    if "download_limit" not in subscription_columns:
        connection.execute("ALTER TABLE subscriptions ADD COLUMN download_limit INTEGER NOT NULL DEFAULT 100")
    if "is_blocked" not in subscription_columns:
        connection.execute("ALTER TABLE subscriptions ADD COLUMN is_blocked INTEGER NOT NULL DEFAULT 0")
    # Preserve existing subscriptions and derive their expiry date if missing.
    connection.execute("""
        UPDATE subscriptions
        SET expires_at = datetime(created_at, '+' || duration_months || ' months')
        WHERE expires_at IS NULL AND payment_status = 'Successful'
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS download_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organization_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            subscription_id INTEGER NOT NULL,
            record_type TEXT NOT NULL,
            record_id INTEGER NOT NULL,
            downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (organization_id) REFERENCES users(id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (subscription_id) REFERENCES subscriptions(id)
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subscription_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            transaction_id TEXT NOT NULL UNIQUE,
            payment_method TEXT NOT NULL,
            amount REAL NOT NULL,
            payment_status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (subscription_id) REFERENCES subscriptions(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Store each detection and its saved image paths.
    connection.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            original_image TEXT NOT NULL,
            result_image TEXT NOT NULL,
            detected_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Images that need human review if confidence is low.
    connection.execute("""
        CREATE TABLE IF NOT EXISTS unusual_creatures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            predicted_class TEXT,
            confidence REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'Needs review',
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS detection_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL,
            organism_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY (detection_id) REFERENCES detections(id)
        )
    """)

    # Pollution detections are stored separately from organism detections.
    connection.execute("""
        CREATE TABLE IF NOT EXISTS pollution_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            original_image TEXT NOT NULL,
            result_image TEXT NOT NULL,
            detected_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS pollution_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pollution_record_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY (pollution_record_id) REFERENCES pollution_records(id)
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
            message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS organization_announcements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organization_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            posted_by INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT,
            FOREIGN KEY (organization_id) REFERENCES users(id),
            FOREIGN KEY (posted_by) REFERENCES users(id)
        )
    """)

    connection.commit()
    connection.close()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def organization_owner_id(user_id):
    """Return the organization account that owns this user, or None."""
    connection = get_db_connection()
    row = connection.execute(
        "SELECT id, role, organization_id FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    connection.close()
    if not row:
        return None
    if row["role"] == "organization":
        return row["id"]
    if row["role"] == "org_member":
        return row["organization_id"]
    return None


def active_subscription_for_org(organization_id):
    if not organization_id:
        return None
    connection = get_db_connection()
    row = connection.execute("""
        SELECT * FROM subscriptions
        WHERE user_id = ? AND payment_status = 'Successful'
          AND (expires_at IS NULL OR datetime(expires_at) > datetime('now', 'localtime'))
        ORDER BY id DESC LIMIT 1
    """, (organization_id,)).fetchone()
    connection.close()
    return row


def organization_usage(organization_id, subscription=None):
    subscription = subscription or active_subscription_for_org(organization_id)
    if not subscription:
        return {"subscription": None, "used": 0, "limit": 0, "remaining": 0, "blocked": True}
    connection = get_db_connection()
    used = connection.execute(
        "SELECT COUNT(*) FROM download_logs WHERE subscription_id = ?",
        (subscription["id"],)
    ).fetchone()[0]
    connection.close()
    limit = max(0, int(subscription["download_limit"] or 0))
    blocked = bool(subscription["is_blocked"]) or used >= limit
    return {"subscription": subscription, "used": used, "limit": limit,
            "remaining": max(0, limit - used), "blocked": blocked}


def organization_has_access(user_id):
    """Require an active paid subscription and a non-exhausted download quota."""
    owner_id = organization_owner_id(user_id)
    subscription = active_subscription_for_org(owner_id)
    if not subscription:
        return False
    usage = organization_usage(owner_id, subscription)
    return not usage["blocked"]


def require_org_service_access():
    if session.get("role") in ("organization", "org_member"):
        owner_id = organization_owner_id(session.get("user_id"))
        usage = organization_usage(owner_id)
        if not usage["subscription"]:
            flash("Your organization needs an active paid subscription. Please ask the organization owner to renew it.")
            return False
        if usage["subscription"]["is_blocked"]:
            flash("The organization's access was blocked by the administrator. Please contact the organization owner/admin.")
            return False
    return True


# ---------------- HOME ----------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- REGISTRATION ----------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        role = request.form.get("role", "")
        contact_number = request.form.get("contact_number", "").strip()
        address = request.form.get("address", "").strip()
        profession = request.form.get("profession", "").strip()
        profession_other = request.form.get("profession_other", "").strip()
        designation = request.form.get("designation", "").strip()
        highest_qualification = request.form.get("highest_qualification", "").strip()
        areas_of_interest = request.form.get("areas_of_interest", "").strip()
        organization_name = request.form.get("organization_name", "").strip()

        if not all([name, email, username, password]):
            flash("Please fill in all fields.")
            return redirect(url_for("register"))

        if role not in ["developer", "organization"]:
            flash("Please select a valid account type.")
            return redirect(url_for("register"))

        if not contact_number or not address or not profession or not designation:
            flash("Contact number, address, profession and designation are required.")
            return redirect(url_for("register"))
        contact_digits = "".join(character for character in contact_number if character.isdigit())
        if (not re.fullmatch(r"\+?[0-9][0-9\s().-]{5,19}", contact_number)
                or not 7 <= len(contact_digits) <= 15):
            flash("Enter a valid contact number with 7 to 15 digits.")
            return redirect(url_for("register"))
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            flash("Enter a valid email address.")
            return redirect(url_for("register"))
        if profession not in {"Student", "Researcher", "Marine Biologist", "Environmental Scientist", "Fisheries Professional", "Other"}:
            flash("Please select a valid profession.")
            return redirect(url_for("register"))
        if profession == "Other" and not profession_other:
            flash("Please specify your profession.")
            return redirect(url_for("register"))
        if role == "organization" and not organization_name:
            flash("Organization name is required for organization accounts.")
            return redirect(url_for("register"))

        if len(password) < 8:
            flash("Password must contain at least 8 characters.")
            return redirect(url_for("register"))

        connection = get_db_connection()

        try:
            connection.execute("""
                INSERT INTO users (name, email, username, password, role, approval_status,
                    contact_number, address, profession, profession_other, designation,
                    highest_qualification, areas_of_interest, organization_name)
                VALUES (?, ?, ?, ?, ?, 'Pending', ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                email,
                username,
                generate_password_hash(password),
                role, contact_number, address, profession, profession_other, designation,
                highest_qualification or None, areas_of_interest or None,
                organization_name or None
            ))

            connection.commit()
            flash("Registration submitted. Please wait for admin approval before signing in.")
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            flash("That username or email is already registered.")
            return redirect(url_for("register"))

        finally:
            connection.close()

    return render_template("register.html")


# ---------------- LOGIN ----------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        connection = get_db_connection()

        user = connection.execute("""
            SELECT * FROM users WHERE username = ?
        """, (username,)).fetchone()

        connection.close()

        if user and check_password_hash(user["password"], password):
            if user["role"] != "admin" and user["approval_status"] != "Approved":
                session.clear()
                flash("Your account is awaiting admin approval or has been rejected. Please contact the administrator.")
                return redirect(url_for("login"))
            login_connection = get_db_connection()
            login_connection.execute(
                "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?",
                (user["id"],)
            )
            login_connection.commit()
            login_connection.close()
            session.clear()
            session["user_id"] = user["id"]
            session["name"] = user["name"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            if user["role"] == "org_member":
                session["organization_id"] = user["organization_id"]

            if user["role"] == "admin":
                return redirect(url_for("admin_dashboard"))
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.before_request
def enforce_active_organization_member():
    """End existing member sessions after their owner deactivates/removes them."""
    if session.get("role") != "org_member" or request.endpoint in {"static", "login", "logout"}:
        return None
    connection = get_db_connection()
    try:
        member = connection.execute(
            "SELECT role, approval_status FROM users WHERE id = ?",
            (session.get("user_id"),)
        ).fetchone()
    finally:
        connection.close()
    if not member or member["role"] != "org_member" or member["approval_status"] != "Approved":
        session.clear()
        flash("Your team account is inactive. Please contact your organization owner.")
        return redirect(url_for("login"))
    return None


# ---------------- USER FEEDBACK ----------------

FEEDBACK_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Feedback | MarineVision</title>
<style>body{font-family:Arial,sans-serif;background:#eef5f7;color:#18313b;margin:0;padding:24px}.box{max-width:650px;margin:25px auto;background:white;padding:26px;border-radius:12px;box-shadow:0 3px 12px #0000000b}label{display:block;margin:14px 0 7px}select,textarea,button{box-sizing:border-box;width:100%;padding:11px;border:1px solid #ccd5d9;border-radius:6px;font:inherit}textarea{min-height:130px;resize:vertical}button{margin-top:18px;background:#075985;color:white;border:0;cursor:pointer}.msg{padding:10px;background:#e0f2fe;border-radius:6px}.links a{margin-right:15px;color:#075985}</style></head>
<body><div class="box"><h1>Share your feedback</h1><p>Your feedback helps us improve MarineVision.</p>{% for message in get_flashed_messages() %}<p class="msg">{{ message }}</p>{% endfor %}
<form method="post"><label for="rating">Rating</label><select id="rating" name="rating" required><option value="">Choose a rating</option><option value="5">5 - Excellent</option><option value="4">4 - Good</option><option value="3">3 - Average</option><option value="2">2 - Poor</option><option value="1">1 - Very poor</option></select>
<label for="message">Comments or suggestions</label><textarea id="message" name="message" maxlength="2000" required placeholder="Tell us about your experience..."></textarea><button type="submit">Submit Feedback</button></form><p class="links"><a href="{{ url_for('dashboard') }}">Back to dashboard</a><a href="{{ url_for('logout') }}">Logout</a></p></div></body></html>
"""

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if "user_id" not in session or session.get("role") == "admin":
        flash("Please sign in with a user or organization account to submit feedback.")
        return redirect(url_for("login"))

    if request.method == "POST":
        message = request.form.get("message", "").strip()
        try:
            rating = int(request.form.get("rating", ""))
        except (TypeError, ValueError):
            rating = 0

        if rating not in range(1, 6) or not message:
            flash("Please select a rating and enter your feedback.")
            return redirect(url_for("feedback"))
        if len(message) > 2000:
            flash("Feedback must be 2000 characters or fewer.")
            return redirect(url_for("feedback"))

        connection = get_db_connection()
        connection.execute(
            "INSERT INTO feedback (user_id, rating, message) VALUES (?, ?, ?)",
            (session["user_id"], rating, message)
        )
        connection.commit()
        connection.close()
        flash("Thank you! Your feedback has been submitted.")
        return redirect(url_for("feedback"))

    return render_template_string(FEEDBACK_PAGE)



# Add a visible Feedback shortcut to existing user-facing pages without
# requiring edits to the separate HTML templates.
@app.after_request
def add_feedback_shortcut(response):
    try:
        is_user_page = (
            response.status_code == 200
            and response.mimetype == "text/html"
            and "user_id" in session
            and session.get("role") != "admin"
            and request.endpoint != "feedback"
        )
        if is_user_page:
            html = response.get_data(as_text=True)
            shortcut = """<a href="/feedback" style="position:fixed;right:22px;bottom:22px;z-index:99999;background:#087f8c;color:white;padding:13px 18px;border-radius:28px;text-decoration:none;font-family:Arial,sans-serif;font-size:15px;font-weight:bold;box-shadow:0 4px 12px rgba(0,0,0,.25)">Give Feedback</a>"""
            if "</body>" in html:
                html = html.replace("</body>", shortcut + "</body>", 1)
                response.set_data(html)
                response.headers["Content-Length"] = str(len(response.get_data()))
    except Exception:
        # Do not let a feedback shortcut interfere with the main application.
        pass
    return response

# ---------------- DASHBOARD ----------------

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Please sign in first.")
        return redirect(url_for("login"))
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))

    role = session.get("role")
    if role == "organization":
        connection = get_db_connection()
        try:
            owner = connection.execute(
                "SELECT * FROM users WHERE id = ? AND role = 'organization'",
                (session["user_id"],)
            ).fetchone()
            if owner is None:
                session.clear()
                return redirect(url_for("login"))
            subscription = connection.execute(
                "SELECT * FROM subscriptions WHERE user_id = ? ORDER BY id DESC LIMIT 1",
                (owner["id"],)
            ).fetchone()
            members = connection.execute("""
                SELECT u.id, u.name, u.username, u.last_login_at,
                    (SELECT COUNT(*) FROM detections d WHERE d.user_id = u.id) AS organism_count,
                    (SELECT COUNT(*) FROM pollution_records p WHERE p.user_id = u.id) AS pollution_count,
                    (SELECT COUNT(*) FROM download_logs dl
                     WHERE dl.organization_id = ? AND dl.user_id = u.id) AS download_count
                FROM users u
                WHERE u.organization_id = ? AND u.role = 'org_member'
                  AND COALESCE(u.approval_status, 'Approved') != 'Removed'
                ORDER BY (organism_count + pollution_count + download_count) DESC, u.name COLLATE NOCASE
            """, (owner["id"], owner["id"])).fetchall()
            membership = "(d.user_id = ? OR d.user_id IN (SELECT id FROM users WHERE role='org_member' AND organization_id=?))"
            organism_total = connection.execute(
                f"SELECT COUNT(*) FROM detections d WHERE {membership}",
                (owner["id"], owner["id"])
            ).fetchone()[0]
            pollution_total = connection.execute(
                "SELECT COUNT(*) FROM pollution_records p WHERE p.user_id = ? OR p.user_id IN (SELECT id FROM users WHERE role='org_member' AND organization_id=?)",
                (owner["id"], owner["id"])
            ).fetchone()[0]
            announcements = connection.execute("""
                SELECT id, title, message, created_at FROM organization_announcements
                WHERE organization_id = ? ORDER BY datetime(created_at) DESC, id DESC LIMIT 5
            """, (owner["id"],)).fetchall()
            members_total = len(members)
            downloads_used = 0
            download_limit = 0
            if subscription:
                downloads_used = connection.execute(
                    "SELECT COUNT(*) FROM download_logs WHERE organization_id = ? AND subscription_id = ?",
                    (owner["id"], subscription["id"])
                ).fetchone()[0]
                download_limit = max(0, int(subscription["download_limit"] or 0))
            totals = {
                "members": members_total,
                "organism": organism_total,
                "pollution": pollution_total,
                "downloads": downloads_used,
                "remaining": max(0, download_limit - downloads_used),
            }
            most_active = members[0] if members and (members[0]["organism_count"] + members[0]["pollution_count"] + members[0]["download_count"]) else None
            organization_name = (owner["organization_name"] or
                (subscription["organization_name"] if subscription else None) or "Not Provided")
        finally:
            connection.close()
        return render_template(
            "organization_dashboard.html", name=session.get("name", owner["name"]),
            organization_name=organization_name, totals=totals, members=members,
            most_active=most_active, announcements=announcements,
            subscription=subscription
        )

    if role == "org_member":
        connection = get_db_connection()
        try:
            member = connection.execute(
                "SELECT organization_id FROM users WHERE id = ? AND role = 'org_member'",
                (session["user_id"],)
            ).fetchone()
            announcements = connection.execute("""
                SELECT id, title, message, created_at FROM organization_announcements
                WHERE organization_id = ? ORDER BY datetime(created_at) DESC, id DESC LIMIT 10
            """, (member["organization_id"],)).fetchall() if member and member["organization_id"] else []
        finally:
            connection.close()
        return render_template("dashboard.html", name=session["name"], role=role,
                               announcements=announcements)

    subscription_details = None
    recent_payments = []
    recent_detections = []

    connection = get_db_connection()

    recent_detections = connection.execute("""
        SELECT * FROM detections WHERE user_id = ? ORDER BY id DESC LIMIT 5
    """, (session["user_id"],)).fetchall()

    connection.close()

    return render_template(
        "dashboard.html",
        name=session["name"],
        role=session["role"],
        subscription=subscription_details,
        payments=recent_payments,
        detections=recent_detections, announcements=[]
    )


def organization_id_for_session(require_owner=False):
    """Resolve an authenticated account's organization from the database."""
    user_id = session.get("user_id")
    if not user_id:
        return None
    connection = get_db_connection()
    try:
        user = connection.execute(
            "SELECT id, role, organization_id FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not user:
            return None
        if user["role"] == "organization":
            return user["id"]
        if require_owner or user["role"] != "org_member" or not user["organization_id"]:
            return None
        owner = connection.execute(
            "SELECT id FROM users WHERE id = ? AND role = 'organization'",
            (user["organization_id"],)
        ).fetchone()
        return owner["id"] if owner else None
    finally:
        connection.close()


def valid_optional_date(value):
    if not value:
        return True
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def organization_history_rows(connection, organization_id, record_type="all",
                              member_id="", start_date="", end_date=""):
    owner_filter = "(owner_user_id = ? OR owner_user_id IN (SELECT id FROM users WHERE role='org_member' AND organization_id=?))"
    records = []
    if record_type in ("all", "organism"):
        query = f"""
            SELECT d.id, d.user_id, u.name AS member_name, 'organism' AS detection_type,
                   d.original_image, d.result_image, d.detected_count, d.created_at
            FROM detections d JOIN users u ON u.id = d.user_id
            WHERE {owner_filter.replace('owner_user_id', 'd.user_id')}
        """
        detection_params = [organization_id, organization_id]
        if member_id:
            query += " AND d.user_id = ?"
            detection_params.append(int(member_id))
        if start_date:
            query += " AND date(d.created_at) >= date(?)"
            detection_params.append(start_date)
        if end_date:
            query += " AND date(d.created_at) <= date(?)"
            detection_params.append(end_date)
        records.extend(dict(row) for row in connection.execute(query, detection_params).fetchall())

    if record_type in ("all", "pollution"):
        query = f"""
            SELECT p.id, p.user_id, u.name AS member_name, 'pollution' AS detection_type,
                   p.original_image, p.result_image, p.detected_count, p.created_at
            FROM pollution_records p JOIN users u ON u.id = p.user_id
            WHERE (p.user_id = ? OR p.user_id IN
                (SELECT id FROM users WHERE role='org_member' AND organization_id=?))
        """
        pollution_params = [organization_id, organization_id]
        if member_id:
            query += " AND p.user_id = ?"
            pollution_params.append(int(member_id))
        if start_date:
            query += " AND date(p.created_at) >= date(?)"
            pollution_params.append(start_date)
        if end_date:
            query += " AND date(p.created_at) <= date(?)"
            pollution_params.append(end_date)
        records.extend(dict(row) for row in connection.execute(query, pollution_params).fetchall())

    for record in records:
        if record["detection_type"] == "organism":
            items = connection.execute(
                "SELECT organism_name AS class_name, confidence FROM detection_items WHERE detection_id = ? ORDER BY id",
                (record["id"],)
            ).fetchall()
        else:
            items = connection.execute(
                "SELECT class_name, confidence FROM pollution_items WHERE pollution_record_id = ? ORDER BY id",
                (record["id"],)
            ).fetchall()
        record["items"] = [dict(item) for item in items]
    records.sort(key=lambda row: (row["created_at"] or "", row["id"]), reverse=True)
    return records


@app.route("/organization/history")
def organization_history():
    organization_id = organization_id_for_session(require_owner=True)
    if not organization_id:
        abort(403)
    record_type = request.args.get("type", "all")
    if record_type not in {"all", "organism", "pollution"}:
        abort(400)
    member_id = request.args.get("member", "")
    start_date, end_date = request.args.get("start", ""), request.args.get("end", "")
    if not valid_optional_date(start_date) or not valid_optional_date(end_date):
        abort(400)
    connection = get_db_connection()
    try:
        members = connection.execute("""
            SELECT id, name FROM users WHERE organization_id = ? AND role='org_member'
              AND COALESCE(approval_status, 'Approved') != 'Removed' ORDER BY name COLLATE NOCASE
        """, (organization_id,)).fetchall()
        if member_id:
            allowed = {str(row["id"]) for row in members}
            if member_id not in allowed:
                abort(403)
        rows = organization_history_rows(
            connection, organization_id, record_type, member_id, start_date, end_date
        )
        return render_template("organization_history.html", records=rows, members=members,
                               selected_type=record_type, selected_member=member_id,
                               start_date=start_date, end_date=end_date)
    finally:
        connection.close()


@app.route("/organization/announcements", methods=["GET", "POST"])
def organization_announcements():
    organization_id = organization_id_for_session()
    if not organization_id:
        abort(403)
    is_owner = organization_id == session.get("user_id")
    connection = get_db_connection()
    try:
        if request.method == "POST":
            if not is_owner:
                abort(403)
            action = request.form.get("action", "create")
            announcement_id = request.form.get("announcement_id", type=int)
            if action == "delete":
                cursor = connection.execute(
                    "DELETE FROM organization_announcements WHERE id = ? AND organization_id = ?",
                    (announcement_id, organization_id)
                )
                connection.commit()
                flash("Announcement deleted." if cursor.rowcount else "Announcement not found.")
            elif action in {"create", "edit"}:
                title = request.form.get("title", "").strip()
                message = request.form.get("message", "").strip()
                if not title or not message or len(title) > 160 or len(message) > 5000:
                    flash("Enter a title (up to 160 characters) and message (up to 5000 characters).")
                elif action == "edit":
                    cursor = connection.execute("""
                        UPDATE organization_announcements SET title = ?, message = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ? AND organization_id = ?
                    """, (title, message, announcement_id, organization_id))
                    connection.commit()
                    flash("Announcement updated." if cursor.rowcount else "Announcement not found.")
                else:
                    connection.execute("""
                        INSERT INTO organization_announcements (organization_id, title, message, posted_by)
                        VALUES (?, ?, ?, ?)
                    """, (organization_id, title, message, session["user_id"]))
                    connection.commit()
                    flash("Announcement posted to your organization.")
            else:
                abort(400)
        announcements = connection.execute("""
            SELECT a.id, a.title, a.message, a.created_at, a.updated_at, u.name AS poster_name
            FROM organization_announcements a JOIN users u ON u.id = a.posted_by
            WHERE a.organization_id = ? ORDER BY datetime(a.created_at) DESC, a.id DESC
        """, (organization_id,)).fetchall()
        return render_template("organization_announcements.html", announcements=announcements,
                               is_owner=is_owner)
    finally:
        connection.close()


@app.route("/organization/reports", methods=["GET", "POST"])
def organization_report():
    organization_id = organization_id_for_session(require_owner=True)
    if not organization_id:
        abort(403)
    if request.method == "GET":
        return render_template("organization_report.html")
    start_date = request.form.get("start_date", "").strip()
    end_date = request.form.get("end_date", "").strip()
    if not valid_optional_date(start_date) or not valid_optional_date(end_date):
        abort(400)
    if start_date and end_date and start_date > end_date:
        flash("Start date must be on or before end date.")
        return redirect(url_for("organization_report"))
    return send_file(io.BytesIO(build_organization_pdf(organization_id, start_date, end_date)),
                     mimetype="application/pdf", as_attachment=True,
                     download_name="marinevision_organization_report.pdf")


def build_organization_pdf(organization_id, start_date="", end_date=""):
    connection = get_db_connection()
    try:
        owner = connection.execute(
            "SELECT name, organization_name FROM users WHERE id = ? AND role='organization'",
            (organization_id,)
        ).fetchone()
        if not owner:
            abort(403)
        subscription = connection.execute(
            "SELECT * FROM subscriptions WHERE user_id = ? ORDER BY id DESC LIMIT 1",
            (organization_id,)
        ).fetchone()
        organization_name = (owner["organization_name"] or
            (subscription["organization_name"] if subscription else None) or "Not Provided")
        records = organization_history_rows(connection, organization_id,
                                            start_date=start_date, end_date=end_date)
        organism_records = [row for row in records if row["detection_type"] == "organism"]
        pollution_records = [row for row in records if row["detection_type"] == "pollution"]
        class_counts = {"organism": {}, "pollution": {}}
        member_counts = {
            row["id"]: {"name": row["name"], "organism": 0, "pollution": 0, "downloads": 0}
            for row in connection.execute("""
                SELECT id, name FROM users WHERE organization_id = ? AND role = 'org_member'
                  AND COALESCE(approval_status, 'Approved') != 'Removed'
            """, (organization_id,)).fetchall()
        }
        for row in records:
            member = member_counts.setdefault(row["user_id"], {
                "name": row["member_name"], "organism": 0, "pollution": 0, "downloads": 0
            })
            member[row["detection_type"]] += 1
            for item in row["items"]:
                names = class_counts[row["detection_type"]]
                names[item["class_name"]] = names.get(item["class_name"], 0) + 1
        download_sql = """
            SELECT dl.user_id, u.name, COUNT(*) AS downloads
            FROM download_logs dl JOIN users u ON u.id = dl.user_id
            WHERE dl.organization_id = ?
              AND (dl.user_id = ? OR dl.user_id IN
                   (SELECT id FROM users WHERE role='org_member' AND organization_id=?))
        """
        download_params = [organization_id, organization_id, organization_id]
        if start_date:
            download_sql += " AND date(dl.downloaded_at) >= date(?)"
            download_params.append(start_date)
        if end_date:
            download_sql += " AND date(dl.downloaded_at) <= date(?)"
            download_params.append(end_date)
        download_sql += " GROUP BY dl.user_id, u.name"
        for row in connection.execute(download_sql, download_params):
            member = member_counts.setdefault(row["user_id"], {
                "name": row["name"], "organism": 0, "pollution": 0, "downloads": 0
            })
            member["downloads"] = row["downloads"]
        if subscription:
            downloads_used = connection.execute(
                "SELECT COUNT(*) FROM download_logs WHERE organization_id = ? AND subscription_id = ?",
                (organization_id, subscription["id"])
            ).fetchone()[0]
            quota = max(0, int(subscription["download_limit"] or 0))
        else:
            downloads_used, quota = 0, 0
        remaining = max(0, quota - downloads_used)
        period = f"{start_date or 'Beginning'} through {end_date or 'Present'}"
        lines = [
            ("title", "MarineVision Organization Report"),
            ("normal", f"Organization: {organization_name}"),
            ("normal", f"Organization head: {owner['name']}"),
            ("normal", f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
            ("normal", f"Selected date range: {period}"),
            ("heading", "Detection Summary"),
            ("normal", f"Total organism detections: {len(organism_records)}"),
            ("normal", f"Total pollution detections: {len(pollution_records)}"),
            ("normal", f"Download quota: {quota}    Used: {downloads_used}    Remaining: {remaining}"),
            ("heading", "Organism Class Counts"),
        ]
        lines.extend(("normal", f"{key}: {value}") for key, value in sorted(class_counts["organism"].items()))
        if not class_counts["organism"]:
            lines.append(("normal", "No organism detections in the selected period."))
        lines.append(("heading", "Pollution Class Counts"))
        lines.extend(("normal", f"{key}: {value}") for key, value in sorted(class_counts["pollution"].items()))
        if not class_counts["pollution"]:
            lines.append(("normal", "No pollution detections in the selected period."))
        lines.append(("heading", "Member Statistics"))
        lines.append(("table", f"{'Member':26} | {'Organisms':9} | {'Pollution':9} | {'Downloads':9}"))
        lines.extend(("table", f"{entry['name'][:26]:26} | {entry['organism']:9} | {entry['pollution']:9} | {entry['downloads']:9}")
                     for _, entry in sorted(member_counts.items(), key=lambda pair: pair[1]["name"].lower()))
        if not member_counts:
            lines.append(("normal", "No member activity in the selected period."))
        return create_simple_pdf(lines)
    finally:
        connection.close()


def create_simple_pdf(lines):
    """Build a small paginated text-and-table PDF without third-party packages."""
    pages, current = [], []
    for style, text in lines:
        width = 84 if style == "table" else 92
        wrapped = textwrap.wrap(str(text), width=width, break_long_words=True) or [""]
        for segment in wrapped:
            if len(current) >= 42:
                pages.append(current)
                current = []
            current.append((style, segment))
    if current or not pages:
        pages.append(current)

    objects = [None]
    def add_object(data):
        objects.append(data)
        return len(objects) - 1

    catalog_id = add_object(b"")
    pages_id = add_object(b"")
    font_regular_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_bold_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    font_table_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
    page_ids = []
    for page_lines in pages:
        commands = ["BT"]
        y = 748
        for style, text in page_lines:
            font = "/F2" if style in {"title", "heading"} else ("/F3" if style == "table" else "/F1")
            size = 16 if style == "title" else (12 if style == "heading" else 8 if style == "table" else 10)
            if style == "heading":
                y -= 7
            if style == "table":
                commands.append(f"0.4 w 48 {y - 4} m 564 {y - 4} l S")
            safe_text = text.encode("latin-1", "replace").decode("latin-1")
            safe_text = safe_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.extend([f"{font} {size} Tf", f"1 0 0 1 48 {y} Tm", f"({safe_text}) Tj"])
            y -= 15
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1")
        content_id = add_object(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        page_id = add_object(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] ".encode() +
            f"/Resources << /Font << /F1 {font_regular_id} 0 R /F2 {font_bold_id} 0 R /F3 {font_table_id} 0 R >> >> ".encode() +
            f"/Contents {content_id} 0 R >>".encode()
        )
        page_ids.append(page_id)
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[catalog_id] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode()
    objects[pages_id] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()

    output = bytearray(b"%PDF-1.4\n% MarineVision report\n")
    offsets = [0]
    for object_id in range(1, len(objects)):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode())
        output.extend(objects[object_id])
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(f"trailer\n<< /Size {len(objects)} /Root {catalog_id} 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode())
    return bytes(output)


@app.route("/analytics")
def personal_analytics():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("role") != "developer":
        abort(403)
    connection = get_db_connection()
    try:
        organism = connection.execute(
            "SELECT COUNT(*) FROM detections WHERE user_id = ?", (session["user_id"],)
        ).fetchone()[0]
        pollution = connection.execute(
            "SELECT COUNT(*) FROM pollution_records WHERE user_id = ?", (session["user_id"],)
        ).fetchone()[0]
        return render_template("personal_analytics.html", organism=organism, pollution=pollution)
    finally:
        connection.close()


# ---------------- SUBSCRIPTION ----------------

@app.route("/subscription", methods=["GET", "POST"])
def subscription():
    if "user_id" not in session:
        flash("Please sign in first.")
        return redirect(url_for("login"))

    if session.get("role") != "organization":
        flash("Only organization accounts can manage subscriptions.")
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        organization_name = request.form.get(
            "organization_name", ""
        ).strip()

        duration_text = request.form.get("duration", "")

        if not organization_name:
            flash("Please enter your organization name.")
            return redirect(url_for("subscription"))

        try:
            duration = int(duration_text)
        except (ValueError, TypeError):
            flash("Please select a valid subscription duration.")
            return redirect(url_for("subscription"))

        if duration not in PLANS:
            flash("Please select a valid subscription plan.")
            return redirect(url_for("subscription"))

        connection = get_db_connection()

        try:
            cursor = connection.execute("""
                INSERT INTO subscriptions
                (user_id, organization_name, duration_months,
                 amount, payment_status)
                VALUES (?, ?, ?, ?, 'Pending')
            """, (
                session["user_id"],
                organization_name,
                duration,
                PLANS[duration]
            ))

            subscription_id = cursor.lastrowid
            connection.commit()

        finally:
            connection.close()

        return redirect(url_for(
            "payment", subscription_id=subscription_id
        ))

    return render_template("subscription.html")


# ---------------- PAYMENT WORKFLOW ----------------

@app.route("/payment/<int:subscription_id>", methods=["GET", "POST"])
def payment(subscription_id):
    if "user_id" not in session:
        flash("Please sign in first.")
        return redirect(url_for("login"))

    if session.get("role") != "organization":
        flash("Only organization accounts can access payments.")
        return redirect(url_for("dashboard"))

    connection = get_db_connection()

    subscription_record = connection.execute("""
        SELECT * FROM subscriptions
        WHERE id = ? AND user_id = ?
    """, (subscription_id, session["user_id"])).fetchone()

    if subscription_record is None:
        connection.close()
        flash("Subscription not found.")
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        payment_method = request.form.get("payment_method", "")
        payment_action = request.form.get("payment_action", "")

        if payment_method not in PAYMENT_METHODS:
            connection.close()
            flash("Please select a valid payment method.")
            return redirect(url_for(
                "payment", subscription_id=subscription_id
            ))

        if payment_action not in ["success", "failure"]:
            connection.close()
            flash("Invalid payment request.")
            return redirect(url_for(
                "payment", subscription_id=subscription_id
            ))

        if subscription_record["payment_status"] == "Successful":
            connection.close()
            flash("This subscription is already marked as successful.")
            return redirect(url_for("dashboard"))

        transaction_id = "MV-" + uuid.uuid4().hex[:12].upper()

        status = (
            "Successful" if payment_action == "success" else "Failed"
        )

        connection.execute("""
            INSERT INTO payments
            (subscription_id, user_id, transaction_id,
             payment_method, amount, payment_status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            subscription_id,
            session["user_id"],
            transaction_id,
            payment_method,
            subscription_record["amount"],
            status
        ))

        if status == "Successful":
            duration_months = int(subscription_record["duration_months"])
            expires_at = (datetime.now() + timedelta(days=duration_months * 30)).strftime("%Y-%m-%d %H:%M:%S")
            # Default download allowances; the administrator can edit the limit.
            default_limits = {1: 100, 3: 350, 6: 800, 12: 2000}
            download_limit = default_limits.get(duration_months, 100)
            connection.execute("""
                UPDATE subscriptions
                SET payment_status = ?, expires_at = ?, download_limit = ?, is_blocked = 0
                WHERE id = ? AND user_id = ?
            """, (status, expires_at, download_limit, subscription_id, session["user_id"]))
        else:
            connection.execute("""
                UPDATE subscriptions SET payment_status = ?
                WHERE id = ? AND user_id = ?
            """, (status, subscription_id, session["user_id"]))

        connection.commit()
        connection.close()

        flash(
            f"Payment workflow status: {status}. "
            f"Transaction reference: {transaction_id}."
        )
        return redirect(url_for("dashboard"))

    connection.close()

    return render_template(
        "payment.html",
        subscription=subscription_record
    )


# ---------------- MARINE ORGANISM DETECTION ----------------

@app.route("/detect", methods=["GET", "POST"])
def detect():
    if "user_id" not in session:
        flash("Please sign in first.")
        return redirect(url_for("login"))

    if session.get("role") in ("organization", "org_member"):
        if not require_org_service_access():
            if session.get("role") == "organization":
                return redirect(url_for("subscription"))
            return redirect(url_for("dashboard"))

    if request.method == "POST":
        if "image" not in request.files:
            flash("Please select an image.")
            return redirect(url_for("detect"))

        file = request.files["image"]

        if not file or file.filename == "":
            flash("Please select an image.")
            return redirect(url_for("detect"))

        if not allowed_file(file.filename):
            flash("Upload a JPG, JPEG, PNG, or WEBP image.")
            return redirect(url_for("detect"))

        current_model = get_organism_model()
        if current_model is None:
            flash("Marine organism detection model is currently unavailable. Please check the server logs.")
            return redirect(url_for("detect"))

        # Generate unique names so uploaded files are not overwritten.
        unique_name = uuid.uuid4().hex
        original_filename = f"{unique_name}.jpg"
        result_filename = f"{unique_name}_result.jpg"

        image_path = os.path.join(UPLOAD_FOLDER, original_filename)
        result_path = os.path.join(RESULT_FOLDER, result_filename)

        connection = None
        t_start = time.time()

        # Validate and preprocess image (resizes large dimensions, handles orientation & RGBA)
        ok, err_msg = preprocess_and_save_upload(file, image_path)
        if not ok:
            app.logger.warning("Upload validation failed for %s: %s", file.filename, err_msg)
            flash("The uploaded file is not a readable supported image. Please upload a valid JPG, PNG, or WEBP file.")
            return redirect(url_for("detect"))

        try:
            # Run the trained YOLO model with explicit CPU, imgsz=640 and confidence threshold
            t_infer_start = time.time()
            prediction = current_model.predict(
                source=image_path,
                conf=MODEL_CONFIDENCE_THRESHOLD,
                imgsz=640,
                device="cpu",
                verbose=False
            )[0]
            t_infer = time.time() - t_infer_start

            # Save the annotated image with bounding boxes
            annotated_image = prediction.plot()
            result_img = Image.fromarray(annotated_image[:, :, ::-1])
            result_img.save(result_path, format="JPEG", quality=85, optimize=True)

            # Explicitly free intermediate buffers
            del annotated_image, result_img

            detection_items = []
            for box in prediction.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                detection_items.append({
                    "name": prediction.names[class_id],
                    "confidence": round(confidence * 100, 2)
                })

            del prediction
            gc.collect()

            # Save detection history in the database
            connection = get_db_connection()
            cursor = connection.execute("""
                INSERT INTO detections
                (user_id, original_image, result_image, detected_count)
                VALUES (?, ?, ?, ?)
            """, (
                session["user_id"],
                "uploads/" + original_filename,
                "results/" + result_filename,
                len(detection_items)
            ))
            detection_id = cursor.lastrowid

            for item in detection_items:
                connection.execute("""
                    INSERT INTO detection_items
                    (detection_id, organism_name, confidence)
                    VALUES (?, ?, ?)
                """, (
                    detection_id,
                    item["name"],
                    item["confidence"]
                ))

            # Flag low confidence or unusual detection for review if applicable
            if not detection_items:
                should_flag = True
                flagged_class = None
                flagged_confidence = 0.0
            else:
                weakest_item = min(
                    detection_items, key=lambda item: item["confidence"]
                )
                should_flag = any(
                    item["confidence"] < (UNUSUAL_CONFIDENCE_THRESHOLD * 100)
                    for item in detection_items
                )
                flagged_class = weakest_item["name"]
                flagged_confidence = weakest_item["confidence"] / 100.0

            if should_flag:
                try:
                    tbl_check = connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='unusual_creatures'"
                    ).fetchone()
                    if tbl_check:
                        connection.execute("""
                            INSERT INTO unusual_creatures
                            (user_id, image_path, predicted_class, confidence, status)
                            VALUES (?, ?, ?, ?, 'Needs review')
                        """, (
                            session["user_id"],
                            "uploads/" + original_filename,
                            flagged_class,
                            flagged_confidence
                        ))
                except Exception as flag_err:
                    app.logger.warning("Could not record unusual creature flag: %s", flag_err)

            connection.commit()
            t_total = time.time() - t_start
            app.logger.info("Organism detection completed in %.2fs (infer: %.2fs, items: %d)",
                            t_total, t_infer, len(detection_items))

            return render_template(
                "detection.html",
                uploaded_image="uploads/" + original_filename,
                result_image="results/" + result_filename,
                detections=detection_items,
                record_id=detection_id
            )

        except Exception as e:
            app.logger.exception("Image detection failed: %s", e)
            if connection is not None:
                connection.rollback()

            for path in [image_path, result_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        app.logger.warning("Could not remove temporary file: %s", path)

            flash("Detection could not be completed. Please try another image.")
            return redirect(url_for("detect"))

        finally:
            if connection is not None:
                connection.close()
            gc.collect()

    return render_template("detection.html")


# ---------------- DETECTION HISTORY ----------------

@app.route("/detection-history")
def detection_history():
    if "user_id" not in session:
        flash("Please sign in first.")
        return redirect(url_for("login"))

    connection = get_db_connection()

    records = connection.execute("""
        SELECT * FROM detections
        WHERE user_id = ?
        ORDER BY id DESC
    """, (session["user_id"],)).fetchall()

    history = []

    for record in records:
        items = connection.execute("""
            SELECT organism_name, confidence
            FROM detection_items
            WHERE detection_id = ?
        """, (record["id"],)).fetchall()

        history.append({
            "record": record,
            "items": items
        })

    connection.close()

    return render_template("detection_history.html", history=history)


# ---------------- MARINE POLLUTION DETECTION ----------------

@app.route("/pollution-detect", methods=["GET", "POST"])
def pollution_detect():
    if "user_id" not in session:
        flash("Please sign in first.")
        return redirect(url_for("login"))

    if session.get("role") in ("organization", "org_member"):
        if not require_org_service_access():
            if session.get("role") == "organization":
                return redirect(url_for("subscription"))
            return redirect(url_for("dashboard"))

    current_model = get_pollution_model()
    if current_model is None:
        return render_template("pollution_detection.html", result=None,
                               model_missing=True)

    if request.method == "POST":
        if "image" not in request.files:
            flash("Please select an image.")
            return redirect(url_for("pollution_detect"))

        file = request.files["image"]
        if not file or not file.filename:
            flash("Please select an image.")
            return redirect(url_for("pollution_detect"))

        if not allowed_file(file.filename):
            flash("Upload a JPG, JPEG, PNG, or WEBP image.")
            return redirect(url_for("pollution_detect"))

        unique_name = uuid.uuid4().hex
        original_filename = f"{unique_name}.jpg"
        result_filename = f"{unique_name}_pollution.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, original_filename)
        result_path = os.path.join(POLLUTION_RESULT_FOLDER, result_filename)
        connection = None
        t_start = time.time()

        ok, err_msg = preprocess_and_save_upload(file, image_path)
        if not ok:
            app.logger.warning("Pollution upload validation failed for %s: %s", file.filename, err_msg)
            flash("The uploaded file is not a readable supported image. Please upload a valid JPG, PNG, or WEBP file.")
            return redirect(url_for("pollution_detect"))

        try:
            t_infer_start = time.time()
            prediction = current_model.predict(
                source=image_path,
                conf=MODEL_CONFIDENCE_THRESHOLD,
                imgsz=640,
                device="cpu",
                verbose=False
            )[0]
            t_infer = time.time() - t_infer_start

            annotated_image = prediction.plot()
            result_img = Image.fromarray(annotated_image[:, :, ::-1])
            result_img.save(result_path, format="JPEG", quality=85, optimize=True)

            del annotated_image, result_img

            items = []
            for box in prediction.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item()) * 100
                items.append({
                    "name": prediction.names[class_id],
                    "confidence": round(confidence, 2)
                })

            del prediction
            gc.collect()

            connection = get_db_connection()
            cursor = connection.execute("""
                INSERT INTO pollution_records
                (user_id, original_image, result_image, detected_count)
                VALUES (?, ?, ?, ?)
            """, (
                session["user_id"],
                "uploads/" + original_filename,
                "pollution_results/" + result_filename,
                len(items)
            ))
            record_id = cursor.lastrowid

            for item in items:
                connection.execute("""
                    INSERT INTO pollution_items
                    (pollution_record_id, class_name, confidence)
                    VALUES (?, ?, ?)
                """, (record_id, item["name"], item["confidence"]))

            connection.commit()
            t_total = time.time() - t_start
            app.logger.info("Pollution detection completed in %.2fs (infer: %.2fs, items: %d)",
                            t_total, t_infer, len(items))

            result = {
                "record_id": record_id,
                "original_image": "uploads/" + original_filename,
                "result_image": "pollution_results/" + result_filename,
                "items": items
            }
            return render_template("pollution_detection.html", result=result,
                                   model_missing=False)

        except Exception as e:
            app.logger.exception("Pollution detection failed: %s", e)
            if connection is not None:
                connection.rollback()
            for path in (image_path, result_path):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        app.logger.warning("Could not remove file: %s", path)

            flash("Pollution detection could not be completed. Please try another image.")
            return redirect(url_for("pollution_detect"))

        finally:
            if connection is not None:
                connection.close()
            gc.collect()

    return render_template("pollution_detection.html", result=None,
                           model_missing=False)


POLLUTION_HISTORY_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pollution Detection History | MarineVision</title>
  <style>
    body{font-family:Arial,sans-serif;margin:0;padding:24px;background:#eef5f7;color:#18313b}
    .wrap{max-width:1100px;margin:auto}.top{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
    .btn{display:inline-block;background:#075985;color:#fff;padding:10px 14px;border-radius:7px;text-decoration:none}
    .card{background:#fff;padding:20px;margin:18px 0;border-radius:12px;box-shadow:0 3px 14px #0000000c}
    .images{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}
    img{width:100%;max-height:320px;object-fit:contain;background:#f8fafc;border:1px solid #dbe5e9;border-radius:8px}
    .muted{color:#526873}.items{border-collapse:collapse;width:100%;margin-top:12px}.items th,.items td{text-align:left;padding:9px;border-bottom:1px solid #e3eaed}
    .empty{background:white;padding:26px;border-radius:12px;margin-top:20px}
  </style>
</head>
<body><main class="wrap">
  <div class="top"><div><h1>Pollution Detection History</h1><p class="muted">Previously analyzed images and detected pollution classes.</p></div>
  <div><a class="btn" href="{{ url_for('pollution_detect') }}">New Detection</a> <a class="btn" href="{{ url_for('dashboard') }}">Dashboard</a></div></div>
  {% if history %}
    {% for entry in history %}
      {% set record = entry.record %}
      <section class="card">
        <h2>Detection #{{ record.id }}</h2>
        <p class="muted">Date: {{ record.created_at }} · Objects detected: {{ record.detected_count }}</p>
        <div class="images">
          <div><h3>Original image</h3><a href="{{ url_for('static', filename=record.original_image) }}" target="_blank" rel="noopener"><img src="{{ url_for('static', filename=record.original_image) }}" alt="Original uploaded image" loading="lazy"></a></div>
          <div><h3>Detection result</h3><a href="{{ url_for('static', filename=record.result_image) }}" target="_blank" rel="noopener"><img src="{{ url_for('static', filename=record.result_image) }}" alt="Pollution detection result" loading="lazy"></a><p><a class="btn" href="{{ url_for('download_detection_result', record_type='pollution', record_id=record.id) }}">Download Result</a></p></div>
        </div>
        {% if entry['items'] %}
          <h3>Detected items</h3><table class="items"><thead><tr><th>Class</th><th>Confidence</th></tr></thead><tbody>
          {% for item in entry['items'] %}<tr><td>{{ item.class_name }}</td><td>{{ '%.2f'|format(item.confidence) }}%</td></tr>{% endfor %}
          </tbody></table>
        {% else %}<p class="muted">No pollution objects were detected in this image.</p>{% endif %}
      </section>
    {% endfor %}
  {% else %}<div class="empty"><h2>No pollution history yet</h2><p>Upload an image from Pollution Detection. After it is processed, the record will appear here.</p><a class="btn" href="{{ url_for('pollution_detect') }}">Start pollution detection</a></div>{% endif %}
</main></body></html>
"""

@app.route("/pollution-history")
def pollution_history():
    if "user_id" not in session:
        flash("Please sign in first.")
        return redirect(url_for("login"))

    connection = None
    try:
        connection = get_db_connection()
        records = connection.execute("""
            SELECT id, user_id, original_image, result_image, detected_count, created_at
            FROM pollution_records
            WHERE user_id = ?
            ORDER BY datetime(created_at) DESC, id DESC
        """, (session["user_id"],)).fetchall()
        history = []
        for record in records:
            items = connection.execute("""
                SELECT class_name, confidence
                FROM pollution_items
                WHERE pollution_record_id = ?
                ORDER BY id ASC
            """, (record["id"],)).fetchall()
            history.append({"record": record, "items": items})
        return render_template_string(POLLUTION_HISTORY_PAGE, history=history)
    except sqlite3.Error:
        app.logger.exception("Could not load pollution detection history")
        flash("Pollution history could not be loaded. Please try again.")
        return redirect(url_for("pollution_detect"))
    finally:
        if connection is not None:
            connection.close()


# ---------------- FORGOT / RESET PASSWORD ----------------

FORGOT_PASSWORD_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forgot Password | MarineVision</title>
<style>
body{font-family:Arial,sans-serif;background:#eef5f7;padding:25px}
.box{max-width:460px;margin:40px auto;background:white;padding:25px;border-radius:12px;box-shadow:0 4px 18px #0001}
h2{color:#075985}input,button{box-sizing:border-box;width:100%;padding:12px;margin:8px 0 16px;border:1px solid #ccd5d9;border-radius:6px}
button{background:#075985;color:white;border:0;cursor:pointer}.msg{background:#e0f2fe;padding:10px;border-radius:6px;overflow-wrap:anywhere}
a{color:#075985}
</style>
</head>
<body><div class="box">
{% if stage == 'email' %}
<h2>Forgot Password?</h2>
<p>Enter the email address registered to your MarineVision account.</p>
{% for message in get_flashed_messages() %}<p class="msg">{{ message }}</p>{% endfor %}
<form method="POST">
<label for="email">Registered Email Address</label>
<input id="email" type="email" name="email" autocomplete="email" required>
<button type="submit">Continue</button>
</form>
{% else %}
<h2>Set a New Password</h2>
<p>Enter and confirm your new password.</p>
{% for message in get_flashed_messages() %}<p class="msg">{{ message }}</p>{% endfor %}
<form method="POST">
<label for="password">New Password (minimum 8 characters)</label>
<input id="password" type="password" name="password" minlength="8" autocomplete="new-password" required>
<label for="confirm_password">Confirm New Password</label>
<input id="confirm_password" type="password" name="confirm_password" minlength="8" autocomplete="new-password" required>
<button type="submit">Update Password</button>
</form>
{% endif %}
<p><a href="{{ url_for('login') }}">Back to user login</a> | <a href="{{ url_for('admin_login') }}">Admin login</a></p>
</div></body></html>
"""


def handle_forgot_password():
    """Simple academic-project reset flow for all roles; does not verify mailbox ownership."""
    reset_user_id = session.get("password_reset_user_id")

    if request.method == "POST" and not reset_user_id:
        email = request.form.get("email", "").strip().lower()
        if not email:
            flash("Please enter your registered email address.")
            return render_template_string(FORGOT_PASSWORD_PAGE, stage="email")

        connection = get_db_connection()
        try:
            user = connection.execute(
                "SELECT id FROM users WHERE lower(email) = ?",
                (email,)
            ).fetchone()
        finally:
            connection.close()

        if user is None:
            flash("No account was found with that email address. Please check it and try again.")
            return render_template_string(FORGOT_PASSWORD_PAGE, stage="email")

        # This session step is NOT proof of email ownership. Use only for a controlled demo.
        session["password_reset_user_id"] = user["id"]
        return render_template_string(FORGOT_PASSWORD_PAGE, stage="password")

    if request.method == "POST" and reset_user_id:
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(password) < 8:
            flash("Password must contain at least 8 characters.")
            return render_template_string(FORGOT_PASSWORD_PAGE, stage="password")
        if password != confirm_password:
            flash("The passwords do not match.")
            return render_template_string(FORGOT_PASSWORD_PAGE, stage="password")

        connection = get_db_connection()
        try:
            user = connection.execute(
                "SELECT id, role FROM users WHERE id = ?",
                (reset_user_id,)
            ).fetchone()
            if user is None:
                session.pop("password_reset_user_id", None)
                flash("Account not found. Please start again.")
                return redirect(url_for("forgot_password"))

            connection.execute(
                "UPDATE users SET password = ? WHERE id = ?",
                (generate_password_hash(password), reset_user_id)
            )
            connection.commit()
            role = user["role"]
        except sqlite3.Error:
            connection.rollback()
            app.logger.exception("Password reset failed")
            flash("Could not update the password. Please try again.")
            return render_template_string(FORGOT_PASSWORD_PAGE, stage="password")
        finally:
            connection.close()

        session.pop("password_reset_user_id", None)
        flash("Password updated. Please log in with your new password.")
        return redirect(url_for("admin_login") if role == "admin" else url_for("login"))

    return render_template_string(
        FORGOT_PASSWORD_PAGE,
        stage="password" if reset_user_id else "email"
    )


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    return handle_forgot_password()


@app.route("/admin/forgot-password", methods=["GET", "POST"])
def admin_forgot_password():
    # All roles use the same reset page.
    return redirect(url_for("forgot_password"))


# ---------------- ADMIN SETUP, LOGIN AND DASHBOARD ----------------
# Admin pages are written inside app.py, so no extra HTML templates are needed.

ADMIN_SETUP_PAGE = """
<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Create Admin | MarineVision</title>
<style>body{font-family:Arial;background:#eef5f7;margin:0;padding:30px}.box{max-width:480px;margin:30px auto;background:white;padding:25px;border-radius:12px;box-shadow:0 4px 18px #0001}input,button{box-sizing:border-box;width:100%;padding:11px;margin:7px 0 14px;border:1px solid #ccd5d9;border-radius:6px}button{background:#075985;color:white;border:0;cursor:pointer}.msg{color:#b91c1c}</style></head>
<body><div class="box"><h1>Create first admin</h1><p>This page works only until an admin account has been created.</p>
{% for message in get_flashed_messages() %}<p class="msg">{{ message }}</p>{% endfor %}
<form method="post"><label>Full name</label><input name="name" required><label>Email</label><input type="email" name="email" required><label>Username</label><input name="username" required><label>Password (minimum 8 characters)</label><input type="password" name="password" minlength="8" required><button type="submit">Create Admin Account</button></form>
<p><a href="{{ url_for('login') }}">Back to normal login</a></p></div></body></html>
"""

ADMIN_LOGIN_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Admin Login | MarineVision</title>
<style>body{font-family:Arial;background:#eef5f7;margin:0;padding:30px}.box{max-width:420px;margin:50px auto;background:white;padding:25px;border-radius:12px;box-shadow:0 4px 18px #0001}input,button{box-sizing:border-box;width:100%;padding:11px;margin:7px 0 14px;border:1px solid #ccd5d9;border-radius:6px}button{background:#075985;color:white;border:0;cursor:pointer}.msg{color:#b91c1c}</style></head>
<body><div class="box"><h1>MarineVision Admin</h1>{% for message in get_flashed_messages() %}<p class="msg">{{ message }}</p>{% endfor %}
<form method="post"><label>Admin username</label><input name="username" required><label>Password</label><input type="password" name="password" required><button type="submit">Sign In</button></form><p><a href="{{ url_for('admin_forgot_password') }}">Forgot password?</a></p><p><a href="{{ url_for('login') }}">User login</a></p></div></body></html>
"""

ADMIN_DASHBOARD_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Admin Dashboard | MarineVision</title>
<style>body{font-family:Arial;background:#eef5f7;margin:0;color:#18313b}.top{background:#075985;color:white;padding:20px 5%;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap}.top a{color:white;margin-left:14px}.wrap{max-width:1200px;margin:25px auto;padding:0 16px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px}.card,section{background:white;padding:18px;border-radius:10px;box-shadow:0 3px 12px #0000000b;margin-bottom:18px}.num{font-size:28px;font-weight:bold;color:#075985}table{width:100%;border-collapse:collapse;display:block;overflow-x:auto}th,td{text-align:left;padding:10px;border-bottom:1px solid #e1e8eb;white-space:nowrap}th{background:#f3f7f8}h2{margin-top:0}.muted{color:#52666e}</style></head>
<body><header class="top"><div><h1>MarineVision Admin Dashboard</h1><p>Welcome, {{ admin_name }}</p></div><nav><a href="{{ url_for('admin_dashboard') }}">Refresh</a><a href="{{ url_for('my_profile') }}">My Profile</a><a href="{{ url_for('admin_download_limits') }}">Download Limits</a><a href="{{ url_for('admin_logout') }}">Logout</a></nav></header>
<main class="wrap">{% for message in get_flashed_messages() %}<section style="color:#075985;font-weight:bold">{{ message }}</section>{% endfor %}<div class="cards">
<div class="card"><div>Registered users</div><div class="num">{{ counts.users }}</div></div>
<div class="card"><div>Organism detections</div><div class="num">{{ counts.detections }}</div></div>
<div class="card"><div>Pollution records</div><div class="num">{{ counts.pollution }}</div></div>
<div class="card"><div>Subscriptions</div><div class="num">{{ counts.subscriptions }}</div></div>
</div>
<section><h2>Pending account requests</h2><p>Approve developer and organization registrations before they can sign in.</p>{% if pending_users %}<table><tr><th>ID</th><th>Name</th><th>Email</th><th>Username</th><th>Role</th><th>Actions</th></tr>{% for u in pending_users %}<tr><td>{{ u['id'] }}</td><td>{{ u['name'] }}</td><td>{{ u['email'] }}</td><td>{{ u['username'] }}</td><td>{{ u['role'] }}</td><td><form method="post" action="{{ url_for('admin_approve_user', user_id=u['id']) }}" style="display:inline"><button type="submit">Approve</button></form> <form method="post" action="{{ url_for('admin_reject_user', user_id=u['id']) }}" style="display:inline" onsubmit="return confirm('Reject and remove this registration?')"><button type="submit" style="background:#b91c1c">Reject</button></form></td></tr>{% endfor %}</table>{% else %}<p>No pending requests.</p>{% endif %}</section>
<section><h2>Registered users and organizations</h2>{% if users %}<table><tr><th>ID</th><th>Name</th><th>Email</th><th>Username</th><th>Role</th><th>Status</th><th>Actions</th></tr>{% for u in users %}<tr><td>{{ u['id'] }}</td><td>{{ u['name'] }}</td><td>{{ u['email'] }}</td><td>{{ u['username'] }}</td><td>{{ u['role'] }}</td><td>{{ u['approval_status'] }}</td><td><a href="{{ url_for('admin_user_profile', user_id=u['id']) }}" style="display:inline-block;background:#075985;color:white;padding:7px 10px;border-radius:5px;text-decoration:none;margin-right:5px">View Profile</a>{% if u['role'] != 'admin' %}<form method="post" action="{{ url_for('admin_remove_user', user_id=u['id']) }}" style="display:inline" onsubmit="return confirm('Permanently remove this account and its related records?')"><button type="submit" style="background:#b91c1c">Remove</button></form>{% else %}Protected admin{% endif %}</td></tr>{% endfor %}</table>{% else %}<p>No users found.</p>{% endif %}</section>
<section><h2>Recent organism detections</h2>{% if detections %}<table><tr><th>ID</th><th>User ID</th><th>Detected count</th><th>Date</th></tr>{% for d in detections %}<tr><td>{{ d['id'] }}</td><td>{{ d['user_id'] }}</td><td>{{ d['detected_count'] }}</td><td>{{ d['created_at'] }}</td></tr>{% endfor %}</table>{% else %}<p>No organism detections yet.</p>{% endif %}</section>
<section><h2>Recent pollution records</h2>{% if pollution_records %}<table><tr><th>ID</th><th>User ID</th><th>Detected count</th><th>Date</th></tr>{% for d in pollution_records %}<tr><td>{{ d['id'] }}</td><td>{{ d['user_id'] }}</td><td>{{ d['detected_count'] }}</td><td>{{ d['created_at'] }}</td></tr>{% endfor %}</table>{% else %}<p>No pollution records yet.</p>{% endif %}</section>
<section><h2>Subscriptions and payments</h2>{% if subscriptions %}<table><tr><th>ID</th><th>User ID</th><th>Organization</th><th>Duration (months)</th><th>Amount</th><th>Status</th></tr>{% for sub in subscriptions %}<tr><td>{{ sub['id'] }}</td><td>{{ sub['user_id'] }}</td><td>{{ sub['organization_name'] }}</td><td>{{ sub['duration_months'] }}</td><td>{{ sub['amount'] }}</td><td>{{ sub['payment_status'] }}</td></tr>{% endfor %}</table>{% else %}<p>No subscriptions yet.</p>{% endif %}</section>
<section><h2>User feedback</h2>{% if feedback_records %}<table><tr><th>ID</th><th>User</th><th>Email</th><th>Rating</th><th>Feedback</th><th>Date</th></tr>{% for f in feedback_records %}<tr><td>{{ f['id'] }}</td><td>{{ f['username'] }}</td><td>{{ f['email'] }}</td><td>{{ f['rating'] }}/5</td><td style="white-space:normal;min-width:220px">{{ f['message'] }}</td><td>{{ f['created_at'] }}</td></tr>{% endfor %}</table>{% else %}<p>No feedback submitted yet.</p>{% endif %}</section>
<p class="muted">Dashboard displays records stored in the existing MarineVision database.</p></main></body></html>
"""


def admin_required():
    return "user_id" in session and session.get("role") == "admin"


@app.route("/admin/setup", methods=["GET", "POST"])
def admin_setup():
    # This first-run page is disabled as soon as an admin account exists.
    connection = get_db_connection()
    existing_admin = connection.execute(
        "SELECT id FROM users WHERE role = 'admin' LIMIT 1"
    ).fetchone()
    connection.close()

    if existing_admin:
        flash("An admin account already exists. Please sign in.")
        return redirect(url_for("admin_login"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not all([name, email, username, password]):
            flash("Please complete every field.")
            return redirect(url_for("admin_setup"))
        if len(password) < 8:
            flash("Password must contain at least 8 characters.")
            return redirect(url_for("admin_setup"))

        connection = get_db_connection()
        try:
            connection.execute("""
                INSERT INTO users (name, email, username, password, role)
                VALUES (?, ?, ?, ?, 'admin')
            """, (name, email, username, generate_password_hash(password)))
            connection.commit()
            flash("Admin account created. Please sign in.")
            return redirect(url_for("admin_login"))
        except sqlite3.IntegrityError:
            flash("That email or username is already in use.")
            return redirect(url_for("admin_setup"))
        finally:
            connection.close()

    return render_template_string(ADMIN_SETUP_PAGE)


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        connection = get_db_connection()
        user = connection.execute(
            "SELECT * FROM users WHERE username = ? AND role = 'admin'",
            (username,)
        ).fetchone()
        connection.close()

        if user and check_password_hash(user["password"], password):
            login_connection = get_db_connection()
            login_connection.execute(
                "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?",
                (user["id"],)
            )
            login_connection.commit()
            login_connection.close()
            session.clear()
            session["user_id"] = user["id"]
            session["name"] = user["name"]
            session["username"] = user["username"]
            session["role"] = "admin"
            return redirect(url_for("admin_dashboard"))

        flash("Incorrect admin username or password.")
        return redirect(url_for("admin_login"))

    return render_template_string(ADMIN_LOGIN_PAGE)


@app.route("/admin/dashboard")
def admin_dashboard():
    if not admin_required():
        flash("Please sign in with an admin account.")
        return redirect(url_for("admin_login"))

    connection = get_db_connection()
    users = connection.execute(
        "SELECT id, name, email, username, role, approval_status FROM users ORDER BY id DESC"
    ).fetchall()
    pending_users = connection.execute(
        "SELECT id, name, email, username, role, approval_status FROM users WHERE role IN ('developer', 'organization') AND approval_status = 'Pending' ORDER BY id ASC"
    ).fetchall()
    detections = connection.execute(
        "SELECT id, user_id, detected_count, created_at FROM detections ORDER BY id DESC LIMIT 20"
    ).fetchall()
    subscriptions = connection.execute(
        "SELECT id, user_id, organization_name, duration_months, amount, payment_status FROM subscriptions ORDER BY id DESC LIMIT 20"
    ).fetchall()

    # Older databases may not yet have these optional feature tables.
    table_names = {r["name"] for r in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    pollution_records = []
    if "pollution_records" in table_names:
        pollution_records = connection.execute(
            "SELECT id, user_id, detected_count, created_at FROM pollution_records ORDER BY id DESC LIMIT 20"
        ).fetchall()
    feedback_records = []
    if "feedback" in table_names:
        feedback_records = connection.execute("""
            SELECT feedback.id, users.username, users.email, feedback.rating,
                   feedback.message, feedback.created_at
            FROM feedback JOIN users ON users.id = feedback.user_id
            ORDER BY feedback.id DESC LIMIT 100
        """).fetchall()

    counts = {
        "users": len(users),
        "detections": connection.execute("SELECT COUNT(*) FROM detections").fetchone()[0],
        "pollution": connection.execute("SELECT COUNT(*) FROM pollution_records").fetchone()[0] if "pollution_records" in table_names else 0,
        "subscriptions": connection.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0],
        "feedback": connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] if "feedback" in table_names else 0,
    }
    connection.close()
    return render_template_string(
        ADMIN_DASHBOARD_PAGE, admin_name=session.get("name", "Admin"),
        counts=counts, users=users, pending_users=pending_users, detections=detections,
        pollution_records=pollution_records, subscriptions=subscriptions,
        feedback_records=feedback_records
    )



# ---------------- MY PROFILE (ALL SIGNED-IN ROLES) ----------------

MY_PROFILE_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>My Profile | MarineVision</title>
<style>
*{box-sizing:border-box}body{font-family:Arial,sans-serif;margin:0;background:#eef6f8;color:#173b46}
header{background:#075985;color:white;padding:22px 5%;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
header a{color:#075985;background:white;text-decoration:none;padding:10px 14px;border-radius:7px;font-weight:bold}
main{max-width:1000px;margin:26px auto;padding:0 16px}.panel{background:white;padding:22px;border-radius:12px;margin-bottom:18px;box-shadow:0 3px 12px #0001}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.field,.stat{background:#f3f8fa;padding:14px;border-radius:8px;overflow-wrap:anywhere}
.label{font-size:12px;color:#617780;text-transform:uppercase;letter-spacing:.04em}
.value{font-weight:bold;font-size:16px;margin-top:7px}
.profile-form{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}.profile-form label{display:block;font-weight:bold;font-size:14px}.profile-form input,.profile-form textarea,.profile-form select{display:block;width:100%;margin-top:6px;padding:10px;border:1px solid #cbd5da;border-radius:7px;font:inherit}.profile-form textarea{min-height:90px}.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}.actions button,.actions a{border:0;border-radius:7px;padding:10px 14px;font-weight:bold;text-decoration:none;cursor:pointer}.primary{background:#075985;color:white}.secondary{background:#e2e8f0;color:#173b46}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.number{font-size:27px;font-weight:bold;color:#075985;margin-top:5px}
table{width:100%;border-collapse:collapse;display:block;overflow-x:auto}
th,td{text-align:left;padding:10px;border-bottom:1px solid #e1e8eb;white-space:nowrap}
th{background:#e0f2fe}.muted{color:#60747d}.message{background:#fff3cd;padding:12px;border-radius:8px;margin-bottom:12px}
</style>
</head>
<body>
<header>
 <div><h1 style="margin:0">My MarineVision Profile</h1><p style="margin:7px 0 0">Your personal account information and activity</p></div>
 <div><a href="{{ url_for('admin_dashboard') if role == 'admin' else url_for('dashboard') }}">Back to Dashboard</a></div>
</header>
<main>
 {% for message in get_flashed_messages() %}<div class="message">{{ message }}</div>{% endfor %}
 <section class="panel">
  <h2>Account Details</h2>
  {% if editing %}
  <form class="profile-form" method="post" action="{{ url_for('edit_profile') }}">
   <label>Full Name<input name="name" value="{{ profile['name'] or '' }}" maxlength="120" required></label>
   <label>Email Address<input type="email" name="email" value="{{ profile['email'] or '' }}" maxlength="254" required></label>
   <label>Username (read-only)<input value="{{ profile['username'] }}" readonly></label>
   <label>Account Role (read-only)<input value="{{ profile['role']|title }}" readonly></label>
   <label>Contact Number<input name="contact_number" value="{{ profile['contact_number'] or '' }}" maxlength="40"></label>
   <label>Address<input name="address" value="{{ profile['address'] or '' }}" maxlength="500"></label>
   <label>Profession / Occupation<select name="profession" id="profession" required>{% for choice in profession_choices %}<option value="{{ choice }}" {% if profile['profession'] == choice %}selected{% endif %}>{{ choice }}</option>{% endfor %}{% if profile['profession'] and profile['profession'] not in profession_choices and profile['profession'] != 'Other' %}<option value="{{ profile['profession'] }}" selected>{{ profile['profession'] }} (current)</option>{% endif %}</select></label>
   <label id="profession-other-wrap" {% if profile['profession'] != 'Other' %}hidden{% endif %}>Other Profession<input name="profession_other" maxlength="120" value="{{ profile['profession_other'] or '' }}"></label>
   <label>Designation / Job Title<input name="designation" value="{{ profile['designation'] or '' }}" maxlength="120"></label>
   <label>Highest Qualification<input name="highest_qualification" value="{{ profile['highest_qualification'] or '' }}" maxlength="160"></label>
   <label>Areas of Interest<textarea name="areas_of_interest" maxlength="1000">{{ profile['areas_of_interest'] or '' }}</textarea></label>
   <div style="grid-column:1/-1"><div class="label">Account Status</div><div class="value">{{ profile['approval_status'] or 'Approved' }}</div><div class="label" style="margin-top:10px">Date Joined</div><div class="value">{{ profile['created_at'] or 'Not Provided' }}</div></div>
   <div class="actions" style="grid-column:1/-1"><button class="primary" type="submit">Save Changes</button><a class="secondary" href="{{ url_for('my_profile') }}">Cancel</a></div>
  </form><script>document.getElementById('profession').addEventListener('change',function(){document.getElementById('profession-other-wrap').hidden=this.value!=='Other';});</script>
  {% else %}
  <div class="grid">
   {% for label, key in [('Full Name','name'),('Email Address','email'),('Username','username'),('Contact Number','contact_number'),('Address','address'),('Profession / Occupation','profession'),('Other Profession','profession_other'),('Designation / Job Title','designation'),('Highest Qualification','highest_qualification'),('Areas of Interest','areas_of_interest')] %}<div class="field"><div class="label">{{ label }}</div><div class="value">{{ profile[key] or 'Not Provided' }}</div></div>{% endfor %}
   <div class="field"><div class="label">Role</div><div class="value">{{ profile['role']|title }}</div></div>
   <div class="field"><div class="label">Account Status</div><div class="value">{{ profile['approval_status'] or 'Approved' }}</div></div>
   <div class="field"><div class="label">Date Joined</div><div class="value">{{ profile['created_at'] or 'Not Provided' }}</div></div>
   {% if organization_name %}<div class="field"><div class="label">Organization</div><div class="value">{{ organization_name }}</div></div>{% endif %}
  </div>
  <div class="actions"><a class="primary" href="{{ url_for('my_profile', edit=1) }}">Edit Profile</a></div>
  {% endif %}
 </section>
 <section class="panel">
  <h2>My Activity Summary</h2>
  <div class="stats">
   <div class="stat"><div>Organism detections</div><div class="number">{{ counts.organism }}</div></div>
   <div class="stat"><div>Pollution records</div><div class="number">{{ counts.pollution }}</div></div>
   <div class="stat"><div>Feedback submitted</div><div class="number">{{ counts.feedback }}</div></div>
  </div>
 </section>
 {% if role == 'organization' %}
 <section class="panel"><h2>Subscription Information</h2>
 {% if subscriptions %}
 <table><tr><th>Organization</th><th>Duration</th><th>Amount</th><th>Status</th><th>Created</th></tr>
 {% for sub in subscriptions %}<tr><td>{{ sub['organization_name'] }}</td><td>{{ sub['duration_months'] }} month(s)</td><td>₹{{ sub['amount'] }}</td><td>{{ sub['payment_status'] }}</td><td>{{ sub['created_at'] }}</td></tr>{% endfor %}
 </table>
 {% else %}<p>No subscription has been created yet.</p><p><a href="{{ url_for('subscription') }}">Manage subscription</a></p>{% endif %}
 </section>
 <section class="panel"><h2>Recent Payments</h2>
 {% if payments %}<table><tr><th>Transaction ID</th><th>Method</th><th>Amount</th><th>Status</th><th>Date</th></tr>
 {% for payment in payments %}<tr><td>{{ payment['transaction_id'] }}</td><td>{{ payment['payment_method'] }}</td><td>₹{{ payment['amount'] }}</td><td>{{ payment['payment_status'] }}</td><td>{{ payment['created_at'] }}</td></tr>{% endfor %}</table>
 {% else %}<p>No payment records yet.</p>{% endif %}
 </section>
 {% endif %}
 <section class="panel"><h2>Recent Organism Detections</h2>
 {% if detections %}<table><tr><th>ID</th><th>Objects Detected</th><th>Date</th></tr>
 {% for row in detections %}<tr><td>{{ row['id'] }}</td><td>{{ row['detected_count'] }}</td><td>{{ row['created_at'] }}</td></tr>{% endfor %}</table>
 {% else %}<p>No organism detection records yet.</p>{% endif %}
 </section>
</main>
</body>
</html>
"""

@app.route("/profile")
def my_profile():
    # A user can only view the profile belonging to their active session.
    if "user_id" not in session:
        flash("Please sign in first.")
        return redirect(url_for("login"))

    user_id = session["user_id"]
    role = session.get("role", "")
    connection = get_db_connection()
    try:
        profile = connection.execute(
            "SELECT id, name, email, username, role, approval_status, created_at, contact_number, address, profession, profession_other, designation, highest_qualification, areas_of_interest FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
        if profile is None:
            session.clear()
            flash("Your account could not be found. Please sign in again.")
            return redirect(url_for("login"))

        table_names = {row["name"] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        counts = {
            "organism": connection.execute(
                "SELECT COUNT(*) FROM detections WHERE user_id = ?", (user_id,)
            ).fetchone()[0],
            "pollution": connection.execute(
                "SELECT COUNT(*) FROM pollution_records WHERE user_id = ?", (user_id,)
            ).fetchone()[0] if "pollution_records" in table_names else 0,
            "feedback": connection.execute(
                "SELECT COUNT(*) FROM feedback WHERE user_id = ?", (user_id,)
            ).fetchone()[0] if "feedback" in table_names else 0,
        }
        subscriptions = connection.execute(
            "SELECT * FROM subscriptions WHERE user_id = ? ORDER BY id DESC",
            (user_id,)
        ).fetchall() if role == "organization" and "subscriptions" in table_names else []
        payments = connection.execute(
            "SELECT * FROM payments WHERE user_id = ? ORDER BY id DESC",
            (user_id,)
        ).fetchall() if role == "organization" and "payments" in table_names else []
        detections = connection.execute(
            "SELECT id, detected_count, created_at FROM detections WHERE user_id = ? ORDER BY id DESC LIMIT 10",
            (user_id,)
        ).fetchall()
        organization_name = subscriptions[0]["organization_name"] if subscriptions else None

        return render_template_string(
            MY_PROFILE_PAGE, profile=profile, role=role, counts=counts,
            subscriptions=subscriptions, payments=payments,
            detections=detections, organization_name=organization_name,
            editing=request.args.get("edit") == "1",
            profession_choices=("Marine Researcher", "Marine Biologist", "Marine Scientist", "Environmentalist", "Oceanographer", "Research Scholar", "Student", "Environmental Engineer", "Pollution Control Specialist", "Other")
        )
    except sqlite3.Error:
        app.logger.exception("Could not load own profile")
        flash("Your profile could not be loaded. Please try again.")
        return redirect(url_for("admin_dashboard") if role == "admin" else url_for("dashboard"))
    finally:
        connection.close()


@app.route("/edit-profile", methods=["POST"])
def edit_profile():
    if "user_id" not in session:
        flash("Please sign in first.")
        return redirect(url_for("login"))

    user_id = session["user_id"]
    fields = {
        "name": request.form.get("name", "").strip(),
        "email": request.form.get("email", "").strip().lower(),
        "contact_number": request.form.get("contact_number", "").strip(),
        "address": request.form.get("address", "").strip(),
        "profession": request.form.get("profession", "").strip(),
        "profession_other": request.form.get("profession_other", "").strip(),
        "designation": request.form.get("designation", "").strip(),
        "highest_qualification": request.form.get("highest_qualification", "").strip(),
        "areas_of_interest": request.form.get("areas_of_interest", "").strip(),
    }
    choices = {"Marine Researcher", "Marine Biologist", "Marine Scientist", "Environmentalist", "Oceanographer", "Research Scholar", "Student", "Environmental Engineer", "Pollution Control Specialist", "Other"}
    max_lengths = {"name": 120, "email": 254, "contact_number": 40, "address": 500, "profession": 120, "profession_other": 120, "designation": 120, "highest_qualification": 160, "areas_of_interest": 1000}

    error = None
    if not fields["name"]:
        error = "Full name is required."
    elif not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", fields["email"]):
        error = "Enter a valid email address."
    elif any(len(value) > max_lengths[key] for key, value in fields.items()):
        error = "One or more profile fields exceed the allowed length."
    elif not fields["profession"]:
        error = "Please select a profession."
    elif fields["profession"] == "Other" and not fields["profession_other"]:
        error = "Please enter your profession."

    connection = get_db_connection()
    try:
        current = connection.execute("SELECT id, profession FROM users WHERE id = ?", (user_id,)).fetchone()
        if current is None:
            session.clear()
            flash("Your account could not be found. Please sign in again.")
            return redirect(url_for("login"))
        if not error and fields["profession"] not in choices and fields["profession"] != current["profession"]:
            error = "Please choose a listed profession."
        if not error and connection.execute(
            "SELECT id FROM users WHERE lower(email) = lower(?) AND id != ?",
            (fields["email"], user_id)
        ).fetchone():
            error = "That email address is already in use."
        if error:
            flash(error)
            return redirect(url_for("my_profile", edit=1))

        profession_other = fields["profession_other"] if fields["profession"] == "Other" else None
        connection.execute("""
            UPDATE users SET name = ?, email = ?, contact_number = ?, address = ?,
                profession = ?, profession_other = ?, designation = ?,
                highest_qualification = ?, areas_of_interest = ? WHERE id = ?
        """, (fields["name"], fields["email"], fields["contact_number"] or None,
              fields["address"] or None, fields["profession"], profession_other,
              fields["designation"] or None, fields["highest_qualification"] or None,
              fields["areas_of_interest"] or None, user_id))
        connection.commit()
        session["name"] = fields["name"]
        flash("Your profile was updated successfully.")
        return redirect(url_for("my_profile"))
    except sqlite3.IntegrityError:
        connection.rollback()
        flash("That email address is already in use.")
        return redirect(url_for("my_profile", edit=1))
    except sqlite3.Error:
        connection.rollback()
        app.logger.exception("Could not update own profile")
        flash("Your profile could not be updated. Please try again.")
        return redirect(url_for("my_profile", edit=1))
    finally:
        connection.close()


# ---------------- INDIVIDUAL USER PROFILE (ADMIN ONLY) ----------------

ADMIN_USER_PROFILE_PAGE = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>User Profile | MarineVision</title>
<style>
*{box-sizing:border-box}body{font-family:Arial,sans-serif;margin:0;background:#eef5f7;color:#18313b}.top{background:#075985;color:white;padding:20px 5%;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}.top a{color:white}.wrap{max-width:1100px;margin:24px auto;padding:0 16px}.panel{background:white;border-radius:12px;padding:22px;margin-bottom:18px;box-shadow:0 3px 12px #0000000b}.profile-head{display:flex;gap:16px;align-items:center;flex-wrap:wrap}.avatar{width:72px;height:72px;border-radius:50%;background:#dbeafe;color:#075985;display:flex;align-items:center;justify-content:center;font-size:28px;font-weight:bold}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.field{background:#f6fafb;border-radius:8px;padding:12px;overflow-wrap:anywhere}.label{font-size:12px;color:#60747d;text-transform:uppercase;letter-spacing:.04em}.value{font-size:16px;font-weight:600;margin-top:6px}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}.stat{background:#f6fafb;padding:16px;border-radius:8px}.number{font-size:26px;color:#075985;font-weight:bold;margin-top:5px}table{width:100%;border-collapse:collapse;display:block;overflow-x:auto}th,td{text-align:left;padding:10px;border-bottom:1px solid #e1e8eb;white-space:nowrap}th{background:#f3f7f8}.btn{display:inline-block;background:#075985;color:white;padding:10px 14px;border-radius:7px;text-decoration:none}.muted{color:#60747d}.pill{display:inline-block;background:#e0f2fe;color:#075985;border-radius:20px;padding:5px 10px;font-size:13px}
</style></head><body>
<header class="top"><div><h1 style="margin:0">Individual User Profile</h1><p style="margin:6px 0 0">MarineVision Admin · User ID {{ profile['id'] }}</p></div><div><a href="{{ url_for('admin_dashboard') }}">Back to Admin Dashboard</a> &nbsp;|&nbsp; <a href="{{ url_for('admin_logout') }}">Logout</a></div></header>
<main class="wrap">
{% for message in get_flashed_messages() %}<section class="panel">{{ message }}</section>{% endfor %}
<section class="panel"><div class="profile-head"><div class="avatar">{{ (profile['name'] or profile['username'] or 'U')[0]|upper }}</div><div><h2 style="margin:0 0 8px">{{ profile['name'] }}</h2><div class="muted">@{{ profile['username'] }}</div><p><span class="pill">{{ profile['role']|title }}</span> &nbsp; <span class="pill">{{ profile['approval_status'] or 'Approved' }}</span></p></div></div></section>
<section class="panel"><h2>Personal and Account Details</h2><div class="grid">
<div class="field"><div class="label">Full name</div><div class="value">{{ profile['name'] }}</div></div>
<div class="field"><div class="label">Username</div><div class="value">{{ profile['username'] }}</div></div>
<div class="field"><div class="label">Email address</div><div class="value">{{ profile['email'] }}</div></div>
<div class="field"><div class="label">Account role</div><div class="value">{{ profile['role']|title }}</div></div>
<div class="field"><div class="label">Approval status</div><div class="value">{{ profile['approval_status'] or 'Approved' }}</div></div>
<div class="field"><div class="label">User ID</div><div class="value">{{ profile['id'] }}</div></div>
</div><p class="muted">Passwords are intentionally never displayed.</p></section>
<section class="panel"><h2>Activity Summary</h2><div class="stats"><div class="stat">Organism detections<div class="number">{{ counts.organism }}</div></div><div class="stat">Pollution records<div class="number">{{ counts.pollution }}</div></div><div class="stat">Feedback submissions<div class="number">{{ counts.feedback }}</div></div></div></section>
{% if profile['role'] == 'organization' %}<section class="panel"><h2>Organization Subscriptions</h2>{% if subscriptions %}<table><tr><th>Organization</th><th>Duration</th><th>Amount</th><th>Payment status</th><th>Date</th></tr>{% for sub in subscriptions %}<tr><td>{{ sub['organization_name'] }}</td><td>{{ sub['duration_months'] }} months</td><td>₹{{ '%.2f'|format(sub['amount']) }}</td><td>{{ sub['payment_status'] }}</td><td>{{ sub['created_at'] }}</td></tr>{% endfor %}</table>{% else %}<p class="muted">No subscription records for this account.</p>{% endif %}
<h3>Payments</h3>{% if payments %}<table><tr><th>Transaction ID</th><th>Method</th><th>Amount</th><th>Status</th><th>Date</th></tr>{% for pay in payments %}<tr><td>{{ pay['transaction_id'] }}</td><td>{{ pay['payment_method'] }}</td><td>₹{{ '%.2f'|format(pay['amount']) }}</td><td>{{ pay['payment_status'] }}</td><td>{{ pay['created_at'] }}</td></tr>{% endfor %}</table>{% else %}<p class="muted">No payment records for this account.</p>{% endif %}</section>{% endif %}
<section class="panel"><h2>Recent Organism Detection History</h2>{% if detections %}<table><tr><th>Detection ID</th><th>Detected count</th><th>Date</th><th>Detected classes</th></tr>{% for d in detections %}<tr><td>{{ d['id'] }}</td><td>{{ d['detected_count'] }}</td><td>{{ d['created_at'] }}</td><td>{{ d['classes'] or 'No objects detected' }}</td></tr>{% endfor %}</table>{% else %}<p class="muted">No organism detections recorded.</p>{% endif %}</section>
<section class="panel"><h2>Recent Pollution Detection History</h2>{% if pollution_records %}<table><tr><th>Record ID</th><th>Detected count</th><th>Date</th><th>Detected classes</th></tr>{% for d in pollution_records %}<tr><td>{{ d['id'] }}</td><td>{{ d['detected_count'] }}</td><td>{{ d['created_at'] }}</td><td>{{ d['classes'] or 'No pollution objects detected' }}</td></tr>{% endfor %}</table>{% else %}<p class="muted">No pollution records recorded.</p>{% endif %}</section>
<section class="panel"><a class="btn" href="{{ url_for('admin_dashboard') }}">Back to All Users</a></section>
</main></body></html>
"""

@app.route("/admin/users/<int:user_id>")
def admin_user_profile(user_id):
    if not admin_required():
        flash("Please sign in with an admin account.")
        return redirect(url_for("admin_login"))

    connection = get_db_connection()
    try:
        profile = connection.execute(
            "SELECT id, name, email, username, role, approval_status FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
        if profile is None:
            flash("That user account was not found.")
            return redirect(url_for("admin_dashboard"))

        table_names = {row["name"] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        counts = {
            "organism": connection.execute("SELECT COUNT(*) FROM detections WHERE user_id = ?", (user_id,)).fetchone()[0],
            "pollution": connection.execute("SELECT COUNT(*) FROM pollution_records WHERE user_id = ?", (user_id,)).fetchone()[0] if "pollution_records" in table_names else 0,
            "feedback": connection.execute("SELECT COUNT(*) FROM feedback WHERE user_id = ?", (user_id,)).fetchone()[0] if "feedback" in table_names else 0,
        }
        subscriptions = connection.execute(
            "SELECT * FROM subscriptions WHERE user_id = ? ORDER BY id DESC", (user_id,)
        ).fetchall() if "subscriptions" in table_names else []
        payments = connection.execute(
            "SELECT * FROM payments WHERE user_id = ? ORDER BY id DESC", (user_id,)
        ).fetchall() if "payments" in table_names else []

        detection_rows = connection.execute(
            "SELECT * FROM detections WHERE user_id = ? ORDER BY id DESC LIMIT 20", (user_id,)
        ).fetchall()
        detections = []
        for row in detection_rows:
            classes = connection.execute(
                "SELECT organism_name FROM detection_items WHERE detection_id = ? ORDER BY id", (row["id"],)
            ).fetchall()
            detections.append({"id": row["id"], "detected_count": row["detected_count"], "created_at": row["created_at"], "classes": ", ".join(item["organism_name"] for item in classes)})

        pollution_records = []
        if "pollution_records" in table_names:
            pollution_rows = connection.execute(
                "SELECT * FROM pollution_records WHERE user_id = ? ORDER BY id DESC LIMIT 20", (user_id,)
            ).fetchall()
            for row in pollution_rows:
                classes = connection.execute(
                    "SELECT class_name FROM pollution_items WHERE pollution_record_id = ? ORDER BY id", (row["id"],)
                ).fetchall() if "pollution_items" in table_names else []
                pollution_records.append({"id": row["id"], "detected_count": row["detected_count"], "created_at": row["created_at"], "classes": ", ".join(item["class_name"] for item in classes)})

        return render_template_string(
            ADMIN_USER_PROFILE_PAGE, profile=profile, counts=counts,
            subscriptions=subscriptions, payments=payments,
            detections=detections, pollution_records=pollution_records
        )
    except sqlite3.Error:
        app.logger.exception("Could not load individual user profile")
        flash("The user profile could not be loaded due to a database error.")
        return redirect(url_for("admin_dashboard"))
    finally:
        connection.close()


@app.route("/admin/users/<int:user_id>/approve", methods=["POST"])
def admin_approve_user(user_id):
    if not admin_required():
        flash("Please sign in with an admin account.")
        return redirect(url_for("admin_login"))
    connection = get_db_connection()
    result = connection.execute(
        "UPDATE users SET approval_status = 'Approved' WHERE id = ? AND role IN ('developer', 'organization')",
        (user_id,)
    )
    connection.commit()
    connection.close()
    flash("Account approved." if result.rowcount else "Account was not found or cannot be approved.")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/users/<int:user_id>/reject", methods=["POST"])
def admin_reject_user(user_id):
    if not admin_required():
        flash("Please sign in with an admin account.")
        return redirect(url_for("admin_login"))
    connection = get_db_connection()
    row = connection.execute(
        "SELECT id FROM users WHERE id = ? AND role IN ('developer', 'organization') AND approval_status = 'Pending'",
        (user_id,)
    ).fetchone()
    connection.close()
    if not row:
        flash("Only pending developer or organization requests can be rejected.")
        return redirect(url_for("admin_dashboard"))
    return remove_user_records(user_id, "Registration request rejected and removed.")


def remove_user_records(user_id, message):
    connection = get_db_connection()
    try:
        # Remove dependent rows first so the account can be deleted cleanly.
        # Payments are linked both to a user and to a subscription.
        connection.execute("DELETE FROM payments WHERE user_id = ?", (user_id,))
        subscription_rows = connection.execute("SELECT id FROM subscriptions WHERE user_id = ?", (user_id,)).fetchall()
        subscription_ids = [row["id"] for row in subscription_rows]
        if subscription_ids:
            marks = ",".join("?" for _ in subscription_ids)
            connection.execute(f"DELETE FROM payments WHERE subscription_id IN ({marks})", subscription_ids)
            connection.execute(f"DELETE FROM subscriptions WHERE id IN ({marks})", subscription_ids)
        for table, column in [("feedback", "user_id"), ("detection_items", "detection_id"), ("pollution_items", "pollution_record_id")]:
            # Items are deleted via their parent IDs below; direct user tables handled separately.
            pass
        detection_rows = connection.execute("SELECT id FROM detections WHERE user_id = ?", (user_id,)).fetchall()
        detection_ids = [row["id"] for row in detection_rows]
        if detection_ids:
            marks = ",".join("?" for _ in detection_ids)
            connection.execute(f"DELETE FROM detection_items WHERE detection_id IN ({marks})", detection_ids)
            connection.execute(f"DELETE FROM detections WHERE id IN ({marks})", detection_ids)
        pollution_rows = connection.execute("SELECT id FROM pollution_records WHERE user_id = ?", (user_id,)).fetchall()
        pollution_ids = [row["id"] for row in pollution_rows]
        if pollution_ids:
            marks = ",".join("?" for _ in pollution_ids)
            connection.execute(f"DELETE FROM pollution_items WHERE pollution_record_id IN ({marks})", pollution_ids)
            connection.execute(f"DELETE FROM pollution_records WHERE id IN ({marks})", pollution_ids)
        for table in ("feedback",):
            exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
            if exists:
                connection.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
        connection.execute("DELETE FROM users WHERE id = ? AND role IN ('developer', 'organization')", (user_id,))
        connection.commit()
        flash(message)
    except sqlite3.Error:
        connection.rollback()
        flash("Could not remove the account because of a database error.")
    finally:
        connection.close()
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/users/<int:user_id>/remove", methods=["POST"])
def admin_remove_user(user_id):
    if not admin_required():
        flash("Please sign in with an admin account.")
        return redirect(url_for("admin_login"))
    connection = get_db_connection()
    user = connection.execute(
        "SELECT id, role FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    connection.close()
    if not user or user["role"] == "admin":
        flash("Admin accounts cannot be removed from this page.")
        return redirect(url_for("admin_dashboard"))
    if session.get("user_id") == user_id:
        flash("You cannot remove the currently signed-in account.")
        return redirect(url_for("admin_dashboard"))
    return remove_user_records(user_id, "User or organization account and its related records were removed.")


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    flash("You have signed out.")
    return redirect(url_for("admin_login"))


# ---------------- ORGANIZATION TEAM, DOWNLOADS AND ADMIN QUOTAS ----------------

ORGANIZATION_TEAM_PAGE = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Organization Team | MarineVision</title>
<style>
*{box-sizing:border-box}body{font-family:Arial,sans-serif;background:#eef5f7;color:#173b46;margin:0}
.top{background:#075b68;color:white;padding:18px 28px;display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap}
.top h1{margin:0}.top nav{display:flex;gap:16px;flex-wrap:wrap}.top a{color:white;text-decoration:none}
.wrap{max-width:1120px;margin:28px auto;padding:0 20px}.intro{line-height:1.6;color:#526873}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
.panel{background:white;padding:22px;border-radius:12px;margin:16px 0;box-shadow:0 3px 12px #0001}
.panel h2{margin-top:0;color:#075b68}.detail{margin:10px 0;line-height:1.5}.detail b{display:inline-block;min-width:150px}
.message{padding:11px;background:#e0f2fe;border-radius:7px;margin:10px 0}.muted{color:#64777b}
table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:11px;border-bottom:1px solid #dbe5e9}th{background:#f2f8f7}
.table-wrap{overflow-x:auto}input,button,.button{padding:10px 13px;border:1px solid #ccd5d9;border-radius:6px;font:inherit}
input{display:block;width:100%;margin:7px 0 14px}button,.button{background:#075985;color:white;border:0;cursor:pointer;text-decoration:none;display:inline-block}
.secondary{background:#64777b}.danger{background:#a52a2a}.actions{display:flex;gap:6px;flex-wrap:wrap}.actions form{display:inline}
</style></head><body>
<header class="top"><h1>Organization Team</h1><nav><a href="{{ url_for('dashboard') }}">Dashboard</a><a href="{{ url_for('detect') }}">Organism Detection</a><a href="{{ url_for('pollution_detect') }}">Pollution Detection</a><a href="{{ url_for('detection_history') }}">Organism History</a><a href="{{ url_for('pollution_history') }}">Pollution History</a><a href="{{ url_for('logout') }}">Logout</a></nav></header>
<main class="wrap"><p class="intro">Organization information, team membership, and shared download usage.</p>
{% for message in get_flashed_messages() %}<p class="message">{{ message }}</p>{% endfor %}
<div class="grid">
<section class="panel"><h2>Organization Information</h2><p class="detail"><b>Organization Name</b>{{ organization_name }}</p><p class="detail"><b>Contact Number</b>{{ organization_contact }}</p><p class="detail"><b>Address</b>{{ organization_address }}</p><p class="detail"><b>Registration Details</b>{{ organization_registration }}</p></section>
<section class="panel"><h2>Organization Head</h2><p class="detail"><b>Name</b>{{ owner['name'] or 'Not Provided' }}</p><p class="detail"><b>Email Address</b>{{ owner['email'] or 'Not Provided' }}</p><p class="detail"><b>Contact Number</b>{{ owner['contact_number'] or 'Not Provided' }}</p><p class="detail"><b>Address</b>{{ owner['address'] or 'Not Provided' }}</p><p class="detail"><b>Profession</b>{{ owner_profession }}</p><p class="detail"><b>Designation</b>{{ owner['designation'] or 'Not Provided' }}</p></section>
<section class="panel"><h2>Subscription &amp; Downloads</h2><p class="detail"><b>Plan</b>{{ subscription_plan }}</p><p class="detail"><b>Status</b>{{ subscription_status }}</p><p class="detail"><b>Expiry Date</b>{{ subscription_expiry }}</p><p class="detail"><b>Total Download Quota</b>{{ usage.limit }}</p><p class="detail"><b>Downloads Used</b>{{ usage.used }}</p><p class="detail"><b>Remaining Downloads</b>{{ usage.remaining }}</p></section>
</div>
<section class="panel"><h2>Team Members</h2><div class="table-wrap"><table><thead><tr><th>Member Name</th><th>Username</th><th>Email Address</th><th>Contact</th><th>Address</th><th>Profession</th><th>Designation</th><th>Qualification</th><th>Areas of Interest</th><th>Role</th><th>Account Status</th><th>Date Joined</th>{% if is_owner %}<th>Actions</th>{% endif %}</tr></thead><tbody>
{% for member in members %}<tr><td>{{ member['name'] or 'Not Provided' }}</td><td>{{ member['username'] or 'Not Provided' }}</td><td>{{ member['email'] or 'Not Provided' }}</td><td>{{ member['contact_number'] or 'Not Provided' }}</td><td>{{ member['address'] or 'Not Provided' }}</td><td>{{ member['profession_other'] if member['profession'] == 'Other' and member['profession_other'] else member['profession'] or 'Not Provided' }}</td><td>{{ member['designation'] or 'Not Provided' }}</td><td>{{ member['highest_qualification'] or 'Not Provided' }}</td><td>{{ member['areas_of_interest'] or 'Not Provided' }}</td><td>{{ member['role'] }}</td><td>{{ 'Active' if member['approval_status'] == 'Approved' else member['approval_status'] }}</td><td>{{ member['created_at'] or 'Not Provided' }}</td>
{% if is_owner %}<td><div class="actions"><form method="post" action="{{ url_for('organization_member_action', member_id=member['id']) }}"><button type="submit" name="action" value="{{ 'deactivate' if member['approval_status'] == 'Approved' else 'activate' }}">{{ 'Deactivate' if member['approval_status'] == 'Approved' else 'Activate' }}</button></form><form method="post" action="{{ url_for('organization_member_action', member_id=member['id']) }}" onsubmit="return confirm('Remove this member from the team? Their history will be retained.')"><button class="danger" type="submit" name="action" value="remove">Remove</button></form></div></td>{% endif %}</tr>
{% else %}<tr><td colspan="{{ 13 if is_owner else 12 }}" class="muted">No team members have been added.</td></tr>{% endfor %}
</tbody></table></div></section>
{% if is_owner %}<p><a class="button" href="#add-member">Add Team Member</a></p><section class="panel" id="add-member"><h2>Add Team Member</h2><form method="post" action="{{ url_for('organization_add_member') }}"><label>Full name</label><input name="name" required><label>Email address</label><input name="email" type="email" required><label>Contact number</label><input name="contact_number" type="tel" required><label>Address</label><input name="address" required><label>Profession</label><select name="profession" id="member-profession" required><option value="">Select profession</option><option>Student</option><option>Researcher</option><option>Marine Biologist</option><option>Environmental Scientist</option><option>Fisheries Professional</option><option>Other</option></select><div id="member-other-wrap" hidden><label>Specify profession</label><input name="profession_other" id="member-profession-other"></div><label>Designation</label><input name="designation" required><label>Highest qualification (optional)</label><input name="highest_qualification"><label>Areas of interest (optional)</label><input name="areas_of_interest"><label>Username</label><input name="username" required><label>Temporary password (minimum 8 characters)</label><input name="password" type="password" minlength="8" required><button type="submit">Create Team Account</button></form><script>const memberProfession=document.getElementById('member-profession'), memberOtherWrap=document.getElementById('member-other-wrap'), memberOther=document.getElementById('member-profession-other');memberProfession.addEventListener('change',()=>{const active=memberProfession.value==='Other';memberOtherWrap.hidden=!active;memberOther.required=active;});</script><p class="muted">Team members can view this page but cannot manage membership or subscriptions.</p></section>{% endif %}
</main></body></html>
"""

ADMIN_DOWNLOAD_LIMITS_PAGE = """
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Download Limits | MarineVision Admin</title><style>body{font-family:Arial;background:#eef5f7;color:#18313b;margin:0;padding:24px}.wrap{max-width:1200px;margin:auto}.panel{background:white;padding:18px;border-radius:10px;margin:16px 0;overflow:auto}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd;white-space:nowrap}input{width:90px;padding:7px}button{padding:8px 12px;background:#075985;color:white;border:0;border-radius:5px;cursor:pointer}.danger{background:#b91c1c}</style></head><body><main class="wrap"><h1>Organization Download Limits</h1><p>Review downloads and change each paid subscription's quota or block/unblock it.</p><p><a href="{{ url_for('admin_dashboard') }}">Back to Admin Dashboard</a></p>{% for message in get_flashed_messages() %}<p>{{ message }}</p>{% endfor %}<section class="panel"><table><tr><th>Organization</th><th>Plan</th><th>Payment</th><th>Expires</th><th>Downloads used</th><th>Limit</th><th>Status</th><th>Update</th></tr>{% for row in rows %}<tr><td>{{ row.organization_name }} (ID {{ row.user_id }})</td><td>{{ row.duration_months }} month(s)</td><td>{{ row.payment_status }}</td><td>{{ row.expires_at or 'Not recorded' }}</td><td>{{ row.used }}</td><td>{{ row.download_limit }}</td><td>{% if row.is_blocked or row.used >= row.download_limit %}Blocked / quota reached{% else %}Active{% endif %}</td><td><form method="post" action="{{ url_for('admin_update_download_limit', subscription_id=row.id) }}"><input type="number" name="download_limit" min="0" value="{{ row.download_limit }}" required><button type="submit" name="action" value="limit">Save limit</button><button type="submit" name="action" value="{{ 'unblock' if row.is_blocked else 'block' }}">{{ 'Unblock' if row.is_blocked else 'Block' }}</button></form></td></tr>{% endfor %}</table></section></main></body></html>
"""


def organization_owner_required():
    if "user_id" not in session or session.get("role") != "organization":
        flash("Only the organization owner can manage team members and subscriptions.")
        return False
    return True


@app.route("/organization/team")
def organization_team():
    connection = get_db_connection()
    try:
        if "user_id" not in session:
            flash("Please sign in first.")
            return redirect(url_for("login"))
        current_user = connection.execute("""
            SELECT id, name, email, role, organization_id, contact_number, address,
                   profession, profession_other, designation, organization_name
            FROM users WHERE id = ?
        """, (session["user_id"],)).fetchone()
        if not current_user:
            session.clear()
            flash("Please sign in again.")
            return redirect(url_for("login"))

        if current_user["role"] == "organization":
            organization_id = current_user["id"]
            owner = current_user
            is_owner = True
        elif current_user["role"] == "org_member" and current_user["organization_id"]:
            organization_id = current_user["organization_id"]
            owner = connection.execute("""
                SELECT id, name, email, role, contact_number, address, profession,
                       profession_other, designation, organization_name FROM users
                WHERE id = ? AND role = 'organization'
            """, (organization_id,)).fetchone()
            is_owner = False
        else:
            owner = None
            organization_id = None
            is_owner = False

        if not owner:
            flash("This page is available only to an organization owner or its team members.")
            return redirect(url_for("dashboard"))

        members = connection.execute("""
            SELECT id, name, username, email, role, approval_status, created_at,
                   contact_number, address, profession, profession_other, designation,
                   highest_qualification, areas_of_interest
            FROM users
            WHERE organization_id = ? AND role = 'org_member'
              AND approval_status != 'Removed'
            ORDER BY id DESC
        """, (organization_id,)).fetchall()

        subscription = active_subscription_for_org(organization_id)
        if not subscription:
            subscription = connection.execute("""
                SELECT * FROM subscriptions WHERE user_id = ?
                ORDER BY id DESC LIMIT 1
            """, (organization_id,)).fetchone()
        if subscription:
            usage = organization_usage(organization_id, subscription)
            subscription_plan = f"{subscription['duration_months']} month(s)"
            subscription_status = subscription["payment_status"] or "Not Provided"
            expiry = subscription["expires_at"]
            if subscription_status == "Successful" and expiry:
                expired = connection.execute(
                    "SELECT datetime(?) <= datetime('now', 'localtime')", (expiry,)
                ).fetchone()[0]
                if expired:
                    subscription_status = "Expired"
            if subscription["is_blocked"]:
                subscription_status = "Blocked by administrator"
            subscription_expiry = expiry or "Not Provided"
        else:
            usage = {"limit": 0, "used": 0, "remaining": 0}
            subscription_plan = "Not Provided"
            subscription_status = "Not Provided"
            subscription_expiry = "Not Provided"

        organization_name = (owner["organization_name"] or
            (subscription["organization_name"] if subscription and subscription["organization_name"] else None) or
            "Not Provided")
        if owner["profession"] == "Other":
            owner_profession = owner["profession_other"] or "Other"
        else:
            owner_profession = owner["profession"] or "Not Provided"
        return render_template_string(
            ORGANIZATION_TEAM_PAGE,
            organization_name=organization_name,
            organization_contact=owner["contact_number"] or "Not Provided",
            organization_address=owner["address"] or "Not Provided",
            organization_registration="Not Provided",
            owner=owner,
            owner_profession=owner_profession,
            subscription_plan=subscription_plan,
            subscription_status=subscription_status,
            subscription_expiry=subscription_expiry,
            usage=usage,
            members=members,
            is_owner=is_owner,
        )
    finally:
        connection.close()


@app.route("/organization/team/add", methods=["POST"])
def organization_add_member():
    if not organization_owner_required():
        return redirect(url_for("dashboard"))
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    contact_number = request.form.get("contact_number", "").strip()
    address = request.form.get("address", "").strip()
    profession = request.form.get("profession", "").strip()
    profession_other = request.form.get("profession_other", "").strip()
    designation = request.form.get("designation", "").strip()
    highest_qualification = request.form.get("highest_qualification", "").strip()
    areas_of_interest = request.form.get("areas_of_interest", "").strip()
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    if (not all([name, email, username, contact_number, address, profession, designation])
            or len(password) < 8):
        flash("Enter the required member details and use a password of at least 8 characters.")
        return redirect(url_for("organization_team"))
    contact_digits = "".join(character for character in contact_number if character.isdigit())
    if (not re.fullmatch(r"\+?[0-9][0-9\s().-]{5,19}", contact_number)
            or not 7 <= len(contact_digits) <= 15
            or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email)):
        flash("Enter a valid email address and contact number (7 to 15 digits).")
        return redirect(url_for("organization_team"))
    if profession not in {"Student", "Researcher", "Marine Biologist", "Environmental Scientist", "Fisheries Professional", "Other"} or (profession == "Other" and not profession_other):
        flash("Select a profession and specify it when choosing Other.")
        return redirect(url_for("organization_team"))
    connection = get_db_connection()
    try:
        connection.execute("""
            INSERT INTO users (name, email, username, password, role, approval_status, organization_id,
                created_at, contact_number, address, profession, profession_other, designation,
                highest_qualification, areas_of_interest, organization_name)
            SELECT ?, ?, ?, ?, 'org_member', 'Approved', ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?,
                COALESCE((SELECT organization_name FROM users WHERE id = ?),
                    (SELECT organization_name FROM subscriptions WHERE user_id = ? ORDER BY id DESC LIMIT 1))
        """, (name, email, username, generate_password_hash(password), session["user_id"],
              contact_number, address, profession, profession_other or None, designation,
              highest_qualification or None, areas_of_interest or None,
              session["user_id"], session["user_id"]))
        connection.commit()
        flash("Team member account created. Share the username and temporary password securely.")
    except sqlite3.IntegrityError:
        flash("That email or username is already registered.")
    finally:
        connection.close()
    return redirect(url_for("organization_team"))


@app.route("/organization/team/member/<int:member_id>", methods=["POST"])
def organization_member_action(member_id):
    if not organization_owner_required():
        return redirect(url_for("dashboard"))
    action = request.form.get("action", "")
    status_by_action = {"activate": "Approved", "deactivate": "Deactivated", "remove": "Removed"}
    if action not in status_by_action:
        abort(400)
    connection = get_db_connection()
    try:
        cursor = connection.execute("""
            UPDATE users SET approval_status = ?
            WHERE id = ? AND organization_id = ? AND role = 'org_member'
              AND approval_status != 'Removed'
        """, (status_by_action[action], member_id, session["user_id"]))
        connection.commit()
        flash({
            "activate": "Team member activated.",
            "deactivate": "Team member deactivated.",
            "remove": "Team member removed. Existing detection and download history was retained.",
        }[action] if cursor.rowcount else "Team member not found in your organization.")
    finally:
        connection.close()
    return redirect(url_for("organization_team"))


def user_can_access_record(record_user_id, current_user_id, current_role, organization_id):
    if int(record_user_id) == int(current_user_id):
        return True
    return current_role == "organization" and organization_id == current_user_id and organization_owner_id(record_user_id) == current_user_id


@app.route("/download/<record_type>/<int:record_id>")
def download_detection_result(record_type, record_id):
    if "user_id" not in session:
        flash("Please sign in first.")
        return redirect(url_for("login"))
    if record_type not in ("organism", "pollution"):
        abort(404)
    owner_id = organization_owner_id(session["user_id"])
    table = "detections" if record_type == "organism" else "pollution_records"
    connection = get_db_connection()
    try:
        # Serialize quota checks and log writes so concurrent requests cannot
        # both consume the final download slot.
        if session.get("role") in ("organization", "org_member"):
            connection.execute("BEGIN IMMEDIATE")
        record = connection.execute(
            f"SELECT id, user_id, result_image FROM {table} WHERE id = ?",
            (record_id,)
        ).fetchone()
        if not record or not user_can_access_record(
            record["user_id"], session["user_id"], session.get("role"), owner_id
        ):
            abort(404)

        relative_path = record["result_image"]
        file_path = os.path.abspath(os.path.join(app.static_folder, relative_path))
        if not file_path.startswith(os.path.abspath(app.static_folder) + os.sep) or not os.path.isfile(file_path):
            flash("The result image file could not be found.")
            return redirect(url_for("dashboard"))

        if session.get("role") in ("organization", "org_member"):
            subscription = connection.execute("""
                SELECT * FROM subscriptions
                WHERE user_id = ? AND payment_status = 'Successful'
                  AND (expires_at IS NULL OR datetime(expires_at) > datetime('now', 'localtime'))
                ORDER BY id DESC LIMIT 1
            """, (owner_id,)).fetchone()
            if not subscription:
                flash("No active paid subscription. Contact the organization owner.")
                return redirect(url_for("dashboard"))
            used = connection.execute(
                "SELECT COUNT(*) FROM download_logs WHERE subscription_id = ?",
                (subscription["id"],)
            ).fetchone()[0]
            if subscription["is_blocked"] or used >= max(0, int(subscription["download_limit"] or 0)):
                flash("The download limit has been reached or access was blocked. Contact the organization owner/admin.")
                return redirect(url_for("dashboard"))
            connection.execute("""
                INSERT INTO download_logs
                (organization_id, user_id, subscription_id, record_type, record_id)
                VALUES (?, ?, ?, ?, ?)
            """, (owner_id, session["user_id"], subscription["id"], record_type, record_id))
            connection.commit()
    finally:
        connection.close()
    return send_file(file_path, as_attachment=True, download_name=f"marinevision_{record_type}_{record_id}.jpg")


@app.route("/admin/download-limits")
def admin_download_limits():
    if not admin_required():
        flash("Please sign in with an admin account.")
        return redirect(url_for("admin_login"))
    connection = get_db_connection()
    rows = connection.execute("""
        SELECT s.*, u.name AS organization_owner_name,
               (SELECT COUNT(*) FROM download_logs dl WHERE dl.subscription_id = s.id) AS used
        FROM subscriptions s JOIN users u ON u.id = s.user_id
        ORDER BY s.id DESC
    """).fetchall()
    connection.close()
    return render_template_string(ADMIN_DOWNLOAD_LIMITS_PAGE, rows=rows)


@app.route("/admin/download-limits/<int:subscription_id>", methods=["POST"])
def admin_update_download_limit(subscription_id):
    if not admin_required():
        flash("Please sign in with an admin account.")
        return redirect(url_for("admin_login"))
    try:
        limit = int(request.form.get("download_limit", ""))
        if limit < 0:
            raise ValueError
    except ValueError:
        flash("Download limit must be zero or a positive whole number.")
        return redirect(url_for("admin_download_limits"))
    connection = get_db_connection()
    action = request.form.get("action", "limit")
    if action not in {"limit", "block", "unblock"}:
        connection.close()
        abort(400)
    if action == "limit":
        cursor = connection.execute("UPDATE subscriptions SET download_limit = ? WHERE id = ?", (limit, subscription_id))
    else:
        cursor = connection.execute("UPDATE subscriptions SET is_blocked = ? WHERE id = ?", (1 if action == "block" else 0, subscription_id))
    connection.commit()
    connection.close()
    flash("Download quota updated." if cursor.rowcount else "Subscription not found.")
    return redirect(url_for("admin_download_limits"))


# ---------------- LOGOUT ----------------

@app.route("/logout")
def logout():
    session.clear()
    flash("You have signed out successfully.")
    return redirect(url_for("login"))


if __name__ == "__main__":
    create_table()
    app.run(debug=True)
