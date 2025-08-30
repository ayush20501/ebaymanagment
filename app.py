import os
import base64
import time
import json
import re
import requests
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, redirect, Response, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from jsonschema import validate
from typing import Optional, Dict, Any, List
import html
from pathlib import Path
from collections import Counter
from openai import OpenAI
import hashlib
from werkzeug.utils import secure_filename
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.getenv("SECRET_KEY")
DB_URL = os.getenv("DB_URL")

# Database connection pool
db_pool = psycopg2.pool.SimpleConnectionPool(1, 20, DB_URL)

# Import the eBay Blueprint
from ebay_routes import ebay_bp
app.register_blueprint(ebay_bp)

# Schemas
PROFILE_SCHEMA = {
    "type": "object",
    "required": ["address_line1", "city", "postal_code", "country"],
    "properties": {
        "address_line1": {"type": "string", "minLength": 1, "maxLength": 200},
        "city": {"type": "string", "minLength": 1, "maxLength": 100},
        "postal_code": {"type": "string", "minLength": 1, "maxLength": 20},
        "country": {"type": "string", "enum": ["GB"]},
        "profile_pic_url": {"type": "string", "format": "uri", "pattern": "^https://.*"}
    }
}

# Database initialization
def init_db():
    conn = db_pool.getconn()
    try:
        with conn.cursor() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id INTEGER PRIMARY KEY,
                    address_line1 TEXT NOT NULL,
                    city TEXT NOT NULL,
                    postal_code TEXT NOT NULL,
                    country TEXT NOT NULL DEFAULT 'GB',
                    profile_pic_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS ebay_tokens (
                    user_id INTEGER PRIMARY KEY,
                    access_token TEXT,
                    refresh_token TEXT,
                    expires_at DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS listing_counts (
                    id SERIAL PRIMARY KEY,
                    total_count INTEGER DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            c.execute("INSERT INTO listing_counts (id, total_count) VALUES (1, 0) ON CONFLICT DO NOTHING")

            admin_email = os.getenv("ADMIN_EMAIL")
            admin_password = os.getenv("ADMIN_PASSWORD")

            if admin_email and admin_password:
                c.execute("SELECT id FROM users WHERE email = %s", (admin_email.lower(),))
                existing_admin = c.fetchone()

                if not existing_admin:
                    c.execute("""
                        INSERT INTO users (email, password_hash, is_active)
                        VALUES (%s, %s, TRUE)
                        RETURNING id
                    """, (admin_email.lower(), generate_password_hash(admin_password)))
                    admin_id = c.fetchone()[0]

                    # Create default admin profile
                    c.execute("""
                        INSERT INTO user_profiles
                        (user_id, address_line1, city, postal_code, country, updated_at)
                        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """, (admin_id, "Admin Address", "Admin City", "ADMIN", "GB"))

                    print(f"Admin user created with ID: {admin_id}")
                else:
                    # Check if admin has a profile, if not create one
                    c.execute("SELECT user_id FROM user_profiles WHERE user_id = %s", (existing_admin[0],))
                    admin_profile = c.fetchone()

                    if not admin_profile:
                        c.execute("""
                            INSERT INTO user_profiles
                            (user_id, address_line1, city, postal_code, country, updated_at)
                            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        """, (existing_admin[0], "Admin Address", "Admin City", "ADMIN", "GB"))
                        print(f"Admin profile created for existing admin with ID: {existing_admin[0]}")
                    else:
                        print(f"Admin user already exists with ID: {existing_admin[0]} and has profile")

            conn.commit()
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        raise
    finally:
        db_pool.putconn(conn)

init_db()

def init_otp_table():
    conn = db_pool.getconn()
    try:
        with conn.cursor() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS otps (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    otp TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            conn.commit()
    except psycopg2.Error as e:
        print(f"Database error creating OTP table: {e}")
        raise
    finally:
        db_pool.putconn(conn)

init_otp_table()

def init_user_listings_table():
    conn = db_pool.getconn()
    try:
        with conn.cursor() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS user_listings (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    listing_id TEXT NOT NULL,
                    offer_id TEXT,
                    sku TEXT,
                    title TEXT NOT NULL,
                    price_value DECIMAL(10,2),
                    price_currency TEXT DEFAULT 'GBP',
                    quantity INTEGER,
                    condition TEXT,
                    category_id TEXT,
                    category_name TEXT,
                    marketplace_id TEXT,
                    view_url TEXT,
                    status TEXT DEFAULT 'ACTIVE',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    UNIQUE(user_id, listing_id)
                )
            """)
            conn.commit()
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        raise
    finally:
        db_pool.putconn(conn)

init_user_listings_table()

# Database helper functions
def get_db_connection():
    conn = db_pool.getconn()
    conn.autocommit = False
    return conn

def close_db_connection(conn):
    db_pool.putconn(conn)

def is_user_active(user_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT is_active FROM users WHERE id = %s", (user_id,))
            user = c.fetchone()
            return user and user[0] == True
    finally:
        close_db_connection(conn)

def get_user_profile(user_id):
    if not is_user_active(user_id):
        return None
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT user_id, address_line1, city, postal_code, country,
                       profile_pic_url, created_at, updated_at
                FROM user_profiles WHERE user_id = %s
            """, (user_id,))
            profile = c.fetchone()
            if profile:
                return {
                    "user_id": profile[0],
                    "address_line1": profile[1],
                    "city": profile[2],
                    "postal_code": profile[3],
                    "country": profile[4],
                    "profile_pic_url": profile[5],
                    "created_at": profile[6],
                    "updated_at": profile[7]
                }
            return None
    finally:
        close_db_connection(conn)

def save_user_profile(user_id, profile_data):
    if not is_user_active(user_id):
        return False
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO user_profiles
                (user_id, address_line1, city, postal_code, country, profile_pic_url, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id) DO UPDATE
                SET address_line1 = EXCLUDED.address_line1,
                    city = EXCLUDED.city,
                    postal_code = EXCLUDED.postal_code,
                    country = EXCLUDED.country,
                    profile_pic_url = EXCLUDED.profile_pic_url,
                    updated_at = CURRENT_TIMESTAMP
            """, (user_id, profile_data['address_line1'], profile_data['city'],
                  profile_data['postal_code'], profile_data['country'], profile_data.get('profile_pic_url')))
            conn.commit()
            return True
    except psycopg2.Error:
        conn.rollback()
        return False
    finally:
        close_db_connection(conn)

def get_user_listings(user_id, limit=50, offset=0):
    """Get user's listings with pagination"""
    if not is_user_active(user_id):
        return []
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT listing_id, offer_id, sku, title, price_value, price_currency,
                       quantity, condition, category_name, view_url, status, created_at
                FROM user_listings
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """, (user_id, limit, offset))
            rows = c.fetchall()

            listings = []
            for row in rows:
                listings.append({
                    'listing_id': row[0],
                    'offer_id': row[1],
                    'sku': row[2],
                    'title': row[3],
                    'price_value': float(row[4]) if row[4] else 0,
                    'price_currency': row[5],
                    'quantity': row[6],
                    'condition': row[7],
                    'category_name': row[8],
                    'view_url': row[9],
                    'status': row[10],
                    'created_at': row[11].isoformat() if row[11] else None
                })
            return listings
    except psycopg2.Error as e:
        print(f"Error fetching user listings: {e}")
        return []
    finally:
        close_db_connection(conn)

def get_total_listings():
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT total_count FROM listing_counts WHERE id = 1")
            row = c.fetchone()
            return row[0] if row else 0
    finally:
        close_db_connection(conn)

def is_admin_user(user_id):
    """Check if a user is an admin by comparing their email with ADMIN_EMAIL"""
    admin_email = os.getenv("ADMIN_EMAIL")
    if not admin_email:
        print("No ADMIN_EMAIL environment variable set")
        return False

    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT email FROM users WHERE id = %s", (user_id,))
            user = c.fetchone()
            print(f"Checking admin status for user ID {user_id}: {user}")
            if user:
                is_admin = user[0].lower() == admin_email.lower()
                print(f"User email: {user[0]}, Admin email: {admin_email}, Is admin: {is_admin}")
                return is_admin
            return False
    finally:
        close_db_connection(conn)

# Static page endpoints
@app.route("/")
def index():
    return send_from_directory(app.template_folder, 'index.html')

@app.route("/profile.html")
def profile_page():
    return send_from_directory(app.template_folder, 'profile.html')

@app.route("/display-profile.html")
def display_profile_page():
    return send_from_directory(app.template_folder, 'display-profile.html')

@app.route("/ebay-auth.html")
def ebay_auth_page():
    return send_from_directory(app.template_folder, 'ebay-auth.html')

@app.route("/dashboard.html")
def dashboard_page():
    return send_from_directory(app.template_folder, 'dashboard.html')

@app.route("/success.html")
def success_page():
    return send_from_directory(app.template_folder, 'success.html')

@app.route("/admin-portal.html")
def admin_portal_page():
    uid = session.get("user_id")
    if not uid:
        return redirect("/login.html")
    if not is_admin_user(uid):
        return jsonify({"error": "Admin access required"}), 403
    return send_from_directory(app.template_folder, 'admin-portal.html')

# Authentication endpoints
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT id, password_hash, is_active FROM users WHERE email = %s", (email,))
            user = c.fetchone()
            if not user:
                return jsonify({"error": "Invalid email or password"}), 401
            if not user[2]:
                return jsonify({"error": "Account is inactive"}), 403
            if check_password_hash(user[1], password):
                session["user_id"] = user[0]
                session["email"] = email

                is_admin = is_admin_user(user[0])
                session["is_admin"] = is_admin
                if is_admin:
                    return jsonify({
                        "status": "success",
                        "message": "Logged in successfully as admin",
                        "redirect": "/admin-portal.html",
                        "is_admin": True
                    })
                else:
                    return jsonify({
                        "status": "success",
                        "message": "Logged in successfully",
                        "redirect": "/dashboard.html",
                        "is_admin": False
                    })
            return jsonify({"error": "Invalid email or password"}), 401
    finally:
        close_db_connection(conn)

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "success", "message": "Logged out successfully"})

# OTP and signup endpoints
otp_store = {}
pending_users = {}

def generate_otp():
    """Generate a 6-digit OTP"""
    return ''.join([str(random.randint(0, 9)) for _ in range(6)])

def send_otp_email(email, otp):
    """Send OTP via email"""
    SMTP_SERVER = os.getenv("EMAIL_HOST")
    SMTP_PORT = 587
    EMAIL_ADDRESS = os.getenv("EMAIL_USER")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASS")

    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = email
        msg['Subject'] = "Your ListFast.ai Verification Code"

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center;">
                <h1 style="color: white; margin: 0;">ListFast.ai</h1>
            </div>
            <div style="padding: 40px 30px; background: #f9f9f9;">
                <h2 style="color: #333; margin-bottom: 20px;">Welcome to ListFast.ai!</h2>
                <p style="color: #666; font-size: 16px; line-height: 1.6;">
                    Thank you for signing up. Please use the verification code below to complete your registration:
                </p>
                <div style="background: white; padding: 30px; border-radius: 10px; text-align: center; margin: 30px 0;">
                    <div style="font-size: 32px; font-weight: bold; color: #667eea; letter-spacing: 8px; font-family: monospace;">
                        {otp}
                    </div>
                    <p style="color: #888; font-size: 14px; margin-top: 15px;">
                        This code will expire in 10 minutes
                    </p>
                </div>
                <p style="color: #666; font-size: 14px;">
                    If you didn't request this code, please ignore this email.
                </p>
            </div>
            <div style="background: #333; padding: 20px; text-align: center;">
                <p style="color: #999; margin: 0; font-size: 12px;">
                    © 2025 ListFast.ai. All rights reserved.
                </p>
            </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(body, 'html'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_ADDRESS, email, text)
        server.quit()

        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT id FROM users WHERE email = %s", (email,))
            existing_user = c.fetchone()
            if existing_user:
                return jsonify({"error": "User with this email already exists"}), 400
            otp = generate_otp()
            otp_store[email] = {
                'otp': otp,
                'password': password,
                'timestamp': datetime.now(),
                'attempts': 0
            }

            if send_otp_email(email, otp):
                return jsonify({'message': 'Verification code sent to your email'}), 200
            else:
                if email in otp_store:
                    del otp_store[email]
                return jsonify({"error": "Failed to send verification email"}), 500

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return jsonify({"error": "Registration failed"}), 500
    finally:
        close_db_connection(conn)

@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        submitted_otp = data.get('otp', '')

        if not email or not submitted_otp:
            return jsonify({'error': 'Email and OTP are required'}), 400

        if email not in otp_store:
            return jsonify({'error': 'No verification code found. Please request a new one.'}), 400

        otp_data = otp_store[email]

        if datetime.now() - otp_data['timestamp'] > timedelta(minutes=10):
            del otp_store[email]
            return jsonify({'error': 'Verification code expired. Please request a new one.'}), 400

        if otp_data['attempts'] >= 5:
            del otp_store[email]
            return jsonify({'error': 'Too many incorrect attempts. Please request a new code.'}), 400

        if submitted_otp != otp_data['otp']:
            otp_data['attempts'] += 1
            return jsonify({'error': 'Invalid verification code'}), 400

        password = otp_data['password']
        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        try:
            with conn.cursor() as c:
                c.execute("""
                    INSERT INTO users (email, password_hash, is_active)
                    VALUES (%s, %s, TRUE)
                    RETURNING id
                """, (email, hashed_password))
                user_id = c.fetchone()[0]
                conn.commit()

                del otp_store[email]

                session["user_id"] = user_id
                session["email"] = email

                return jsonify({
                    'message': 'Email verified successfully! Account created.',
                    'user_id': user_id
                }), 200

        except psycopg2.errors.UniqueViolation:
            conn.rollback()
            if email in otp_store:
                del otp_store[email]
            return jsonify({'error': 'Email already exists'}), 400
        except psycopg2.Error as e:
            conn.rollback()
            print(f"Database error: {e}")
            return jsonify({'error': 'Account creation failed'}), 500
        finally:
            close_db_connection(conn)

    except Exception as e:
        print(f"OTP verification error: {e}")
        return jsonify({'error': 'Verification failed'}), 500

@app.route('/resend-otp', methods=['POST'])
def resend_otp():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()

        if not email:
            return jsonify({'error': 'Email is required'}), 400

        if email not in otp_store:
            return jsonify({'error': 'No pending verification for this email'}), 400

        otp_data = otp_store[email]
        if datetime.now() - otp_data['timestamp'] < timedelta(minutes=1):
            return jsonify({'error': 'Please wait before requesting a new code'}), 429

        new_otp = generate_otp()
        otp_store[email].update({
            'otp': new_otp,
            'timestamp': datetime.now(),
            'attempts': 0
        })

        if send_otp_email(email, new_otp):
            return jsonify({'message': 'New verification code sent'}), 200
        else:
            return jsonify({'error': 'Failed to send verification email'}), 500

    except Exception as e:
        print(f"Resend OTP error: {e}")
        return jsonify({'error': 'Failed to resend code'}), 500

# Profile endpoints
@app.route("/profile", methods=["POST"])
def create_profile():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_user_active(session["user_id"]):
        return jsonify({"error": "Account is inactive"}), 403
    data = request.get_json() or {}
    try:
        validate(instance=data, schema=PROFILE_SCHEMA)
    except Exception as e:
        return jsonify({"error": "Invalid profile data", "details": str(e)}), 400
    profile_data = {
        "address_line1": data["address_line1"].strip(),
        "city": data["city"].strip(),
        "postal_code": data["postal_code"].strip().upper(),
        "country": "GB",
        "profile_pic_url": data.get("profile_pic_url")
    }
    try:
        if save_user_profile(session["user_id"], profile_data):
            return jsonify({"status": "success", "message": "Profile created successfully"})
        return jsonify({"error": "Account is inactive"}), 403
    except Exception as e:
        return jsonify({"error": "Failed to save profile"}), 500

@app.route("/profile", methods=["GET"])
def get_profile():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_user_active(session["user_id"]):
        return jsonify({"error": "Account is inactive"}), 403
    profile = get_user_profile(session["user_id"])
    return jsonify({"profile": profile})

# User stats endpoints
@app.route("/user-stats")
def get_user_stats():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_user_active(session["user_id"]):
        return jsonify({"error": "Account is inactive"}), 403
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT COUNT(*) FROM user_listings WHERE user_id = %s", (session["user_id"],))
            listing_count = c.fetchone()[0]

            c.execute("""
                SELECT SUM(price_value * quantity) as total_value,
                       COUNT(CASE WHEN status = 'ACTIVE' THEN 1 END) as active_count
                FROM user_listings
                WHERE user_id = %s
            """, (session["user_id"],))
            stats_row = c.fetchone()

            total_value = float(stats_row[0]) if stats_row[0] else 0
            active_count = stats_row[1] if stats_row[1] else 0

            return jsonify({
                "total_listings": listing_count,
                "active_listings": active_count,
                "total_inventory_value": total_value,
                "email": session.get("email")
            })
    except psycopg2.Error as e:
        return jsonify({"error": "Failed to fetch stats"}), 500
    finally:
        close_db_connection(conn)

# Admin endpoints
@app.route("/admin/users")
def admin_get_users():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_admin_user(session["user_id"]):
        return jsonify({"error": "Admin access required"}), 403
    admin_email = os.getenv("ADMIN_EMAIL")
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT id, email, is_active, created_at,
                       (SELECT COUNT(*) FROM user_listings WHERE user_id = users.id) as listing_count
                FROM users
                WHERE email != %s
                ORDER BY created_at DESC
            """, (admin_email.lower(),))
            rows = c.fetchall()

            users = []
            for row in rows:
                users.append({
                    'id': row[0],
                    'email': row[1],
                    'is_active': row[2],
                    'created_at': row[3].isoformat() if row[3] else None,
                    'listing_count': row[4]
                })

            return jsonify({"users": users})
    except psycopg2.Error as e:
        return jsonify({"error": "Failed to fetch users"}), 500
    finally:
        close_db_connection(conn)

@app.route("/admin/users/<int:user_id>/toggle-status", methods=["POST"])
def admin_toggle_user_status(user_id):
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_admin_user(session["user_id"]):
        return jsonify({"error": "Admin access required"}), 403
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT is_active FROM users WHERE id = %s", (user_id,))
            user = c.fetchone()
            if not user:
                return jsonify({"error": "User not found"}), 404

            new_status = not user[0]
            c.execute("UPDATE users SET is_active = %s WHERE id = %s", (new_status, user_id))
            conn.commit()

            return jsonify({
                "status": "success",
                "message": f"User {'activated' if new_status else 'deactivated'} successfully",
                "is_active": new_status
            })
    except psycopg2.Error as e:
        conn.rollback()
        return jsonify({"error": "Failed to update user status"}), 500
    finally:
        close_db_connection(conn)

@app.route("/admin/stats")
def admin_get_stats():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_admin_user(session["user_id"]):
        return jsonify({"error": "Admin access required"}), 403
    admin_email = os.getenv("ADMIN_EMAIL")
    if not admin_email:
        return jsonify({"error": "ADMIN_EMAIL not set in environment variables"}), 500
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT COUNT(*) FROM users WHERE email != %s", (admin_email.lower(),))
            total_users = c.fetchone()[0]

            c.execute("SELECT COUNT(*) FROM users WHERE is_active = TRUE AND email != %s", (admin_email.lower(),))
            active_users = c.fetchone()[0]

            c.execute("""
                SELECT COUNT(*) FROM user_listings ul
                JOIN users u ON ul.user_id = u.id
                WHERE u.email != %s
            """, (admin_email.lower(),))
            total_listings = c.fetchone()[0]

            c.execute("""
                SELECT SUM(price_value * quantity) FROM user_listings ul
                JOIN users u ON ul.user_id = u.id
                WHERE u.email != %s
            """, (admin_email.lower(),))
            total_value = float(c.fetchone()[0] or 0)

            c.execute("""
                SELECT COUNT(*) FROM users
                WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
                AND email != %s
            """, (admin_email.lower(),))
            recent_registrations = c.fetchone()[0]

            c.execute("""
                SELECT COUNT(*) FROM user_listings ul
                JOIN users u ON ul.user_id = u.id
                WHERE ul.created_at >= CURRENT_DATE - INTERVAL '7 days'
                AND u.email != %s
            """, (admin_email.lower(),))
            recent_listings = c.fetchone()[0]

            return jsonify({
                "total_users": total_users,
                "active_users": active_users,
                "total_listings": total_listings,
                "total_value": total_value,
                "recent_registrations": recent_registrations,
                "recent_listings": recent_listings
            })
    except psycopg2.Error as e:
        return jsonify({"error": "Failed to fetch stats"}), 500
    finally:
        close_db_connection(conn)

@app.route("/admin/listings")
def admin_get_all_listings():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_admin_user(session["user_id"]):
        return jsonify({"error": "Admin access required"}), 403
    page = request.args.get('page', 1, type=int)
    limit = 50
    offset = (page - 1) * limit
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT ul.listing_id, ul.title, ul.price_value, ul.price_currency,
                       ul.quantity, ul.status, ul.created_at, u.email
                FROM user_listings ul
                JOIN users u ON ul.user_id = u.id
                ORDER BY ul.created_at DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))
            rows = c.fetchall()

            listings = []
            for row in rows:
                listings.append({
                    'listing_id': row[0],
                    'title': row[1],
                    'price_value': float(row[2]) if row[2] else 0,
                    'price_currency': row[3],
                    'quantity': row[4],
                    'status': row[5],
                    'created_at': row[6].isoformat() if row[6] else None,
                    'user_email': row[7]
                })

            return jsonify({
                "listings": listings,
                "page": page,
                "has_more": len(listings) == limit
            })
    except psycopg2.Error as e:
        return jsonify({"error": "Failed to fetch listings"}), 500
    finally:
        close_db_connection(conn)

@app.route("/debug-admin")
def debug_admin():
    """Debug endpoint to check admin status and create admin if needed"""
    admin_email = os.getenv("ADMIN_EMAIL")
    admin_password = os.getenv("ADMIN_PASSWORD")

    if not admin_email or not admin_password:
        return jsonify({
            "error": "ADMIN_EMAIL or ADMIN_PASSWORD not set in environment variables",
            "admin_email_set": bool(admin_email),
            "admin_password_set": bool(admin_password)
        })

    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT id, email, is_active FROM users WHERE email = %s", (admin_email.lower(),))
            admin_user = c.fetchone()

            if admin_user:
                return jsonify({
                    "admin_exists": True,
                    "admin_id": admin_user[0],
                    "admin_email": admin_user[1],
                    "admin_active": admin_user[2],
                    "admin_email_env": admin_email
                })
            else:
                # Create admin user
                c.execute("""
                    INSERT INTO users (email, password_hash, is_active)
                    VALUES (%s, %s, TRUE)
                    RETURNING id
                """, (admin_email.lower(), generate_password_hash(admin_password)))
                admin_id = c.fetchone()[0]

                # Create default admin profile
                c.execute("""
                    INSERT INTO user_profiles
                    (user_id, address_line1, city, postal_code, country, updated_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (admin_id, "Admin Address", "Admin City", "ADMIN", "GB"))

                conn.commit()

                return jsonify({
                    "admin_created": True,
                    "admin_id": admin_id,
                    "admin_email": admin_email,
                    "message": "Admin user created successfully"
                })
    except Exception as e:
        conn.rollback()
        return jsonify({"error": f"Database error: {str(e)}"})
    finally:
        close_db_connection(conn)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return send_from_directory(app.template_folder, 'index.html')

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Image upload endpoint
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 5 * 1024 * 1024
IMGBB_API_KEY = os.getenv('IMGBB_API_KEY')
IMGBB_UPLOAD_URL = 'https://api.imgbb.com/1/upload'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload-profile-image", methods=["POST"])
def upload_profile_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        encoded_image = base64.b64encode(file.read()).decode("utf-8")
        payload = {
            "key": IMGBB_API_KEY,
            "image": encoded_image,
            "name": secure_filename(file.filename)
        }

        response = requests.post(IMGBB_UPLOAD_URL, data=payload, timeout=30)
        result = response.json()

        if result.get("success"):
            return jsonify({
                "status": "success",
                "image_url": result["data"]["url"]
            })
        return jsonify({"error": "Upload failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    if not all([os.getenv("CLIENT_ID"), os.getenv("CLIENT_SECRET"), os.getenv("RU_NAME"), DB_URL]):
        print("Missing required environment variables (CLIENT_ID, CLIENT_SECRET, RU_NAME, or DB_URL)")
        exit(1)
    app.run(host="127.0.0.1", port=5000, debug=True)
