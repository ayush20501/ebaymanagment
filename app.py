from flask import Flask, render_template, request, jsonify, redirect, Response, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from jsonschema import validate
from typing import Optional, Dict, Any, List
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
import os
import base64
import time
import json
import re
import requests
import html
from pathlib import Path
import hashlib
from werkzeug.utils import secure_filename
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
from datetime import datetime, timedelta
from single_publish_logic import publish_item, is_user_active

app = Flask(__name__, static_folder='static', template_folder='templates')
load_dotenv()

app.secret_key = os.getenv("SECRET_KEY")
DB_URL = os.getenv("DB_URL")

EBAY_ENV = os.getenv("EBAY_ENV", "PRODUCTION")
BASE = os.getenv("EBAY_BASE")
AUTH = os.getenv("EBAY_AUTH")
TOKEN = os.getenv("EBAY_TOKEN_URL")
API = os.getenv("EBAY_API")
MARKETPLACE_ID = os.getenv("EBAY_MARKETPLACE_ID")
LANG = os.getenv("EBAY_LANG")

CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")
RU_NAME = os.getenv("EBAY_RU_NAME")

EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = os.getenv("EMAIL_PORT")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

db_pool = psycopg2.pool.SimpleConnectionPool(
    1, 20, DB_URL
)

SCOPES = " ".join([
    "https://api.ebay.com/oauth/api_scope",
    "https://api.ebay.com/oauth/api_scope/sell.inventory",
    "https://api.ebay.com/oauth/api_scope/sell.account",
])

PROFILE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "UserProfile",
    "type": "object",
    "required": ["email", "first_name", "last_name", "address_line1", "city", "postal_code", "country"],
    "properties": {
        "email": {"type": "string", "format": "email", "maxLength": 255},
        "first_name": {"type": "string", "minLength": 1, "maxLength": 50},
        "last_name": {"type": "string", "minLength": 1, "maxLength": 50},
        "address_line1": {"type": "string", "minLength": 1, "maxLength": 100},
        "address_line2": {"type": "string", "maxLength": 100},
        "city": {"type": "string", "minLength": 1, "maxLength": 50},
        "state": {"type": "string", "maxLength": 50},
        "postal_code": {"type": "string", "minLength": 1, "maxLength": 20},
        "country": {"type": "string", "minLength": 2, "maxLength": 2},
        "phone": {"type": "string", "maxLength": 20}
    },
    "additionalProperties": False
}

SIGNUP_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "SignupSchema",
    "type": "object",
    "required": ["email", "password", "first_name", "last_name"],
    "properties": {
        "email": {"type": "string", "format": "email", "maxLength": 255},
        "password": {"type": "string", "minLength": 8, "maxLength": 128},
        "first_name": {"type": "string", "minLength": 1, "maxLength": 50},
        "last_name": {"type": "string", "minLength": 1, "maxLength": 50}
    },
    "additionalProperties": False
}

def get_db_connection():
    return db_pool.getconn()

def close_db_connection(conn):
    db_pool.putconn(conn)

def _b64_basic():
    return "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()

def get_user_profile(user_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT email, first_name, last_name, address_line1, address_line2, city, state, postal_code, country, phone
                FROM user_profiles
                WHERE user_id = %s
            """, (user_id,))
            profile = c.fetchone()
            if profile:
                return {
                    'email': profile[0],
                    'first_name': profile[1],
                    'last_name': profile[2],
                    'address_line1': profile[3],
                    'address_line2': profile[4],
                    'city': profile[5],
                    'state': profile[6],
                    'postal_code': profile[7],
                    'country': profile[8],
                    'phone': profile[9]
                }
            return None
    finally:
        close_db_connection(conn)

def get_user_tokens(user_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT access_token, refresh_token, access_expiry FROM user_tokens WHERE user_id = %s", (user_id,))
            tokens = c.fetchone()
            if tokens:
                return {
                    'access': tokens[0],
                    'refresh': tokens[1],
                    'access_expiry': tokens[2]
                }
            return {'access': None, 'refresh': None, 'access_expiry': None}
    finally:
        close_db_connection(conn)

def ensure_access_token(user_id):
    tokens = get_user_tokens(user_id)
    now = int(time.time())
    if tokens['access'] and tokens['access_expiry'] and now < tokens['access_expiry'] - 300:
        return tokens['access']
    
    if not tokens['refresh']:
        raise RuntimeError("No refresh token available")
    
    headers = {
        "Authorization": _b64_basic(),
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "refresh_token",
        "refresh_token": tokens['refresh'],
        "scope": SCOPES
    }
    r = requests.post(TOKEN, headers=headers, data=data, timeout=30)
    r.raise_for_status()
    resp = r.json()
    new_access = resp.get("access_token")
    expires_in = int(resp.get("expires_in", 7200))
    new_expiry = now + expires_in
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute(
                "UPDATE user_tokens SET access_token = %s, access_expiry = %s WHERE user_id = %s",
                (new_access, new_expiry, user_id)
            )
            conn.commit()
        return new_access
    finally:
        close_db_connection(conn)

def save_user_listing(user_id, listing_data):
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO user_listings (
                    user_id, listing_id, offer_id, sku, title, price_value,
                    price_currency, quantity, condition, category_id,
                    category_name, marketplace_id, view_url, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                user_id,
                listing_data['listing_id'],
                listing_data['offer_id'],
                listing_data['sku'],
                listing_data['title'],
                listing_data['price_value'],
                listing_data['price_currency'],
                listing_data['quantity'],
                listing_data['condition'],
                listing_data['category_id'],
                listing_data['category_name'],
                listing_data['marketplace_id'],
                listing_data['view_url'],
                int(time.time())
            ))
            conn.commit()
    finally:
        close_db_connection(conn)

def update_listing_count():
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("UPDATE stats SET total_listings = total_listings + 1 WHERE id = 1")
            conn.commit()
    finally:
        close_db_connection(conn)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    body = request.get_json(force=True) or {}
    try:
        validate(instance=body, schema=SIGNUP_SCHEMA)
    except Exception as e:
        return jsonify({"error": "Invalid signup data", "details": str(e)}), 400
    
    email = body['email']
    password = body['password']
    first_name = body['first_name']
    last_name = body['last_name']
    
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT id FROM users WHERE email = %s", (email,))
            if c.fetchone():
                return jsonify({"error": "Email already exists"}), 400
            
            verification_code = str(random.randint(100000, 999999))
            c.execute("""
                INSERT INTO users (email, password, first_name, last_name, verification_code, is_active)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (email, hashed_password, first_name, last_name, verification_code, False))
            user_id = c.fetchone()[0]
            conn.commit()
            
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = email
            msg['Subject'] = 'Verify Your Email'
            body_text = f'Your verification code is: {verification_code}'
            msg.attach(MIMEText(body_text, 'plain'))
            
            with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg)
                
            return jsonify({"message": "User created, verification code sent", "user_id": user_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        close_db_connection(conn)

@app.route('/verify-email', methods=['POST'])
def verify_email():
    body = request.get_json(force=True) or {}
    user_id = body.get('user_id')
    code = body.get('code')
    
    if not user_id or not code:
        return jsonify({"error": "user_id and code are required"}), 400
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT verification_code, is_active FROM users WHERE id = %s", (user_id,))
            user = c.fetchone()
            if not user:
                return jsonify({"error": "User not found"}), 404
            if user[1]:
                return jsonify({"error": "User already verified"}), 400
            if user[0] != code:
                return jsonify({"error": "Invalid verification code"}), 400
            
            c.execute("UPDATE users SET is_active = TRUE, verification_code = NULL WHERE id = %s", (user_id,))
            conn.commit()
            return jsonify({"message": "Email verified successfully"}), 200
    finally:
        close_db_connection(conn)

@app.route('/login', methods=['POST'])
def login():
    body = request.get_json(force=True) or {}
    email = body.get('email')
    password = body.get('password')
    
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT id, password, is_active FROM users WHERE email = %s", (email,))
            user = c.fetchone()
            if not user:
                return jsonify({"error": "Invalid email or password"}), 401
            user_id, hashed_password, is_active = user
            if not is_active:
                return jsonify({"error": "Account not verified"}), 403
            if not check_password_hash(hashed_password, password):
                return jsonify({"error": "Invalid email or password"}), 401
            
            session['user_id'] = user_id
            return jsonify({"message": "Logged in successfully", "user_id": user_id}), 200
    finally:
        close_db_connection(conn)

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({"message": "Logged out successfully"}), 200

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return jsonify({"error": "Please log in first"}), 401
    
    user_id = session['user_id']
    if not is_user_active(user_id, db_pool):
        return jsonify({"error": "Account is inactive"}), 403
    
    if request.method == 'GET':
        profile = get_user_profile(user_id)
        if not profile:
            return jsonify({"error": "Profile not found"}), 404
        return jsonify(profile), 200
    
    body = request.get_json(force=True) or {}
    try:
        validate(instance=body, schema=PROFILE_SCHEMA)
    except Exception as e:
        return jsonify({"error": "Invalid profile data", "details": str(e)}), 400
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO user_profiles (
                    user_id, email, first_name, last_name, address_line1, address_line2,
                    city, state, postal_code, country, phone
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    email = EXCLUDED.email,
                    first_name = EXCLUDED.first_name,
                    last_name = EXCLUDED.last_name,
                    address_line1 = EXCLUDED.address_line1,
                    address_line2 = EXCLUDED.address_line2,
                    city = EXCLUDED.city,
                    state = EXCLUDED.state,
                    postal_code = EXCLUDED.postal_code,
                    country = EXCLUDED.country,
                    phone = EXCLUDED.phone
            """, (
                user_id, body['email'], body['first_name'], body['last_name'],
                body['address_line1'], body.get('address_line2'), body['city'],
                body.get('state'), body['postal_code'], body['country'], body.get('phone')
            ))
            conn.commit()
            return jsonify({"message": "Profile updated successfully"}), 200
    finally:
        close_db_connection(conn)

@app.route('/ebay-auth-url', methods=['GET'])
def ebay_auth_url():
    if 'user_id' not in session:
        return jsonify({"error": "Please log in first"}), 401
    
    if not is_user_active(session['user_id'], db_pool):
        return jsonify({"error": "Account is inactive"}), 403
    
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": RU_NAME,
        "response_type": "code",
        "scope": SCOPES,
        "prompt": "login"
    }
    query = "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())
    return jsonify({"url": f"{AUTH}?{query}"}), 200

@app.route('/ebay-callback', methods=['GET'])
def ebay_callback():
    code = request.args.get('code')
    expires = request.args.get('expires_in')
    if not code:
        return jsonify({"error": "No authorization code provided"}), 400
    
    if 'user_id' not in session:
        return jsonify({"error": "Please log in first"}), 401
    
    user_id = session['user_id']
    if not is_user_active(user_id, db_pool):
        return jsonify({"error": "Account is inactive"}), 403
    
    headers = {
        "Authorization": _b64_basic(),
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": RU_NAME
    }
    try:
        r = requests.post(TOKEN, headers=headers, data=data, timeout=30)
        r.raise_for_status()
        resp = r.json()
        access_token = resp.get("access_token")
        refresh_token = resp.get("refresh_token")
        expires_in = int(resp.get("expires_in", 7200))
        access_expiry = int(time.time()) + expires_in
        
        conn = get_db_connection()
        try:
            with conn.cursor() as c:
                c.execute("""
                    INSERT INTO user_tokens (user_id, access_token, refresh_token, access_expiry)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        access_token = EXCLUDED.access_token,
                        refresh_token = EXCLUDED.refresh_token,
                        access_expiry = EXCLUDED.access_expiry
                """, (user_id, access_token, refresh_token, access_expiry))
                conn.commit()
            return jsonify({"message": "eBay tokens saved successfully"}), 200
        finally:
            close_db_connection(conn)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to exchange code: {str(e)}"}), 500

@app.route("/publish-item", methods=["POST"])
def publish_item_endpoint():
    body = request.get_json(force=True) or {}
    result, status_code = publish_item(
        session=session,
        body=body,
        db_pool=db_pool,
        get_user_profile=get_user_profile,
        get_user_tokens=get_user_tokens,
        ensure_access_token=ensure_access_token,
        save_user_listing=save_user_listing,
        update_listing_count=update_listing_count
    )
    return jsonify(result), status_code

@app.route('/listings', methods=['GET'])
def get_listings():
    if 'user_id' not in session:
        return jsonify({"error": "Please log in first"}), 401
    
    user_id = session['user_id']
    if not is_user_active(user_id, db_pool):
        return jsonify({"error": "Account is inactive"}), 403
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT listing_id, offer_id, sku, title, price_value, price_currency,
                       quantity, condition, category_id, category_name, marketplace_id,
                       view_url, created_at
                FROM user_listings
                WHERE user_id = %s
                ORDER BY created_at DESC
            """, (user_id,))
            listings = []
            for row in c.fetchall():
                listings.append({
                    'listing_id': row[0],
                    'offer_id': row[1],
                    'sku': row[2],
                    'title': row[3],
                    'price': {'value': row[4], 'currency': row[5]},
                    'quantity': row[6],
                    'condition': row[7],
                    'category_id': row[8],
                    'category_name': row[9],
                    'marketplace_id': row[10],
                    'view_url': row[11],
                    'created_at': row[12]
                })
            return jsonify({"listings": listings}), 200
    finally:
        close_db_connection(conn)

@app.route('/stats', methods=['GET'])
def get_stats():
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT total_listings FROM stats WHERE id = 1")
            total_listings = c.fetchone()[0] or 0
            return jsonify({"total_listings": total_listings}), 200
    finally:
        close_db_connection(conn)

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    if not all([CLIENT_ID, CLIENT_SECRET, RU_NAME, DB_URL]):
        print("Missing required environment variables (CLIENT_ID, CLIENT_SECRET, RU_NAME, or DB_URL)")
        exit(1)
    
    app.run(host="127.0.0.1", port=5000, debug=True)
