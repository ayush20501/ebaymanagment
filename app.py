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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Add OpenAI schemas and constants
KEYWORDS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "KeywordExtraction",
    "type": "object",
    "required": ["category_keywords", "search_keywords"],
    "properties": {
        "category_keywords": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string", "minLength": 1, "maxLength": 40}},
        "search_keywords": {"type": "array", "minItems": 3, "maxItems": 12, "items": {"type": "string", "minLength": 1, "maxLength": 30}},
        "brand": {"type": "string"},
        "identifiers": {
            "type": "object",
            "properties": {
                "isbn": {"type": "string"},
                "ean": {"type": "string"},
                "gtin": {"type": "string"},
                "mpn": {"type": "string"}
            },
            "additionalProperties": False
        },
        "normalized_title": {"type": "string", "maxLength": 80}
    },
    "additionalProperties": False
}
ASPECTS_FILL_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "AspectsFill",
    "type": "object",
    "required": ["filled", "missing_required"],
    "properties": {
        "filled": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1
            }
        },
        "missing_required": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0
        },
        "notes": {"type": "string"}
    },
    "additionalProperties": False
}
SMALL_WORDS = {
    "a", "an", "the", "and", "or", "nor", "but", "for", "so", "yet",
    "at", "by", "in", "of", "on", "to", "up", "off", "as", "if",
    "per", "via", "vs", "vs."
}
MAX_LEN = 30
db_pool = psycopg2.pool.SimpleConnectionPool(
    1, 20, DB_URL
)
SCOPES = " ".join([
    "https://api.ebay.com/oauth/api_scope",
    "https://api.ebay.com/oauth/api_scope/sell.inventory",
    "https://api.ebay.com/oauth/api_scope/sell.account",
])
LISTING_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["raw_text", "price", "quantity"],
    "properties": {
        "raw_text": {"type": "string", "minLength": 1, "maxLength": 8000},
        "images": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "price": {
            "type": "object",
            "required": ["value", "currency"],
            "properties": {
                "value": {"type": "number", "minimum": 0.01},
                "currency": {"type": "string", "enum": ["GBP", "USD", "EUR"]}
            }
        },
        "quantity": {"type": "integer", "minimum": 1, "maximum": 999},
        "condition": {"type": "string", "enum": ["NEW", "USED", "REFURBISHED"]},
        "use_html_description": {"type": "boolean"},
        "use_simple_prompt_description": {"type": "boolean"},
        "include_debug": {"type": "boolean"}
    }
}
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
# OpenAI Helper Functions
def clean_keywords(keywords):
    """Clean and truncate keywords to MAX_LEN"""
    cleaned = []
    for kw in keywords:
        kw = kw.strip()
        if len(kw) > MAX_LEN:
            kw = kw[:MAX_LEN].rsplit(" ", 1)[0]  # Cut at last full word
        cleaned.append(kw)
    return cleaned
def call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    """Call OpenAI in JSON mode and return a dict."""
    if not OPENAI_API_KEY:
        raise NotImplementedError("OPENAI_API_KEY not set; LLM features disabled.")
    # Ensure prompts contain 'json' to satisfy the API requirement
    sys_p = (system_prompt or "").strip()
    usr_p = (user_prompt or "").strip()
    if "json" not in sys_p.lower():
        sys_p = "Return a JSON object only. " + sys_p
    if "json" not in usr_p.lower():
        usr_p = usr_p + "\n\nReturn a JSON object only."
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
            ],
            temperature=0.0,
        )
        data = json.loads(resp.choices[0].message.content)
        return data
    except Exception as e:
        # Fallback retry: drop response_format and coerce JSON-only reply
        try:
            resp2 = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_p + "\nReturn only valid JSON. No prose."},
                    {"role": "user", "content": usr_p + "\nOnly valid JSON. No prose."},
                ],
                temperature=0.0,
            )
            txt = (resp2.choices[0].message.content or "").strip()
            # Extract the largest JSON object in the text (defensive)
            start = txt.find("{")
            end = txt.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise RuntimeError(f"LLM did not return JSON: {txt[:200]}")
            payload = txt[start:end+1]
            return json.loads(payload)
        except Exception as e2:
            raise RuntimeError(f"LLM JSON call failed: {e}\nFallback failed: {e2}")
def call_llm_text_simple(user_prompt: str, system_prompt: Optional[str] = None) -> str:
    """Simple text-based LLM call"""
    if not OPENAI_API_KEY:
        raise NotImplementedError("OPENAI_API_KEY not set; LLM features disabled.")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()
def build_description_simple_from_raw(raw_text: str, html_mode: bool = True) -> Dict[str, str]:
    """Generate a description using simple prompt, with HTML support."""
    if html_mode:
        prompt = (
            "Return HTML only. Use ONLY <p>, <ul>, <li>, <br>, <strong>, <em> tags. "
            "No headings (h1–h6), no tables, no images, no scripts. "
            "Write eBay product description for this product: " + str(raw_text)
        )
    else:
        prompt = (
            "Write eBay product description for this product (plain text only, "
            "no headings, no bullet points, no bold): " + str(raw_text)
        )
    
    try:
        out = call_llm_text_simple(prompt)
        out = out[:6000].strip()
        if html_mode:
            html_desc = out
            text_desc = _strip_html(html_desc)
            return {"html": html_desc, "text": text_desc}
        else:
            return {"html": out, "text": out}
    except Exception:
        fallback = _clean_text(raw_text, limit=2000)
        return {"html": fallback if not html_mode else f"<p>{fallback}</p>", "text": fallback}
def _strip_html(s: str) -> str:
    """Simple HTML → text fallback"""
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</(p|li|h[1-6])>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(re.sub(r"\n{3,}", "\n\n", s)).strip()
def _aspect_name(x: Any) -> Optional[str]:
    """Extract aspect name from various formats"""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return (
            x.get("aspectName")
            or x.get("localizedAspectName")
            or x.get("name")
            or (x.get("aspect") or {}).get("name")
        )
    return None
def apply_aspect_constraints(filled: Dict[str, List[str]], aspects_raw: list):
    """Apply eBay aspect constraints like max length"""
    def _constraint_map(aspects_raw: list) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for a in aspects_raw or []:
            name = a.get("localizedAspectName") or (a.get("aspect") or {}).get("name")
            cons = a.get("aspectConstraint", {}) or {}
            if not name:
                continue
            out[name] = {
                "max_len": cons.get("aspectValueMaxLength"),
                "mode": a.get("aspectMode")
            }
        return out
    
    def _trim_to_limit(s: str, limit: int) -> str:
        s = (s or "").strip()
        if limit <= 0 or len(s) <= limit:
            return s
        cut = s[:limit]
        if " " in cut:
            wcut = cut.rsplit(" ", 1)[0]
            if len(wcut) >= max(10, limit - 10):
                return wcut
        return cut
    cmap = _constraint_map(aspects_raw)
    adjusted: Dict[str, List[str]] = {}
    for k, vals in (filled or {}).items():
        vlist = []
        max_len = cmap.get(k, {}).get("max_len")
        mode = cmap.get(k, {}).get("mode")
        for v in (vals or []):
            nv = str(v).strip()
            if mode == "FREE_TEXT" and isinstance(max_len, int) and max_len > 0 and len(nv) > max_len:
                nv = _trim_to_limit(nv, max_len)
            vlist.append(nv)
        if vlist:
            adjusted[k] = vlist
    return adjusted
def _fallback_title(raw_text: str) -> str:
    """Generate fallback title from raw text"""
    t = (raw_text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return (t[:80]) if t else "Untitled Item"
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
def _b64_basic():
    return "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
def _now():
    return time.time()
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
def get_user_tokens(user_id):
    if not is_user_active(user_id):
        return {"access": None, "refresh": None, "exp": 0}
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT access_token, refresh_token, expires_at FROM ebay_tokens WHERE user_id = %s", (user_id,))
            row = c.fetchone()
            if row:
                return {"access": row[0], "refresh": row[1], "exp": row[2] or 0}
            return {"access": None, "refresh": None, "exp": 0}
    finally:
        close_db_connection(conn)
def save_user_tokens(user_id, access_token, refresh_token, expires_in):
    if not is_user_active(user_id):
        return False
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            expires_at = _now() + expires_in
            c.execute("""
                INSERT INTO ebay_tokens
                (user_id, access_token, refresh_token, expires_at, updated_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id) DO UPDATE
                SET access_token = EXCLUDED.access_token,
                    refresh_token = EXCLUDED.refresh_token,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = CURRENT_TIMESTAMP
            """, (user_id, access_token, refresh_token, expires_at))
            conn.commit()
            return True
    except psycopg2.Error:
        conn.rollback()
        return False
    finally:
        close_db_connection(conn)
def update_listing_count():
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("UPDATE listing_counts SET total_count = total_count + 1, updated_at = CURRENT_TIMESTAMP WHERE id = 1")
            conn.commit()
    finally:
        close_db_connection(conn)
def save_user_listing(user_id, listing_data):
    """Save user listing to database"""
    if not is_user_active(user_id):
        return False
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO user_listings
                (user_id, listing_id, offer_id, sku, title, price_value, price_currency,
                 quantity, condition, category_id, category_name, marketplace_id, view_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, listing_id) DO UPDATE
                SET title = EXCLUDED.title,
                    price_value = EXCLUDED.price_value,
                    quantity = EXCLUDED.quantity,
                    view_url = EXCLUDED.view_url
            """, (
                user_id, listing_data['listing_id'], listing_data.get('offer_id'),
                listing_data.get('sku'), listing_data['title'], 
                listing_data.get('price_value'), listing_data.get('price_currency', 'GBP'),
                listing_data.get('quantity'), listing_data.get('condition'),
                listing_data.get('category_id'), listing_data.get('category_name'),
                listing_data.get('marketplace_id'), listing_data.get('view_url')
            ))
            conn.commit()
            return True
    except psycopg2.Error as e:
        print(f"Error saving user listing: {e}")
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
def ensure_access_token(user_id):
    if not is_user_active(user_id):
        raise RuntimeError("User account is inactive")
    tokens = get_user_tokens(user_id)
    if tokens["access"] and _now() < tokens["exp"] - 60:
        return tokens["access"]
    if not tokens["refresh"]:
        raise RuntimeError("No refresh token. Please authenticate with eBay.")
    r = requests.post(
        TOKEN,
        headers={
            "Authorization": _b64_basic(),
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "refresh_token",
            "refresh_token": tokens["refresh"],
            "scope": SCOPES,
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    save_user_tokens(user_id, data["access_token"], data.get("refresh_token"), data["expires_in"])
    return data["access_token"]
def smart_titlecase(s: str) -> str:
    """Title-case words while keeping acronyms, numbers, and small words sane."""
    if not s:
        return s
    words = s.strip().split()
    out = []
    for i, w in enumerate(words):
        # keep acronyms or already-cased words (e.g., USB, EAL, 4K, HD)
        if re.search(r"[A-Z]{2,}", w) or re.search(r"\d[A-Za-z]|[A-Za-z]\d", w):
            out.append(w)
            continue
        # hyphenated or slashed parts get cased individually
        def cap_core(token: str) -> str:
            if not token:
                return token
            # preserve apostrophes: sainsbury's -> Sainsbury's
            if "'" in token:
                head, *rest = token.split("'")
                return head[:1].upper() + head[1:].lower() + "".join("'" + r.lower() for r in rest)
            return token[:1].upper() + token[1:].lower()
        def cap_compound(token: str) -> str:
            parts = re.split(r"(-|/)", token)  # keep delimiters
            return "".join(cap_core(p) if p not in ("-", "/") else p for p in parts)
        lower = w.lower()
        if 0 < i < len(words) - 1 and lower in SMALL_WORDS and not re.search(r"[:–—-]$", out[-1] if out else ""):
            out.append(lower)
        else:
            out.append(cap_compound(w))
    # First & last words always capitalised
    if out:
        out[0] = out[0][:1].upper() + out[0][1:]
        out[-1] = out[-1][:1].upper() + out[-1][1:]
    return " ".join(out)
def _clean_text(t: str, limit=6000) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()[:limit]
def _https_only(urls):
    return [u for u in (urls or []) if isinstance(u, str) and u.startswith("https://")]
def _gen_sku(prefix="ITEM"):
    ts = str(int(time.time() * 1000))
    h = hashlib.sha1(ts.encode()).hexdigest()[:6].upper()
    return f"{prefix}-{ts[-6:]}-{h}"
def get_first_policy_id(kind: str, access: str, marketplace: str) -> str:
    url = f"{BASE}/sell/account/v1/{kind}_policy"
    headers = {
        "Authorization": f"Bearer {access}",
        "Accept-Language": LANG,
        "Content-Language": LANG,
        "X-EBAY-C-MARKETPLACE-ID": marketplace,
    }
    r = requests.get(url, headers=headers, params={"marketplace_id": marketplace}, timeout=30)
    r.raise_for_status()
    list_key = f"{kind}Policies"
    items = r.json().get(list_key, [])
    if not items:
        raise RuntimeError(f"No {kind} policies found in {marketplace}.")
    id_key = f"{kind}PolicyId"
    return items[0][id_key]
def get_or_create_location(access: str, marketplace: str, profile_data: dict) -> str:
    url = f"{BASE}/sell/inventory/v1/location"
    headers = {
        "Authorization": f"Bearer {access}",
        "Accept-Language": LANG,
        "Content-Language": LANG,
        "X-EBAY-C-MARKETPLACE-ID": marketplace,
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    locs = r.json().get("locations", [])
    if locs:
        return locs[0]["merchantLocationKey"]
    merchant_location_key = "PRIMARY_LOCATION"
    create_url = f"{BASE}/sell/inventory/v1/location/{merchant_location_key}"
    payload = {
        "name": "Primary Warehouse",
        "location": {
            "address": {
                "addressLine1": profile_data['address_line1'],
                "city": profile_data['city'],
                "postalCode": profile_data['postal_code'],
                "country": profile_data['country']
            }
        },
        "locationType": "WAREHOUSE",
        "merchantLocationStatus": "ENABLED",
    }
    r = requests.post(create_url, headers=headers | {"Content-Type": "application/json"}, json=payload, timeout=30)
    r.raise_for_status()
    return merchant_location_key
def get_category_tree_id():
    access = ensure_access_token(session.get("user_id"))
    r = requests.get(
        f"{API}/commerce/taxonomy/v1/get_default_category_tree_id",
        params={"marketplace_id": MARKETPLACE_ID},
        headers={"Authorization": f"Bearer {access}"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["categoryTreeId"]
def suggest_leaf_category(tree_id: str, query: str):
    access = ensure_access_token(session.get("user_id"))
    r = requests.get(
        f"{API}/commerce/taxonomy/v1/category_tree/{tree_id}/get_category_suggestions",
        params={"q": query},
        headers={"Authorization": f"Bearer {access}"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json() or {}
    suggestions = data.get("categorySuggestions") or []
    for node in suggestions:
        cat = node.get("category") or {}
        # Prefer leaves when available; treat missing flag as leaf
        if node.get("categoryTreeNodeLevel", 0) > 0 and node.get("leafCategoryTreeNode", True):
            return cat["categoryId"], cat["categoryName"]
    if suggestions:
        cat = suggestions[0]["category"]
        return cat["categoryId"], cat["categoryName"]
    raise RuntimeError("No category suggestions found")
def browse_majority_category(query: str):
    access = ensure_access_token(session.get("user_id"))
    r = requests.get(
        f"{API}/buy/browse/v1/item_summary/search",
        params={"q": query, "limit": 50},
        headers={
            "Authorization": f"Bearer {access}",
            "X-EBAY-C-MARKETPLACE-ID": MARKETPLACE_ID,
        },
        timeout=30,
    )
    r.raise_for_status()
    items = (r.json() or {}).get("itemSummaries", []) or []
    cats = [it.get("categoryId") for it in items if it.get("categoryId")]
    if not cats:
        return None, None
    top_id, _ = Counter(cats).most_common(1)[0]
    return top_id, None
def get_required_and_recommended_aspects(tree_id: str, category_id: str):
    access = ensure_access_token(session.get("user_id"))
    url = f"https://api.ebay.com/commerce/taxonomy/v1/category_tree/{tree_id}/get_item_aspects_for_category"
    r = requests.get(
        url,
        params={"category_id": category_id},
        headers={
            "Authorization": f"Bearer {access}",
            "Accept-Language": LANG,
            "X-EBAY-C-MARKETPLACE-ID": MARKETPLACE_ID,
        },
        timeout=30,
    )
    r.raise_for_status()
    aspects = r.json().get("aspects", [])
    required, recommended = [], []
    for a in aspects:
        name = a.get("localizedAspectName") or a.get("aspect", {}).get("name")
        cons = a.get("aspectConstraint", {})
        if not name:
            continue
        if cons.get("aspectRequired"):
            required.append({"aspect": {"name": name}})
        elif cons.get("aspectUsage") == "RECOMMENDED":
            recommended.append({"aspect": {"name": name}})
    return {
        "required": required,
        "recommended": recommended,
        "raw": aspects,
    }
@app.route("/")
def index():
    return send_from_directory(app.template_folder, 'index.html')
# @app.route("/signup.html")
# def signup_page():
#     return send_from_directory(app.template_folder, 'signup.html')
# @app.route("/login.html")
# def login_page():
#     return send_from_directory(app.template_folder, 'login.html')
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
# @app.route("/signup", methods=["POST"])
# def signup():
#     data = request.get_json() or {}
#     email = data.get("email", "").strip().lower()
#     password = data.get("password", "")
#     if not email or not password:
#         return jsonify({"error": "Email and password are required"}), 400
#     if len(password) < 6:
#         return jsonify({"error": "Password must be at least 6 characters"}), 400
#     conn = get_db_connection()
#     try:
#         with conn.cursor() as c:
#             c.execute("INSERT INTO users (email, password_hash, is_active) VALUES (%s, %s, TRUE) RETURNING id",
#                       (email, generate_password_hash(password)))
#             user_id = c.fetchone()[0]
#             conn.commit()
#             session["user_id"] = user_id
#             session["email"] = email
#             return jsonify({"status": "success", "message": "User registered successfully"})
#     except psycopg2.errors.UniqueViolation:
#         conn.rollback()
#         return jsonify({"error": "Email already exists"}), 400
#     except psycopg2.Error:
#         conn.rollback()
#         return jsonify({"error": "Registration failed"}), 500
#     finally:
#         close_db_connection(conn)
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
@app.route("/ebay-login")
def ebay_login():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_user_active(session["user_id"]):
        return jsonify({"error": "Account is inactive"}), 403
    profile = get_user_profile(session["user_id"])
    if not profile:
        return jsonify({"error": "Please create your profile first"}), 400
    from urllib.parse import quote
    scope_enc = quote(SCOPES, safe="")
    ru_enc = quote(RU_NAME, safe="")
    if session.get("force_ebay_login"):
        url = f"{AUTH}/oauth2/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={ru_enc}&scope={scope_enc}&state=xyz123&prompt=login"
    else:
        url = f"{AUTH}/oauth2/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={ru_enc}&scope={scope_enc}&state=xyz123"
    return redirect(url)
@app.route("/callback")
def callback():
    if "user_id" not in session:
        return Response("Please log in first", status=401)
    if not is_user_active(session["user_id"]):
        return Response("Account is inactive", status=403)
    code = request.args.get("code")
    if not code:
        return Response("Missing authorization code", status=400)
    try:
        r = requests.post(
            TOKEN,
            headers={"Authorization": _b64_basic(), "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "authorization_code", "code": code, "redirect_uri": RU_NAME},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if save_user_tokens(session["user_id"], data["access_token"], data.get("refresh_token"), data["expires_in"]):
            return redirect("/ebay-auth.html?ebay_auth=success")
        return Response("Account is inactive", status=403)
    except Exception as e:
        return redirect("/ebay-auth.html?error=auth_failed")
@app.route("/auth-status")
def auth_status():
    if "user_id" not in session:
        return jsonify({
            "is_logged_in": False,
            "has_profile": False,
            "has_ebay_auth": False
        })
    if not is_user_active(session["user_id"]):
        return jsonify({
            "is_logged_in": True,
            "is_active": False,
            "email": session.get("email"),
            "has_profile": False,
            "has_ebay_auth": False
        })
    profile = get_user_profile(session["user_id"])
    tokens = get_user_tokens(session["user_id"])
    is_admin = is_admin_user(session["user_id"])
    return jsonify({
        "is_logged_in": True,
        "is_active": True,
        "email": session.get("email"),
        "has_profile": bool(profile),
        "has_ebay_auth": bool(tokens["refresh"]),
        "access_exp_in": max(0, int(tokens["exp"] - _now())) if tokens["access"] else 0,
        "is_admin": is_admin
    })
def parse_ebay_error(response_text):
    """Parse eBay API error response and return user-friendly message"""
    try:
        error_data = json.loads(response_text)
        
        # Handle different error response formats
        if 'errors' in error_data:
            errors = error_data['errors']
            if errors and len(errors) > 0:
                first_error = errors[0]
                error_id = first_error.get('errorId')
                message = first_error.get('message', '')
                
                # Handle specific error cases
                if error_id == 25002:  # Duplicate listing error
                    # Extract item title and listing ID from the message
                    if 'identical items' in message.lower():
                        return "This item already exists in your eBay listings. eBay doesn't allow identical items from the same seller."
                elif error_id == 25001:  # Category error
                    return "There was an issue with the product category. Please try with a different product description."
                elif error_id == 25003:  # Policy error
                    return "There's an issue with your eBay selling policies. Please check your eBay account settings."
                elif 'listing policies' in message.lower():
                    return "Your eBay account is missing required selling policies. Please set up payment, return, and shipping policies in your eBay account."
                elif 'inventory item' in message.lower():
                    return "Failed to create the product listing. Please check your product details and try again."
                else:
                    # Return the original message if it's user-friendly
                    return message
        
        # Fallback to original response if parsing fails
        return f"eBay API error: {response_text}"
        
    except (json.JSONDecodeError, KeyError, TypeError):
        return f"Unknown eBay error: {response_text}"
# NEW AI-ENHANCED PUBLISH ENDPOINT
@app.route("/publish-item", methods=["POST"])
def publish_item():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_user_active(session["user_id"]):
        return jsonify({"error": "Account is inactive"}), 403
    profile = get_user_profile(session["user_id"])
    if not profile:
        return jsonify({"error": "Please create your profile first"}), 400
    tokens = get_user_tokens(session["user_id"])
    if not tokens["refresh"]:
        return jsonify({"error": "Please authenticate with eBay first"}), 400
    body = request.get_json(force=True) or {}
    try:
        validate(instance=body, schema=LISTING_SCHEMA)
    except Exception as e:
        return jsonify({"error": "Invalid input", "details": str(e)}), 400
    
    raw_text_in = _clean_text(body.get("raw_text"), limit=8000)
    images = _https_only(body.get("images"))
    marketplaceId = MARKETPLACE_ID
    price = body.get("price")
    quantity = int(body.get("quantity", 1))
    condition = (body.get("condition") or "NEW").upper()
    include_debug = bool(body.get("include_debug", False))
    use_simple_prompt = bool(body.get("use_simple_prompt_description", True))
    want_html = bool(body.get("use_html_description", True))
    
    if not raw_text_in and not images:
        return jsonify({"error": "Raw text or images required"}), 400
    
    try:
        # —— Step 1: AI Extract Keywords & Metadata ——
        if not OPENAI_API_KEY:
            # Fallback if no OpenAI
            normalized_title = smart_titlecase(raw_text_in[:80]) or _fallback_title(raw_text_in)
            category_keywords = []
            brand = None
        else:
            system_prompt = (
                "You extract concise keywords for eBay category selection and search. "
                "Return STRICT JSON per the schema. Use ONLY facts present in the input. "
                "Do NOT invent identifiers; if absent, omit the field. "
                "Lowercase all keywords. No punctuation, no duplicates. "
                "search_keywords must be less than 30 characters"
            )
            user_prompt = f"""MARKETPLACE: {marketplaceId}
RAW_TEXT:
{raw_text_in}
IMAGE_URLS (for context only, do not copy text from them):
{chr(10).join(images) if images else "(none)"}
OUTPUT RULES:
- category_keywords: 1–5 short phrases (2–3 words) that best describe the product category.
- search_keywords: 3–12 search terms buyers would type (mix of unigrams/bigrams/trigrams), all lowercase.
- All search_keywords must be ≤ 30 characters.
- normalized_title: <=80 chars, clean and factual (no emojis/promo).
- brand: only if explicitly present in RAW_TEXT.
- identifiers: only if explicitly present (isbn/ean/gtin/mpn)."""
            
            try:
                s1 = call_llm_json(system_prompt, user_prompt)
                validate(instance=s1, schema=KEYWORDS_SCHEMA)
                s1["search_keywords"] = clean_keywords(s1.get("search_keywords", []))
                normalized_title = s1.get("normalized_title") or _fallback_title(raw_text_in)
                category_keywords = s1.get("category_keywords", [])
                brand = s1.get("brand")
            except Exception as e:
                print(f"[AI Keywords Error] {e}")
                # Fallback
                normalized_title = smart_titlecase(raw_text_in[:80]) or _fallback_title(raw_text_in)
                category_keywords = []
                brand = None
        # —— Step 2: Find eBay Category ——
        tree_id = get_category_tree_id()
        query = (" ".join(category_keywords)).strip() or normalized_title
        try:
            cat_id, cat_name = suggest_leaf_category(tree_id, query)
        except Exception:
            cat_id, cat_name = browse_majority_category(query)
            if not cat_id:
                return jsonify({"error": "No category found from taxonomy or browse", "query": query}), 404
        # —— Step 3: Get Required/Recommended Aspects ——
        aspects_info = get_required_and_recommended_aspects(tree_id, cat_id)
        req_in = aspects_info.get("required", [])
        rec_in = aspects_info.get("recommended", [])
        req_names = [n for n in (_aspect_name(x) for x in req_in) if n]
        rec_names = [n for n in (_aspect_name(x) for x in rec_in) if n]
        # —— Step 4: AI Fill Aspects ——
        filled_aspects = {}
        if OPENAI_API_KEY and (req_names or rec_names):
            system_prompt2 = (
                "You fill eBay item aspects from provided text/images. NEVER leave required aspects empty; "
                "extract when explicit, infer when reasonable, otherwise use 'Does not apply'/'Unknown' where acceptable."
            )
            user_prompt2 = f"""
INPUT TEXT:
{normalized_title}
IMAGE URLS (context only, do not OCR):
{chr(10).join(images) if images else "(none)"}
ASPECTS:
- REQUIRED: {req_names}
- RECOMMENDED: {rec_names}
OUTPUT RULES:
{{
  "filled": {{"AspectName": ["value1","value2"]}},
  "missing_required": ["AspectName"],
  "notes": "optional"
}}
"""
            try:
                s3 = call_llm_json(system_prompt2, user_prompt2)
                validate(instance=s3, schema=ASPECTS_FILL_SCHEMA)
                
                allowed = set(req_names + rec_names)
                for k, vals in (s3.get("filled") or {}).items():
                    if k in allowed and isinstance(vals, list):
                        clean_vals = list(dict.fromkeys([str(v).strip() for v in vals if str(v).strip()]))
                        if clean_vals:
                            filled_aspects[k] = clean_vals
                
                # Apply constraints
                filled_aspects = apply_aspect_constraints(filled_aspects, aspects_info.get("raw"))
                if "Book Title" in filled_aspects:
                    filled_aspects["Book Title"] = [v[:65] for v in filled_aspects["Book Title"]]
                    
            except Exception as e:
                print(f"[AI Aspects Error] {e}")
                # Fallback: fill required aspects with "Unknown"
                filled_aspects = {name: ["Unknown"] for name in req_names}
        else:
            # No AI - basic fallback
            filled_aspects = {name: ["Unknown"] for name in req_names}
        # —— Step 5: AI Generate Description ——
        if OPENAI_API_KEY and use_simple_prompt:
            try:
                desc_bundle = build_description_simple_from_raw(raw_text_in, html_mode=want_html)
                description_text = desc_bundle["text"]
                description_html = desc_bundle["html"] if want_html else description_text
            except Exception as e:
                print(f"[AI Description Error] {e}")
                description_text = raw_text_in[:2000]
                description_html = f"<p>{description_text}</p>" if want_html else description_text
        else:
            description_text = raw_text_in[:2000] 
            description_html = f"<p>{description_text}</p>" if want_html else description_text
        # —— Step 6: Create eBay Listing ——
        access = ensure_access_token(session["user_id"])
        lang = "en-GB" if marketplaceId == "EBAY_GB" else "en-US"
        headers = {
            "Authorization": f"Bearer {access}",
            "Content-Type": "application/json",
            "Content-Language": lang,
            "Accept-Language": lang,
            "X-EBAY-C-MARKETPLACE-ID": marketplaceId,
        }
        sku = _gen_sku("RAW")
        title = smart_titlecase(normalized_title)[:80]
        # Create inventory item
        inv_url = f"{BASE}/sell/inventory/v1/inventory_item/{sku}"
        inv_payload = {
            "product": {
                "title": title,
                "description": description_text,
                "aspects": filled_aspects,
                "imageUrls": images
            },
            "condition": condition,
            "availability": {"shipToLocationAvailability": {"quantity": quantity}}
        }
        r = requests.put(inv_url, headers=headers, json=inv_payload, timeout=30)
        if r.status_code not in (200, 201, 204):
            error_msg = parse_ebay_error(r.text)
            return jsonify({"error": error_msg, "step": "inventory_item"}), 400
        # Get policies
        try:
            fulfillment_policy_id = get_first_policy_id("fulfillment", access, marketplaceId)
            payment_policy_id = get_first_policy_id("payment", access, marketplaceId)
            return_policy_id = get_first_policy_id("return", access, marketplaceId)
        except RuntimeError as e:
            return jsonify({"error": f"Missing eBay policies: {str(e)}. Please set up your selling policies in your eBay account."}), 400
        
        merchant_location_key = get_or_create_location(access, marketplaceId, profile)
        # Create offer
        offer_payload = {
            "sku": sku,
            "marketplaceId": marketplaceId,
            "format": "FIXED_PRICE",
            "availableQuantity": quantity,
            "categoryId": cat_id,
            "listingDescription": description_html,
            "pricingSummary": {"price": price},
            "listingPolicies": {
                "fulfillmentPolicyId": fulfillment_policy_id,
                "paymentPolicyId": payment_policy_id,
                "returnPolicyId": return_policy_id
            },
            "merchantLocationKey": merchant_location_key
        }
        offer_url = f"{BASE}/sell/inventory/v1/offer"
        r = requests.post(offer_url, headers=headers, json=offer_payload, timeout=30)
        if r.status_code not in (200, 201):
            error_msg = parse_ebay_error(r.text)
            return jsonify({"error": error_msg, "step": "create_offer"}), 400
        offer_id = r.json().get("offerId")
        
        # Publish listing
        pub_url = f"{BASE}/sell/inventory/v1/offer/{offer_id}/publish"
        r = requests.post(pub_url, headers=headers, timeout=30)
        if r.status_code not in (200, 201):
            error_msg = parse_ebay_error(r.text)
            return jsonify({"error": error_msg, "step": "publish"}), 400
        pub = r.json()
        listing_id = pub.get("listingId") or (pub.get("listingIds") or [None])[0]
        view_url = f"https://www.ebay.co.uk/itm/{listing_id}" if marketplaceId == "EBAY_GB" else None
        
        update_listing_count()
        listing_data = {
            'listing_id': listing_id,
            'offer_id': offer_id,
            'sku': sku,
            'title': title,
            'price_value': price['value'],
            'price_currency': price['currency'],
            'quantity': quantity,
            'condition': condition,
            'category_id': cat_id,
            'category_name': cat_name,
            'marketplace_id': marketplaceId,
            'view_url': view_url
        }
        save_user_listing(session["user_id"], listing_data)
        
        result = {
            "status": "published",
            "offerId": offer_id,
            "listingId": listing_id,
            "viewItemUrl": view_url,
            "sku": sku,
            "marketplaceId": marketplaceId,
            "categoryId": cat_id,
            "categoryName": cat_name,
            "title": title,
            "aspects": filled_aspects
        }
        if include_debug:
            result["debug"] = {
                "step1_extract_keywords": {
                    "normalized_title": normalized_title,
                    "category_keywords": category_keywords,
                    "brand": brand
                },
                "step2_category": {
                    "queryUsed": query,
                    "categoryTreeId": tree_id,
                    "categoryId": cat_id,
                    "categoryName": cat_name
                },
                "step3_aspects": {
                    "requiredAspects": req_names,
                    "recommendedAspects": rec_names,
                    "filled": filled_aspects
                },
                "description": {
                    "text": description_text,
                    "html": description_html,
                    "used_html": want_html
                }
            }
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Network error while communicating with eBay: {str(e)}"}), 500
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
@app.route("/total-listings")
def get_total_listings_route():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_user_active(session["user_id"]):
        return jsonify({"error": "Account is inactive"}), 403
    total = get_total_listings()
    return jsonify({"total_listings": total})
@app.route("/user-stats")
def get_user_stats():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_user_active(session["user_id"]):
        return jsonify({"error": "Account is inactive"}), 403
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            # Get listing count for user
            c.execute("SELECT COUNT(*) FROM user_listings WHERE user_id = %s", (session["user_id"],))
            listing_count = c.fetchone()[0]
            
            # Get total value of listings
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
        
@app.route("/my-listings")
def get_my_listings():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_user_active(session["user_id"]):
        return jsonify({"error": "Account is inactive"}), 403
    page = request.args.get('page', 1, type=int)
    limit = 20
    offset = (page - 1) * limit
    
    listings = get_user_listings(session["user_id"], limit, offset)
    
    return jsonify({
        "listings": listings,
        "page": page,
        "has_more": len(listings) == limit
    })
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
            # Get current status
            c.execute("SELECT is_active FROM users WHERE id = %s", (user_id,))
            user = c.fetchone()
            if not user:
                return jsonify({"error": "User not found"}), 404
            
            # Toggle status
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
@app.errorhandler(404)
def not_found(error):
    return send_from_directory(app.template_folder, 'index.html')
@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500
@app.route("/preview-item", methods=["POST"])
def preview_item():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_user_active(session["user_id"]):
        return jsonify({"error": "Account is inactive"}), 403
    profile = get_user_profile(session["user_id"])
    if not profile:
        return jsonify({"error": "Please create your profile first"}), 400
    tokens = get_user_tokens(session["user_id"])
    if not tokens["refresh"]:
        return jsonify({"error": "Please authenticate with eBay first"}), 400
    body = request.get_json(force=True) or {}
    try:
        validate(instance=body, schema=LISTING_SCHEMA)
    except Exception as e:
        return jsonify({"error": "Invalid input", "details": str(e)}), 400
    
    raw_text_in = _clean_text(body.get("raw_text"), limit=8000)
    images = _https_only(body.get("images"))
    marketplaceId = MARKETPLACE_ID
    price = body.get("price")
    quantity = int(body.get("quantity", 1))
    condition = (body.get("condition") or "NEW").upper()
    include_debug = bool(body.get("include_debug", False))
    use_simple_prompt = bool(body.get("use_simple_prompt_description", True))
    want_html = bool(body.get("use_html_description", True))
    
    if not raw_text_in and not images:
        return jsonify({"error": "Raw text or images required"}), 400
    
    try:
        # —— Step 1: AI Extract Keywords & Metadata ——
        if not OPENAI_API_KEY:
            normalized_title = smart_titlecase(raw_text_in[:80]) or _fallback_title(raw_text_in)
            category_keywords = []
            brand = None
        else:
            system_prompt = (
                "You extract concise keywords for eBay category selection and search. "
                "Return STRICT JSON per the schema. Use ONLY facts present in the input. "
                "Do NOT invent identifiers; if absent, omit the field. "
                "Lowercase all keywords. No punctuation, no duplicates. "
                "search_keywords must be ≤ 30 characters"
            )
            user_prompt = f"""MARKETPLACE: {marketplaceId}
RAW_TEXT:
{raw_text_in}
IMAGE_URLS (for context only, do not copy text from them):
{chr(10).join(images) if images else "(none)"}
OUTPUT RULES:
- category_keywords: 1–5 short phrases (2–3 words) that best describe the product category.
- search_keywords: 3–12 search terms buyers would type (mix of unigrams/bigrams/trigrams), all lowercase.
- All search_keywords must be ≤ 30 characters.
- normalized_title: <=80 chars, clean and factual (no emojis/promo).
- brand: only if explicitly present in RAW_TEXT.
- identifiers: only if explicitly present (isbn/ean/gtin/mpn)."""
            
            try:
                s1 = call_llm_json(system_prompt, user_prompt)
                validate(instance=s1, schema=KEYWORDS_SCHEMA)
                s1["search_keywords"] = clean_keywords(s1.get("search_keywords", []))
                normalized_title = s1.get("normalized_title") or _fallback_title(raw_text_in)
                category_keywords = s1.get("category_keywords", [])
                brand = s1.get("brand")
            except Exception as e:
                print(f"[AI Keywords Error] {e}")
                normalized_title = smart_titlecase(raw_text_in[:80]) or _fallback_title(raw_text_in)
                category_keywords = []
                brand = None
        # —— Step 2: Find eBay Category ——
        tree_id = get_category_tree_id()
        query = (" ".join(category_keywords)).strip() or normalized_title
        try:
            cat_id, cat_name = suggest_leaf_category(tree_id, query)
        except Exception:
            cat_id, cat_name = browse_majority_category(query)
            if not cat_id:
                return jsonify({"error": "No category found from taxonomy or browse", "query": query}), 404
        # —— Step 3: Get Required/Recommended Aspects ——
        aspects_info = get_required_and_recommended_aspects(tree_id, cat_id)
        req_in = aspects_info.get("required", [])
        rec_in = aspects_info.get("recommended", [])
        req_names = [n for n in (_aspect_name(x) for x in req_in) if n]
        rec_names = [n for n in (_aspect_name(x) for x in rec_in) if n]
        # —— Step 4: AI Fill Aspects ——
        filled_aspects = {}
        if OPENAI_API_KEY and (req_names or rec_names):
            system_prompt2 = (
                "You fill eBay item aspects from provided text/images. NEVER leave required aspects empty; "
                "extract when explicit, infer when reasonable, otherwise use 'Does not apply'/'Unknown' where acceptable."
            )
            user_prompt2 = f"""
INPUT TEXT:
{normalized_title}
IMAGE URLS (context only, do not OCR):
{chr(10).join(images) if images else "(none)"}
ASPECTS:
- REQUIRED: {req_names}
- RECOMMENDED: {rec_names}
OUTPUT RULES:
{{
  "filled": {{"AspectName": ["value1","value2"]}},
  "missing_required": ["AspectName"],
  "notes": "optional"
}}
"""
            try:
                s3 = call_llm_json(system_prompt2, user_prompt2)
                validate(instance=s3, schema=ASPECTS_FILL_SCHEMA)
                
                allowed = set(req_names + rec_names)
                for k, vals in (s3.get("filled") or {}).items():
                    if k in allowed and isinstance(vals, list):
                        clean_vals = list(dict.fromkeys([str(v).strip() for v in vals if str(v).strip()]))
                        if clean_vals:
                            filled_aspects[k] = clean_vals
                
                filled_aspects = apply_aspect_constraints(filled_aspects, aspects_info.get("raw"))
                if "Book Title" in filled_aspects:
                    filled_aspects["Book Title"] = [v[:65] for v in filled_aspects["Book Title"]]
                    
            except Exception as e:
                print(f"[AI Aspects Error] {e}")
                filled_aspects = {name: ["Unknown"] for name in req_names}
        else:
            filled_aspects = {name: ["Unknown"] for name in req_names}
        # —— Step 5: AI Generate Description ——
        if OPENAI_API_KEY and use_simple_prompt:
            try:
                desc_bundle = build_description_simple_from_raw(raw_text_in, html_mode=want_html)
                description_text = desc_bundle["text"]
                description_html = desc_bundle["html"] if want_html else description_text
            except Exception as e:
                print(f"[AI Description Error] {e}")
                description_text = raw_text_in[:2000]
                description_html = f"<p>{description_text}</p>" if want_html else description_text
        else:
            description_text = raw_text_in[:2000] 
            description_html = f"<p>{description_text}</p>" if want_html else description_text
        # —— Step 6: Prepare Preview Data ——
        title = smart_titlecase(normalized_title)[:80]
        result = {
            "title": title,
            "description": {
                "text": description_text,
                "html": description_html,
                "used_html": want_html
            },
            "aspects": filled_aspects,
            "sku": _gen_sku("RAW"),
            "price": price,
            "quantity": quantity,
            "condition": condition,
            "category_id": cat_id,
            "category_name": cat_name,
            "marketplace_id": marketplaceId,
            "images": images
        }
        if include_debug:
            result["debug"] = {
                "step1_extract_keywords": {
                    "normalized_title": normalized_title,
                    "category_keywords": category_keywords,
                    "brand": brand
                },
                "step2_category": {
                    "queryUsed": query,
                    "categoryTreeId": tree_id,
                    "categoryId": cat_id,
                    "categoryName": cat_name
                },
                "step3_aspects": {
                    "requiredAspects": req_names,
                    "recommendedAspects": rec_names,
                    "filled": filled_aspects
                },
                "description": {
                    "text": description_text,
                    "html": description_html,
                    "used_html": want_html
                }
            }
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Network error while communicating with eBay: {str(e)}"}), 500
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
@app.route("/publish-item-from-preview", methods=["POST"])
def publish_item_from_preview():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_user_active(session["user_id"]):
        return jsonify({"error": "Account is inactive"}), 403
    profile = get_user_profile(session["user_id"])
    if not profile:
        return jsonify({"error": "Please create your profile first"}), 400
    tokens = get_user_tokens(session["user_id"])
    if not tokens["refresh"]:
        return jsonify({"error": "Please authenticate with eBay first"}), 400
    body = request.get_json(force=True) or {}
    required_fields = ["title", "description", "aspects", "sku", "price", "quantity", "condition", "category_id", "marketplace_id", "images"]
    for field in required_fields:
        if field not in body:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    title = _clean_text(body.get("title"), limit=80)
    description_text = _clean_text(body.get("description").get("text"), limit=2000)
    description_html = body.get("description").get("html") if body.get("description").get("used_html") else description_text
    aspects = body.get("aspects", {})
    sku = body.get("sku")
    price = body.get("price")
    quantity = int(body.get("quantity", 1))
    condition = body.get("condition").upper()
    category_id = body.get("category_id")
    marketplace_id = body.get("marketplace_id")
    images = _https_only(body.get("images"))
    if not title or not description_text or not images:
        return jsonify({"error": "Title, description, and at least one image URL are required"}), 400
    try:
        # —— Step 1: Create eBay Listing ——
        access = ensure_access_token(session["user_id"])
        lang = "en-GB" if marketplace_id == "EBAY_GB" else "en-US"
        headers = {
            "Authorization": f"Bearer {access}",
            "Content-Type": "application/json",
            "Content-Language": lang,
            "Accept-Language": lang,
            "X-EBAY-C-MARKETPLACE-ID": marketplace_id,
        }
        # Create inventory item
        inv_url = f"{BASE}/sell/inventory/v1/inventory_item/{sku}"
        inv_payload = {
            "product": {
                "title": title,
                "description": description_text,
                "aspects": aspects,
                "imageUrls": images
            },
            "condition": condition,
            "availability": {"shipToLocationAvailability": {"quantity": quantity}}
        }
        r = requests.put(inv_url, headers=headers, json=inv_payload, timeout=30)
        if r.status_code not in (200, 201, 204):
            error_msg = parse_ebay_error(r.text)
            return jsonify({"error": error_msg, "step": "inventory_item"}), 400
        # Get policies
        try:
            fulfillment_policy_id = get_first_policy_id("fulfillment", access, marketplace_id)
            payment_policy_id = get_first_policy_id("payment", access, marketplace_id)
            return_policy_id = get_first_policy_id("return", access, marketplace_id)
        except RuntimeError as e:
            return jsonify({"error": f"Missing eBay policies: {str(e)}. Please set up your selling policies in your eBay account."}), 400
        
        merchant_location_key = get_or_create_location(access, marketplace_id, profile)
        # Create offer
        offer_payload = {
            "sku": sku,
            "marketplaceId": marketplace_id,
            "format": "FIXED_PRICE",
            "availableQuantity": quantity,
            "categoryId": category_id,
            "listingDescription": description_html,
            "pricingSummary": {"price": price},
            "listingPolicies": {
                "fulfillmentPolicyId": fulfillment_policy_id,
                "paymentPolicyId": payment_policy_id,
                "returnPolicyId": return_policy_id
            },
            "merchantLocationKey": merchant_location_key
        }
        offer_url = f"{BASE}/sell/inventory/v1/offer"
        r = requests.post(offer_url, headers=headers, json=offer_payload, timeout=30)
        if r.status_code not in (200, 201):
            error_msg = parse_ebay_error(r.text)
            return jsonify({"error": error_msg, "step": "create_offer"}), 400
        offer_id = r.json().get("offerId")
        
        # Publish listing
        pub_url = f"{BASE}/sell/inventory/v1/offer/{offer_id}/publish"
        r = requests.post(pub_url, headers=headers, timeout=30)
        if r.status_code not in (200, 201):
            error_msg = parse_ebay_error(r.text)
            return jsonify({"error": error_msg, "step": "publish"}), 400
        pub = r.json()
        listing_id = pub.get("listingId") or (pub.get("listingIds") or [None])[0]
        view_url = f"https://www.ebay.co.uk/itm/{listing_id}" if marketplace_id == "EBAY_GB" else None
        
        update_listing_count()
        listing_data = {
            'listing_id': listing_id,
            'offer_id': offer_id,
            'sku': sku,
            'title': title,
            'price_value': price['value'],
            'price_currency': price['currency'],
            'quantity': quantity,
            'condition': condition,
            'category_id': category_id,
            'category_name': body.get("category_name"),
            'marketplace_id': marketplace_id,
            'view_url': view_url
        }
        save_user_listing(session["user_id"], listing_data)
        
        result = {
            "status": "published",
            "offerId": offer_id,
            "listingId": listing_id,
            "viewItemUrl": view_url,
            "sku": sku,
            "marketplaceId": marketplace_id,
            "categoryId": category_id,
            "categoryName": body.get("category_name"),
            "title": title,
            "aspects": aspects
        }
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Network error while communicating with eBay: {str(e)}"}), 500
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
# ------------------------------------------------------------------------------------------------------------
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
# --------------------------------------------------------------------
def generate_and_store_otp(user_id, email, expires_in=600):
    """Generate a 6-digit OTP and store it in the database."""
    otp = str(random.randint(100000, 999999))
    expires_at = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(_now() + expires_in))
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("DELETE FROM otps WHERE user_id = %s", (user_id,))
            c.execute("""
                INSERT INTO otps (user_id, otp, expires_at)
                VALUES (%s, %s, %s)
            """, (user_id, otp, expires_at))
            conn.commit()
        return otp
    except psycopg2.Error as e:
        print(f"Error storing OTP: {e}")
        conn.rollback()
        return None
    finally:
        close_db_connection(conn)
def send_otp_email_password(to_email, otp):
    smtp_host = os.getenv("EMAIL_HOST")
    smtp_port = os.getenv("EMAIL_PORT", 587)
    smtp_user = os.getenv("EMAIL_USER")
    smtp_pass = os.getenv("EMAIL_PASS")
    if not all([smtp_host, smtp_port, smtp_user, smtp_pass]):
        print("Missing email configuration in environment variables")
        return False
    subject = "ListFast.ai Password Reset OTP"
    # Updated HTML for Password Reset OTP
    body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center;">
                <h1 style="color: white; margin: 0;">ListFast.ai</h1>
            </div>
            <div style="padding: 40px 30px; background: #f9f9f9;">
                <h2 style="color: #333; margin-bottom: 20px;">Password Reset Request</h2>
                <p style="color: #666; font-size: 16px; line-height: 1.6;">
                    We received a request to reset the password for your ListFast.ai account.
                    Use the OTP below to proceed with resetting your password:
                </p>
                <div style="background: white; padding: 30px; border-radius: 10px; text-align: center; margin: 30px 0;">
                    <div style="font-size: 32px; font-weight: bold; color: #667eea; letter-spacing: 8px; font-family: monospace;">
                        {otp}
                    </div>
                    <p style="color: #888; font-size: 14px; margin-top: 15px;">
                        This OTP is valid for 10 minutes.
                    </p>
                </div>
                <p style="color: #666; font-size: 14px;">
                    If you did not request a password reset, please ignore this email or contact support at 
                    <a href="mailto:rahul@listfast.ai" style="color:#667eea;">rahul@listfast.ai</a>.
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
    msg = MIMEMultipart("alternative")
    msg['From'] = smtp_user
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))  # <-- send HTML email
    try:
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending OTP email: {e}")
        return False
@app.route("/send-password-change-otp", methods=["POST"])
def send_password_change_otp():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_user_active(session["user_id"]):
        return jsonify({"error": "Account is inactive"}), 403
    email = session.get("email")
    if not email:
        return jsonify({"error": "No email associated with this account"}), 400
    otp = generate_and_store_otp(session["user_id"], email)
    if not otp:
        return jsonify({"error": "Failed to generate OTP"}), 500
    if not send_otp_email_password(email, otp):
        return jsonify({"error": "Failed to send verification code. Please try again."}), 500
    return jsonify({"status": "success", "message": "Verification code sent to your email"})
@app.route("/change-password", methods=["POST"])
def change_password():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_user_active(session["user_id"]):
        return jsonify({"error": "Account is inactive"}), 403
    data = request.get_json() or {}
    otp = data.get("otp", "").strip()
    new_password = data.get("new_password", "").strip()
    if not otp or len(otp) != 6 or not otp.isdigit():
        return jsonify({"error": "Please enter a valid 6-digit verification code"}), 400
    if len(new_password) < 6:
        return jsonify({"error": "Password must be at least 6 characters long"}), 400
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT otp, expires_at FROM otps 
                WHERE user_id = %s AND otp = %s
            """, (session["user_id"], otp))
            otp_record = c.fetchone()
            if not otp_record:
                return jsonify({"error": "Invalid or expired verification code"}), 400
            otp_stored, expires_at = otp_record
            if _now() > time.mktime(expires_at.timetuple()):
                return jsonify({"error": "Verification code has expired"}), 400
            password_hash = generate_password_hash(new_password)
            c.execute("""
                UPDATE users 
                SET password_hash = %s 
                WHERE id = %s
            """, (password_hash, session["user_id"]))
            c.execute("DELETE FROM otps WHERE user_id = %s", (session["user_id"],))
            conn.commit()
            return jsonify({"status": "success", "message": "Password updated successfully"})
    except psycopg2.Error as e:
        conn.rollback()
        return jsonify({"error": "Failed to update password due to database error"}), 500
    finally:
        close_db_connection(conn)
otp_store = {}
pending_users = {}  
SMTP_SERVER = os.getenv("EMAIL_HOST")
SMTP_PORT = 587
EMAIL_ADDRESS = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASS")
    
def generate_otp():
    """Generate a 6-digit OTP"""
    return ''.join([str(random.randint(0, 9)) for _ in range(6)])
def send_otp_email(email, otp):
    """Send OTP via email"""
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
def cleanup_expired_otps():
    """Remove expired OTPs from memory"""
    current_time = datetime.now()
    expired_emails = []
    
    for email, otp_data in otp_store.items():
        if current_time - otp_data['timestamp'] > timedelta(minutes=10):
            expired_emails.append(email)
    
    for email in expired_emails:
        del otp_store[email]
@app.route("/revoke-ebay-auth", methods=["POST"])
def revoke_ebay_auth():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_user_active(session["user_id"]):
        return jsonify({"error": "Account is inactive"}), 403
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("DELETE FROM ebay_tokens WHERE user_id = %s", (session["user_id"],))
            conn.commit()
            session["force_ebay_login"] = True
        return jsonify({"status": "success", "message": "eBay authentication revoked"})
    except psycopg2.Error as e:
        conn.rollback()
        return jsonify({"error": "Failed to revoke eBay authentication"}), 500
    finally:
        close_db_connection(conn)
    
if __name__ == "__main__":
    if not all([CLIENT_ID, CLIENT_SECRET, RU_NAME, DB_URL]):
        print("Missing required environment variables (CLIENT_ID, CLIENT_SECRET, RU_NAME, or DB_URL)")
        exit(1)
    app.run(host="127.0.0.1", port=5000, debug=True)

