import os
import base64
import time
import json
import re
import requests
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
from flask import Flask, request, jsonify, redirect, Response, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from jsonschema import validate
from typing import Dict, List, Optional
import html
from collections import Counter
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
from datetime import datetime, timedelta
from urllib.parse import quote

app = Flask(__name__, static_folder='static', template_folder='templates')
load_dotenv()

# Configuration
app.secret_key = os.getenv("SECRET_KEY")
DB_URL = os.getenv("DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
IMGBB_UPLOAD_URL = 'https://api.imgbb.com/1/upload'

EBAY_CONFIG = {
    "env": os.getenv("EBAY_ENV", "PRODUCTION"),
    "base": os.getenv("EBAY_BASE"),
    "auth": os.getenv("EBAY_AUTH"),
    "token": os.getenv("EBAY_TOKEN_URL"),
    "api": os.getenv("EBAY_API"),
    "marketplace_id": os.getenv("EBAY_MARKETPLACE_ID"),
    "lang": os.getenv("EBAY_LANG"),
    "client_id": os.getenv("EBAY_CLIENT_ID"),
    "client_secret": os.getenv("EBAY_CLIENT_SECRET"),
    "ru_name": os.getenv("EBAY_RU_NAME"),
    "scopes": " ".join([
        "https://api.ebay.com/oauth/api_scope",
        "https://api.ebay.com/oauth/api_scope/sell.inventory",
        "https://api.ebay.com/oauth/api_scope/sell.account",
    ])
}

EMAIL_CONFIG = {
    "host": os.getenv("EMAIL_HOST"),
    "port": os.getenv("EMAIL_PORT", 587),
    "user": os.getenv("EMAIL_USER"),
    "password": os.getenv("EMAIL_PASS")
}

# JSON Schemas
KEYWORDS_SCHEMA = {
    "type": "object",
    "required": ["category_keywords", "search_keywords"],
    "properties": {
        "category_keywords": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string", "minLength": 1, "maxLength": 40}},
        "search_keywords": {"type": "array", "minItems": 3, "maxItems": 12, "items": {"type": "string", "minLength": 1, "maxLength": 30}},
        "brand": {"type": "string"},
        "identifiers": {"type": "object", "properties": {"isbn": {"type": "string"}, "ean": {"type": "string"}, "gtin": {"type": "string"}, "mpn": {"type": "string"}}, "additionalProperties": False},
        "normalized_title": {"type": "string", "maxLength": 80}
    },
    "additionalProperties": False
}

ASPECTS_FILL_SCHEMA = {
    "type": "object",
    "required": ["filled", "missing_required"],
    "properties": {
        "filled": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1}},
        "missing_required": {"type": "array", "items": {"type": "string"}, "minItems": 0},
        "notes": {"type": "string"}
    },
    "additionalProperties": False
}

LISTING_SCHEMA = {
    "type": "object",
    "required": ["raw_text", "price", "quantity"],
    "properties": {
        "raw_text": {"type": "string", "minLength": 1, "maxLength": 8000},
        "images": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "price": {"type": "object", "required": ["value", "currency"], "properties": {"value": {"type": "number", "minimum": 0.01}, "currency": {"type": "string", "enum": ["GBP", "USD", "EUR"]}}},
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

# Database Pool
db_pool = psycopg2.pool.SimpleConnectionPool(1, 20, DB_URL)
SMALL_WORDS = {"a", "an", "the", "and", "or", "nor", "but", "for", "so", "yet", "at", "by", "in", "of", "on", "to", "up", "off", "as", "if", "per", "via", "vs", "vs."}
MAX_LEN = 30

# Utility Functions
def _now():
    return time.time()

def get_db_connection():
    conn = db_pool.getconn()
    conn.autocommit = False
    return conn

def close_db_connection(conn):
    db_pool.putconn(conn)

def clean_keywords(keywords):
    return [kw.strip()[:MAX_LEN].rsplit(" ", 1)[0] if len(kw.strip()) > MAX_LEN else kw.strip() for kw in keywords]

def clean_text(text: str, limit=6000) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()[:limit]

def https_only(urls):
    return [u for u in (urls or []) if isinstance(u, str) and u.startswith("https://")]

def smart_titlecase(s: str) -> str:
    if not s:
        return s
    words = s.strip().split()
    out = []
    for i, w in enumerate(words):
        if re.search(r"[A-Z]{2,}", w) or re.search(r"\d[A-Za-z]|[A-Za-z]\d", w):
            out.append(w)
            continue
        def cap_core(token: str) -> str:
            if not token:
                return token
            if "'" in token:
                head, *rest = token.split("'")
                return head[:1].upper() + head[1:].lower() + "".join("'" + r.lower() for r in rest)
            return token[:1].upper() + token[1:].lower()
        def cap_compound(token: str) -> str:
            parts = re.split(r"(-|/)", token)
            return "".join(cap_core(p) if p not in ("-", "/") else p for p in parts)
        lower = w.lower()
        if 0 < i < len(words) - 1 and lower in SMALL_WORDS and not re.search(r"[:–—-]$", out[-1] if out else ""):
            out.append(lower)
        else:
            out.append(cap_compound(w))
    if out:
        out[0] = out[0][:1].upper() + out[0][1:]
        out[-1] = out[-1][:1].upper() + out[-1][1:]
    return " ".join(out)

def parse_ebay_error(response_text):
    try:
        error_data = json.loads(response_text)
        if 'errors' in error_data and error_data['errors']:
            first_error = error_data['errors'][0]
            error_id = first_error.get('errorId')
            message = first_error.get('message', '')
            error_map = {
                25002: "This item already exists in your eBay listings. eBay doesn't allow identical items from the same seller.",
                25001: "There was an issue with the product category. Please try with a different product description.",
                25003: "There's an issue with your eBay selling policies. Please check your eBay account settings."
            }
            if error_id in error_map:
                return error_map[error_id]
            if 'listing policies' in message.lower():
                return "Your eBay account is missing required selling policies. Please set up payment, return, and shipping policies in your eBay account."
            if 'inventory item' in message.lower():
                return "Failed to create the product listing. Please check your product details and try again."
            return message
        return f"eBay API error: {response_text}"
    except (json.JSONDecodeError, KeyError, TypeError):
        return f"Unknown eBay error: {response_text}"

# Database Initialization
def init_db():
    conn = get_db_connection()
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
            c.execute("INSERT INTO listing_counts (id, total_count) VALUES (1, 0) ON CONFLICT DO NOTHING")
            
            admin_email = os.getenv("ADMIN_EMAIL")
            admin_password = os.getenv("ADMIN_PASSWORD")
            if admin_email and admin_password:
                c.execute("SELECT id FROM users WHERE email = %s", (admin_email.lower(),))
                if not c.fetchone():
                    c.execute("""
                        INSERT INTO users (email, password_hash, is_active) 
                        VALUES (%s, %s, TRUE) 
                        RETURNING id
                    """, (admin_email.lower(), generate_password_hash(admin_password)))
                    admin_id = c.fetchone()[0]
                    c.execute("""
                        INSERT INTO user_profiles
                        (user_id, address_line1, city, postal_code, country, updated_at)
                        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """, (admin_id, "Admin Address", "Admin City", "ADMIN", "GB"))
            conn.commit()
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        conn.rollback()
        raise
    finally:
        close_db_connection(conn)

init_db()

# User Management
def is_user_active(user_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT is_active FROM users WHERE id = %s", (user_id,))
            user = c.fetchone()
            return user and user[0]
    finally:
        close_db_connection(conn)

def is_admin_user(user_id):
    admin_email = os.getenv("ADMIN_EMAIL")
    if not admin_email:
        return False
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT email FROM users WHERE id = %s", (user_id,))
            user = c.fetchone()
            return user and user[0].lower() == admin_email.lower()
    finally:
        close_db_connection(conn)

def get_user_profile(user_id):
    if not is_user_active(user_id):
        return None
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT user_id, address_line1, city, postal_code, country, profile_pic_url, created_at, updated_at 
                FROM user_profiles WHERE user_id = %s
            """, (user_id,))
            profile = c.fetchone()
            return {
                "user_id": profile[0],
                "address_line1": profile[1],
                "city": profile[2],
                "postal_code": profile[3],
                "country": profile[4],
                "profile_pic_url": profile[5],
                "created_at": profile[6],
                "updated_at": profile[7]
            } if profile else None
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

# eBay Authentication
def get_user_tokens(user_id):
    if not is_user_active(user_id):
        return {"access": None, "refresh": None, "exp": 0}
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT access_token, refresh_token, expires_at FROM ebay_tokens WHERE user_id = %s", (user_id,))
            row = c.fetchone()
            return {"access": row[0], "refresh": row[1], "exp": row[2] or 0} if row else {"access": None, "refresh": None, "exp": 0}
    finally:
        close_db_connection(conn)

def save_user_tokens(user_id, access_token, refresh_token, expires_in):
    if not is_user_active(user_id):
        return False
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO ebay_tokens
                (user_id, access_token, refresh_token, expires_at, updated_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id) DO UPDATE
                SET access_token = EXCLUDED.access_token,
                    refresh_token = EXCLUDED.refresh_token,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = CURRENT_TIMESTAMP
            """, (user_id, access_token, refresh_token, _now() + expires_in))
            conn.commit()
            return True
    except psycopg2.Error:
        conn.rollback()
        return False
    finally:
        close_db_connection(conn)

def ensure_access_token(user_id):
    tokens = get_user_tokens(user_id)
    if tokens["access"] and _now() < tokens["exp"] - 60:
        return tokens["access"]
    if not tokens["refresh"]:
        raise RuntimeError("No refresh token. Please authenticate with eBay.")
    r = requests.post(
        EBAY_CONFIG["token"],
        headers={"Authorization": f"Basic {base64.b64encode(f'{EBAY_CONFIG['client_id']}:{EBAY_CONFIG['client_secret']}'.encode()).decode()}", "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "refresh_token", "refresh_token": tokens["refresh"], "scope": EBAY_CONFIG["scopes"]},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    save_user_tokens(user_id, data["access_token"], data.get("refresh_token"), data["expires_in"])
    return data["access_token"]

# eBay API Helpers
def get_ebay_headers(access_token, marketplace_id):
    lang = "en-GB" if marketplace_id == "EBAY_GB" else "en-US"
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Content-Language": lang,
        "Accept-Language": lang,
        "X-EBAY-C-MARKETPLACE-ID": marketplace_id,
    }

def get_first_policy_id(kind: str, access: str, marketplace: str) -> str:
    r = requests.get(
        f"{EBAY_CONFIG['base']}/sell/account/v1/{kind}_policy",
        headers=get_ebay_headers(access, marketplace),
        params={"marketplace_id": marketplace},
        timeout=30
    )
    r.raise_for_status()
    items = r.json().get(f"{kind}Policies", [])
    if not items:
        raise RuntimeError(f"No {kind} policies found in {marketplace}.")
    return items[0][f"{kind}PolicyId"]

def get_or_create_location(access: str, marketplace: str, profile_data: dict) -> str:
    r = requests.get(
        f"{EBAY_CONFIG['base']}/sell/inventory/v1/location",
        headers=get_ebay_headers(access, marketplace),
        timeout=30
    )
    r.raise_for_status()
    locs = r.json().get("locations", [])
    if locs:
        return locs[0]["merchantLocationKey"]
    
    merchant_location_key = "PRIMARY_LOCATION"
    payload = {
        "name": "Primary Warehouse",
        "location": {"address": {
            "addressLine1": profile_data['address_line1'],
            "city": profile_data['city'],
            "postalCode": profile_data['postal_code'],
            "country": profile_data['country']
        }},
        "locationType": "WAREHOUSE",
        "merchantLocationStatus": "ENABLED",
    }
    r = requests.post(
        f"{EBAY_CONFIG['base']}/sell/inventory/v1/location/{merchant_location_key}",
        headers=get_ebay_headers(access, marketplace),
        json=payload,
        timeout=30
    )
    r.raise_for_status()
    return merchant_location_key

def get_category_tree_id(access_token):
    r = requests.get(
        f"{EBAY_CONFIG['api']}/commerce/taxonomy/v1/get_default_category_tree_id",
        params={"marketplace_id": EBAY_CONFIG["marketplace_id"]},
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["categoryTreeId"]

def suggest_leaf_category(tree_id: str, query: str, access_token):
    r = requests.get(
        f"{EBAY_CONFIG['api']}/commerce/taxonomy/v1/category_tree/{tree_id}/get_category_suggestions",
        params={"q": query},
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    r.raise_for_status()
    suggestions = r.json().get("categorySuggestions", [])
    for node in suggestions:
        cat = node.get("category", {})
        if node.get("categoryTreeNodeLevel", 0) > 0 and node.get("leafCategoryTreeNode", True):
            return cat["categoryId"], cat["categoryName"]
    if suggestions:
        cat = suggestions[0]["category"]
        return cat["categoryId"], cat["categoryName"]
    raise RuntimeError("No category suggestions found")

def browse_majority_category(query: str, access_token):
    r = requests.get(
        f"{EBAY_CONFIG['api']}/buy/browse/v1/item_summary/search",
        params={"q": query, "limit": 50},
        headers={"Authorization": f"Bearer {access_token}", "X-EBAY-C-MARKETPLACE-ID": EBAY_CONFIG["marketplace_id"]},
        timeout=30,
    )
    r.raise_for_status()
    items = r.json().get("itemSummaries", [])
    cats = [it.get("categoryId") for it in items if it.get("categoryId")]
    if not cats:
        return None, None
    top_id, _ = Counter(cats).most_common(1)[0]
    return top_id, None

def get_required_and_recommended_aspects(tree_id: str, category_id: str, access_token):
    r = requests.get(
        f"{EBAY_CONFIG['api']}/commerce/taxonomy/v1/category_tree/{tree_id}/get_item_aspects_for_category",
        params={"category_id": category_id},
        headers=get_ebay_headers(access_token, EBAY_CONFIG["marketplace_id"]),
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
    return {"required": required, "recommended": recommended, "raw": aspects}

# AI Helpers
def call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    if not OPENAI_API_KEY:
        raise NotImplementedError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=OPENAI_API_KEY)
    sys_p = "Return a JSON object only. " + system_prompt.strip()
    usr_p = user_prompt.strip() + "\n\nReturn a JSON object only."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys_p + "\nReturn only valid JSON."}, {"role": "user", "content": usr_p + "\nOnly valid JSON."}],
            temperature=0.0,
        )
        txt = resp.choices[0].message.content.strip()
        start, end = txt.find("{"), txt.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"LLM did not return JSON: {txt[:200]}")
        return json.loads(txt[start:end+1])

def call_llm_text_simple(user_prompt: str, system_prompt: Optional[str] = None) -> str:
    if not OPENAI_API_KEY:
        raise NotImplementedError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = [{"role": "user", "content": user_prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
    return resp.choices[0].message.content.strip()

def build_description(raw_text: str, html_mode: bool = True) -> Dict[str, str]:
    prompt = (
        f"Write eBay product description for: {raw_text}\n"
        f"{'Return HTML with <p>, <ul>, <li>, <br>, <strong>, <em> tags only.' if html_mode else 'Plain text only, no headings, no bullet points, no bold.'}"
    )
    try:
        out = call_llm_text_simple(prompt)[:6000].strip()
        html_desc = out if html_mode else f"<p>{out}</p>"
        text_desc = re.sub(r"<[^>]+>", "", re.sub(r"<br\s*/?>", "\n", out, flags=re.I)).strip() if html_mode else out
        return {"html": html_desc, "text": text_desc}
    except Exception:
        fallback = clean_text(raw_text, limit=2000)
        return {"html": f"<p>{fallback}</p>" if html_mode else fallback, "text": fallback}

def apply_aspect_constraints(filled: Dict[str, List[str]], aspects_raw: list):
    cmap = {a.get("localizedAspectName") or (a.get("aspect") or {}).get("name"): {
        "max_len": (a.get("aspectConstraint", {}) or {}).get("aspectValueMaxLength"),
        "mode": a.get("aspectMode")
    } for a in aspects_raw or []}
    adjusted = {}
    for k, vals in (filled or {}).items():
        if k not in cmap:
            continue
        vlist = [str(v).strip()[:cmap[k]["max_len"]] if cmap[k]["mode"] == "FREE_TEXT" and cmap[k]["max_len"] else str(v).strip() for v in vals or []]
        if vlist:
            adjusted[k] = vlist
    return adjusted

# Listing Management
def save_user_listing(user_id, listing_data):
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
            return [{
                'listing_id': row[0], 'offer_id': row[1], 'sku': row[2], 'title': row[3],
                'price_value': float(row[4]) if row[4] else 0, 'price_currency': row[5],
                'quantity': row[6], 'condition': row[7], 'category_name': row[8],
                'view_url': row[9], 'status': row[10], 'created_at': row[11].isoformat() if row[11] else None
            } for row in c.fetchall()]
    except psycopg2.Error as e:
        print(f"Error fetching user listings: {e}")
        return []
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

def get_total_listings():
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT total_count FROM listing_counts WHERE id = 1")
            row = c.fetchone()
            return row[0] if row else 0
    finally:
        close_db_connection(conn)

# eBay Listing Logic
def process_item_data(body, user_id, preview_mode=False):
    try:
        validate(instance=body, schema=LISTING_SCHEMA)
    except Exception as e:
        raise ValueError(f"Invalid input: {str(e)}")
    
    raw_text = clean_text(body.get("raw_text"), limit=8000)
    images = https_only(body.get("images"))
    marketplace_id = EBAY_CONFIG["marketplace_id"]
    price = body.get("price")
    quantity = int(body.get("quantity", 1))
    condition = (body.get("condition") or "NEW").upper()
    include_debug = bool(body.get("include_debug", False))
    use_simple_prompt = bool(body.get("use_simple_prompt_description", True))
    want_html = bool(body.get("use_html_description", True))
    
    if not raw_text and not images:
        raise ValueError("Raw text or images required")
    
    access_token = ensure_access_token(user_id)
    
    # Step 1: Extract Keywords
    normalized_title, category_keywords, brand = None, [], None
    if OPENAI_API_KEY:
        try:
            s1 = call_llm_json(
                "Extract concise keywords for eBay category selection and search. Return STRICT JSON per schema. Use ONLY input facts. Lowercase keywords, no punctuation, no duplicates.",
                f"""MARKETPLACE: {marketplace_id}
RAW_TEXT: {raw_text}
IMAGE_URLS: {chr(10).join(images) if images else "(none)"}
OUTPUT RULES:
- category_keywords: 1–5 short phrases (2–3 words) for category.
- search_keywords: 3–12 search terms, ≤30 chars.
- normalized_title: ≤80 chars, clean, factual.
- brand: only if explicit in RAW_TEXT.
- identifiers: only if explicit (isbn/ean/gtin/mpn)."""
            )
            validate(instance=s1, schema=KEYWORDS_SCHEMA)
            s1["search_keywords"] = clean_keywords(s1.get("search_keywords", []))
            normalized_title = s1.get("normalized_title") or clean_text(raw_text, 80)
            category_keywords = s1.get("category_keywords", [])
            brand = s1.get("brand")
        except Exception as e:
            print(f"[AI Keywords Error] {e}")
    
    normalized_title = normalized_title or smart_titlecase(clean_text(raw_text, 80))
    
    # Step 2: Find Category
    tree_id = get_category_tree_id(access_token)
    query = " ".join(category_keywords) or normalized_title
    try:
        cat_id, cat_name = suggest_leaf_category(tree_id, query, access_token)
    except Exception:
        cat_id, cat_name = browse_majority_category(query, access_token)
        if not cat_id:
            raise RuntimeError(f"No category found for query: {query}")
    
    # Step 3: Get Aspects
    aspects_info = get_required_and_recommended_aspects(tree_id, cat_id, access_token)
    req_names = [a.get("aspect", {}).get("name") for a in aspects_info.get("required", []) if a.get("aspect", {}).get("name")]
    rec_names = [a.get("aspect", {}).get("name") for a in aspects_info.get("recommended", []) if a.get("aspect", {}).get("name")]
    
    # Step 4: Fill Aspects
    filled_aspects = {name: ["Unknown"] for name in req_names}
    if OPENAI_API_KEY and (req_names or rec_names):
        try:
            s3 = call_llm_json(
                "Fill eBay item aspects from text/images. Never leave required aspects empty; use 'Unknown' if needed.",
                f"""INPUT TEXT: {normalized_title}
IMAGE URLS: {chr(10).join(images) if images else "(none)"}
ASPECTS:
- REQUIRED: {req_names}
- RECOMMENDED: {rec_names}
OUTPUT: {{"filled": {{"AspectName": ["value1"]}}, "missing_required": [], "notes": ""}}"""
            )
            validate(instance=s3, schema=ASPECTS_FILL_SCHEMA)
            allowed = set(req_names + rec_names)
            filled_aspects = {k: list(dict.fromkeys([str(v).strip() for v in vals if str(v).strip()])) for k, vals in (s3.get("filled") or {}).items() if k in allowed and vals}
            filled_aspects = apply_aspect_constraints(filled_aspects, aspects_info.get("raw"))
            if "Book Title" in filled_aspects:
                filled_aspects["Book Title"] = [v[:65] for v in filled_aspects["Book Title"]]
        except Exception as e:
            print(f"[AI Aspects Error] {e}")
    
    # Step 5: Generate Description
    description_text = raw_text[:2000]
    description_html = f"<p>{description_text}</p>"
    if OPENAI_API_KEY and use_simple_prompt:
        try:
            desc_bundle = build_description(raw_text, html_mode=want_html)
            description_text = desc_bundle["text"]
            description_html = desc_bundle["html"] if want_html else description_text
        except Exception as e:
            print(f"[AI Description Error] {e}")
    
    title = smart_titlecase(normalized_title)[:80]
    sku = f"ITEM-{int(_now() * 1000)}-{hashlib.sha1(str(_now()).encode()).hexdigest()[:6].upper()}"
    
    result = {
        "title": title,
        "description": {"text": description_text, "html": description_html, "used_html": want_html},
        "aspects": filled_aspects,
        "sku": sku,
        "price": price,
        "quantity": quantity,
        "condition": condition,
        "category_id": cat_id,
        "category_name": cat_name,
        "marketplace_id": marketplace_id,
        "images": images
    }
    
    if not preview_mode:
        profile = get_user_profile(user_id)
        headers = get_ebay_headers(access_token, marketplace_id)
        
        # Create Inventory Item
        r = requests.put(
            f"{EBAY_CONFIG['base']}/sell/inventory/v1/inventory_item/{sku}",
            headers=headers,
            json={
                "product": {"title": title, "description": description_text, "aspects": filled_aspects, "imageUrls": images},
                "condition": condition,
                "availability": {"shipToLocationAvailability": {"quantity": quantity}}
            },
            timeout=30
        )
        if r.status_code not in (200, 201, 204):
            raise RuntimeError(parse_ebay_error(r.text))
        
        # Get Policies
        try:
            fulfillment_policy_id = get_first_policy_id("fulfillment", access_token, marketplace_id)
            payment_policy_id = get_first_policy_id("payment", access_token, marketplace_id)
            return_policy_id = get_first_policy_id("return", access_token, marketplace_id)
        except RuntimeError as e:
            raise RuntimeError(f"Missing eBay policies: {str(e)}. Please set up your selling policies in your eBay account.")
        
        merchant_location_key = get_or_create_location(access_token, marketplace_id, profile)
        
        # Create Offer
        offer_payload = {
            "sku": sku,
            "marketplaceId": marketplace_id,
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
        r = requests.post(f"{EBAY_CONFIG['base']}/sell/inventory/v1/offer", headers=headers, json=offer_payload, timeout=30)
        if r.status_code not in (200, 201):
            raise RuntimeError(parse_ebay_error(r.text))
        
        offer_id = r.json().get("offerId")
        
        # Publish Listing
        r = requests.post(f"{EBAY_CONFIG['base']}/sell/inventory/v1/offer/{offer_id}/publish", headers=headers, timeout=30)
        if r.status_code not in (200, 201):
            raise RuntimeError(parse_ebay_error(r.text))
        
        pub = r.json()
        listing_id = pub.get("listingId") or (pub.get("listingIds") or [None])[0]
        view_url = f"https://www.ebay.co.uk/itm/{listing_id}" if marketplace_id == "EBAY_GB" else None
        
        update_listing_count()
        save_user_listing(user_id, {
            'listing_id': listing_id, 'offer_id': offer_id, 'sku': sku, 'title': title,
            'price_value': price['value'], 'price_currency': price['currency'], 'quantity': quantity,
            'condition': condition, 'category_id': cat_id, 'category_name': cat_name,
            'marketplace_id': marketplace_id, 'view_url': view_url
        })
        
        result.update({
            "status": "published",
            "offerId": offer_id,
            "listingId": listing_id,
            "viewItemUrl": view_url,
        })
    
    if include_debug:
        result["debug"] = {
            "extract_keywords": {"normalized_title": normalized_title, "category_keywords": category_keywords, "brand": brand},
            "category": {"queryUsed": query, "categoryTreeId": tree_id, "categoryId": cat_id, "categoryName": cat_name},
            "aspects": {"requiredAspects": req_names, "recommendedAspects": rec_names, "filled": filled_aspects},
            "description": {"text": description_text, "html": description_html, "used_html": want_html}
        }
    
    return result

# Routes
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
    if not session.get("user_id") or not is_admin_user(session["user_id"]):
        return jsonify({"error": "Admin access required"}), 403
    return send_from_directory(app.template_folder, 'admin-portal.html')

@app.route("/debug-admin")
def debug_admin():
    admin_email, admin_password = os.getenv("ADMIN_EMAIL"), os.getenv("ADMIN_PASSWORD")
    if not admin_email or not admin_password:
        return jsonify({"error": "ADMIN_EMAIL or ADMIN_PASSWORD not set"})
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT id, email, is_active FROM users WHERE email = %s", (admin_email.lower(),))
            admin_user = c.fetchone()
            if admin_user:
                return jsonify({"admin_exists": True, "admin_id": admin_user[0], "admin_email": admin_user[1], "admin_active": admin_user[2]})
            
            c.execute("INSERT INTO users (email, password_hash, is_active) VALUES (%s, %s, TRUE) RETURNING id",
                      (admin_email.lower(), generate_password_hash(admin_password)))
            admin_id = c.fetchone()[0]
            c.execute("INSERT INTO user_profiles (user_id, address_line1, city, postal_code, country, updated_at) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)",
                      (admin_id, "Admin Address", "Admin City", "ADMIN", "GB"))
            conn.commit()
            return jsonify({"admin_created": True, "admin_id": admin_id, "admin_email": admin_email})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": f"Database error: {str(e)}"})
    finally:
        close_db_connection(conn)

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email, password = data.get("email", "").strip().lower(), data.get("password", "")
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT id, password_hash, is_active FROM users WHERE email = %s", (email,))
            user = c.fetchone()
            if not user or not user[2] or not check_password_hash(user[1], password):
                return jsonify({"error": "Invalid credentials or inactive account"}), 401
            session["user_id"], session["email"], session["is_admin"] = user[0], email, is_admin_user(user[0])
            return jsonify({
                "status": "success",
                "message": "Logged in successfully",
                "redirect": "/admin-portal.html" if session["is_admin"] else "/dashboard.html",
                "is_admin": session["is_admin"]
            })
    finally:
        close_db_connection(conn)

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "success", "message": "Logged out successfully"})

@app.route("/profile", methods=["GET", "POST"])
def profile():
    if not session.get("user_id") or not is_user_active(session["user_id"]):
        return jsonify({"error": "Please log in or account inactive"}), 401
    
    if request.method == "GET":
        profile = get_user_profile(session["user_id"])
        return jsonify({"profile": profile})
    
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
    return jsonify({"status": "success", "message": "Profile saved"}) if save_user_profile(session["user_id"], profile_data) else jsonify({"error": "Failed to save profile"}), 500

@app.route("/ebay-login")
def ebay_login():
    if not session.get("user_id") or not is_user_active(session["user_id"]) or not get_user_profile(session["user_id"]):
        return jsonify({"error": "Please log in and create profile first"}), 401
    scope_enc = quote(EBAY_CONFIG["scopes"], safe="")
    ru_enc = quote(EBAY_CONFIG["ru_name"], safe="")
    return redirect(f"{EBAY_CONFIG['auth']}/oauth2/authorize?client_id={EBAY_CONFIG['client_id']}&response_type=code&redirect_uri={ru_enc}&scope={scope_enc}&state=xyz123")

@app.route("/callback")
def callback():
    if not session.get("user_id") or not is_user_active(session["user_id"]):
        return Response("Please log in or account inactive", status=401)
    
    code = request.args.get("code")
    if not code:
        return Response("Missing authorization code", status=400)
    
    try:
        r = requests.post(
            EBAY_CONFIG["token"],
            headers={"Authorization": f"Basic {base64.b64encode(f'{EBAY_CONFIG['client_id']}:{EBAY_CONFIG['client_secret']}'.encode()).decode()}", "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "authorization_code", "code": code, "redirect_uri": EBAY_CONFIG["ru_name"]},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        return redirect("/ebay-auth.html?ebay_auth=success") if save_user_tokens(session["user_id"], data["access_token"], data.get("refresh_token"), data["expires_in"]) else Response("Account inactive", status=403)
    except Exception:
        return redirect("/ebay-auth.html?error=auth_failed")

@app.route("/auth-status")
def auth_status():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"is_logged_in": False, "has_profile": False, "has_ebay_auth": False})
    
    profile, tokens = get_user_profile(user_id), get_user_tokens(user_id)
    return jsonify({
        "is_logged_in": True,
        "is_active": is_user_active(user_id),
        "email": session.get("email"),
        "has_profile": bool(profile),
        "has_ebay_auth": bool(tokens["refresh"]),
        "access_exp_in": max(0, int(tokens["exp"] - _now())) if tokens["access"] else 0,
        "is_admin": is_admin_user(user_id)
    })

@app.route("/publish-item", methods=["POST"])
def publish_item():
    if not session.get("user_id") or not is_user_active(session["user_id"]) or not get_user_profile(session["user_id"]) or not get_user_tokens(session["user_id"])["refresh"]:
        return jsonify({"error": "Please log in, create profile, and authenticate with eBay"}), 401
    
    try:
        return jsonify(process_item_data(request.get_json(force=True) or {}, session["user_id"]))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Network error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/preview-item", methods=["POST"])
def preview_item():
    if not session.get("user_id") or not is_user_active(session["user_id"]) or not get_user_profile(session["user_id"]) or not get_user_tokens(session["user_id"])["refresh"]:
        return jsonify({"error": "Please log in, create profile, and authenticate with eBay"}), 401
    
    try:
        return jsonify(process_item_data(request.get_json(force=True) or {}, session["user_id"], preview_mode=True))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Network error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/publish-item-from-preview", methods=["POST"])
def publish_item_from_preview():
    if not session.get("user_id") or not is_user_active(session["user_id"]) or not get_user_profile(session["user_id"]) or not get_user_tokens(session["user_id"])["refresh"]:
        return jsonify({"error": "Please log in, create profile, and authenticate with eBay"}), 401
    
    body = request.get_json(force=True) or {}
    required_fields = ["title", "description", "aspects", "sku", "price", "quantity", "condition", "category_id", "marketplace_id", "images"]
    if not all(field in body for field in required_fields):
        return jsonify({"error": f"Missing required fields: {', '.join(f for f in required_fields if f not in body)}"}), 400
    
    try:
        title = clean_text(body.get("title"), 80)
        description_text = clean_text(body.get("description").get("text"), 2000)
        description_html = body.get("description").get("html") if body.get("description").get("used_html") else description_text
        aspects, sku, price, quantity = body.get("aspects", {}), body.get("sku"), body.get("price"), int(body.get("quantity", 1))
        condition, category_id, marketplace_id = body.get("condition").upper(), body.get("category_id"), body.get("marketplace_id")
        images = https_only(body.get("images"))
        
        if not title or not description_text or not images:
            return jsonify({"error": "Title, description, and images required"}), 400
        
        access = ensure_access_token(session["user_id"])
        headers = get_ebay_headers(access, marketplace_id)
        profile = get_user_profile(session["user_id"])
        
        r = requests.put(
            f"{EBAY_CONFIG['base']}/sell/inventory/v1/inventory_item/{sku}",
            headers=headers,
            json={
                "product": {"title": title, "description": description_text, "aspects": aspects, "imageUrls": images},
                "condition": condition,
                "availability": {"shipToLocationAvailability": {"quantity": quantity}}
            },
            timeout=30
        )
        if r.status_code not in (200, 201, 204):
            raise RuntimeError(parse_ebay_error(r.text))
        
        try:
            fulfillment_policy_id = get_first_policy_id("fulfillment", access, marketplace_id)
            payment_policy_id = get_first_policy_id("payment", access, marketplace_id)
            return_policy_id = get_first_policy_id("return", access, marketplace_id)
        except RuntimeError as e:
            raise RuntimeError(f"Missing eBay policies: {str(e)}. Please set up your selling policies in your eBay account.")
        
        merchant_location_key = get_or_create_location(access, marketplace_id, profile)
        
        r = requests.post(
            f"{EBAY_CONFIG['base']}/sell/inventory/v1/offer",
            headers=headers,
            json={
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
            },
            timeout=30
        )
        if r.status_code not in (200, 201):
            raise RuntimeError(parse_ebay_error(r.text))
        
        offer_id = r.json().get("offerId")
        r = requests.post(f"{EBAY_CONFIG['base']}/sell/inventory/v1/offer/{offer_id}/publish", headers=headers, timeout=30)
        if r.status_code not in (200, 201):
            raise RuntimeError(parse_ebay_error(r.text))
        
        pub = r.json()
        listing_id = pub.get("listingId") or (pub.get("listingIds") or [None])[0]
        view_url = f"https://www.ebay.co.uk/itm/{listing_id}" if marketplace_id == "EBAY_GB" else None
        
        update_listing_count()
        save_user_listing(session["user_id"], {
            'listing_id': listing_id, 'offer_id': offer_id, 'sku': sku, 'title': title,
            'price_value': price['value'], 'price_currency': price['currency'], 'quantity': quantity,
            'condition': condition, 'category_id': category_id, 'category_name': body.get("category_name"),
            'marketplace_id': marketplace_id, 'view_url': view_url
        })
        
        return jsonify({
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
        })
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Network error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/total-listings")
def get_total_listings_route():
    if not session.get("user_id") or not is_user_active(session["user_id"]):
        return jsonify({"error": "Please log in or account inactive"}), 401
    return jsonify({"total_listings": get_total_listings()})

@app.route("/user-stats")
def get_user_stats():
    if not session.get("user_id") or not is_user_active(session["user_id"]):
        return jsonify({"error": "Please log in or account inactive"}), 401
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT COUNT(*) FROM user_listings WHERE user_id = %s", (session["user_id"],))
            listing_count = c.fetchone()[0]
            c.execute("""
                SELECT SUM(price_value * quantity), COUNT(CASE WHEN status = 'ACTIVE' THEN 1 END)
                FROM user_listings WHERE user_id = %s
            """, (session["user_id"],))
            total_value, active_count = c.fetchone()
            return jsonify({
                "total_listings": listing_count,
                "active_listings": active_count or 0,
                "total_inventory_value": float(total_value or 0),
                "email": session.get("email")
            })
    except psycopg2.Error:
        return jsonify({"error": "Failed to fetch stats"}), 500
    finally:
        close_db_connection(conn)

@app.route("/my-listings")
def get_my_listings():
    if not session.get("user_id") or not is_user_active(session["user_id"]):
        return jsonify({"error": "Please log in or account inactive"}), 401
    
    page = request.args.get('page', 1, type=int)
    limit = 20
    offset = (page - 1) * limit
    listings = get_user_listings(session["user_id"], limit, offset)
    return jsonify({"listings": listings, "page": page, "has_more": len(listings) == limit})

@app.route("/admin/users")
def admin_get_users():
    if not session.get("user_id") or not is_admin_user(session["user_id"]):
        return jsonify({"error": "Admin access required"}), 403
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT id, email, is_active, created_at, (SELECT COUNT(*) FROM user_listings WHERE user_id = users.id)
                FROM users WHERE email != %s ORDER BY created_at DESC
            """, (os.getenv("ADMIN_EMAIL").lower(),))
            return jsonify({"users": [{
                'id': row[0], 'email': row[1], 'is_active': row[2],
                'created_at': row[3].isoformat() if row[3] else None, 'listing_count': row[4]
            } for row in c.fetchall()]})
    except psycopg2.Error:
        return jsonify({"error": "Failed to fetch users"}), 500
    finally:
        close_db_connection(conn)

@app.route("/admin/users/<int:user_id>/toggle-status", methods=["POST"])
def admin_toggle_user_status(user_id):
    if not session.get("user_id") or not is_admin_user(session["user_id"]):
        return jsonify({"error": "Admin access required"}), 403
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT is_active FROM users WHERE id = %s", (user_id,))
            user = c.fetchone()
            if not user:
                return jsonify({"error": "User not found"}), 404
            c.execute("UPDATE users SET is_active = %s WHERE id = %s", (not user[0], user_id))
            conn.commit()
            return jsonify({"status": "success", "message": f"User {'activated' if not user[0] else 'deactivated'} successfully", "is_active": not user[0]})
    except psycopg2.Error:
        conn.rollback()
        return jsonify({"error": "Failed to update user status"}), 500
    finally:
        close_db_connection(conn)

@app.route("/admin/stats")
def admin_get_stats():
    if not session.get("user_id") or not is_admin_user(session["user_id"]):
        return jsonify({"error": "Admin access required"}), 403
    
    admin_email = os.getenv("ADMIN_EMAIL")
    if not admin_email:
        return jsonify({"error": "ADMIN_EMAIL not set"}), 500
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT COUNT(*) FROM users WHERE email != %s", (admin_email.lower(),))
            total_users = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM users WHERE is_active = TRUE AND email != %s", (admin_email.lower(),))
            active_users = c.fetchone()[0]
            c.execute("SELECT COUNT(*), SUM(price_value * quantity) FROM user_listings ul JOIN users u ON ul.user_id = u.id WHERE u.email != %s", (admin_email.lower(),))
            total_listings, total_value = c.fetchone()
            c.execute("SELECT COUNT(*) FROM users WHERE created_at >= CURRENT_DATE - INTERVAL '7 days' AND email != %s", (admin_email.lower(),))
            recent_registrations = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM user_listings ul JOIN users u ON ul.user_id = u.id WHERE ul.created_at >= CURRENT_DATE - INTERVAL '7 days' AND u.email != %s", (admin_email.lower(),))
            recent_listings = c.fetchone()[0]
            return jsonify({
                "total_users": total_users,
                "active_users": active_users,
                "total_listings": total_listings or 0,
                "total_value": float(total_value or 0),
                "recent_registrations": recent_registrations,
                "recent_listings": recent_listings
            })
    except psycopg2.Error:
        return jsonify({"error": "Failed to fetch stats"}), 500
    finally:
        close_db_connection(conn)

@app.route("/admin/listings")
def admin_get_all_listings():
    if not session.get("user_id") or not is_admin_user(session["user_id"]):
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
                FROM user_listings ul JOIN users u ON ul.user_id = u.id
                ORDER BY ul.created_at DESC LIMIT %s OFFSET %s
            """, (limit, offset))
            return jsonify({
                "listings": [{
                    'listing_id': row[0], 'title': row[1], 'price_value': float(row[2]) if row[2] else 0,
                    'price_currency': row[3], 'quantity': row[4], 'status': row[5],
                    'created_at': row[6].isoformat() if row[6] else None, 'user_email': row[7]
                } for row in c.fetchall()],
                "page": page,
                "has_more": len(c.fetchall()) == limit
            })
    except psycopg2.Error:
        return jsonify({"error": "Failed to fetch listings"}), 500
    finally:
        close_db_connection(conn)

@app.route("/upload-profile-image", methods=["POST"])
def upload_profile_image():
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    try:
        encoded_image = base64.b64encode(file.read()).decode("utf-8")
        response = requests.post(IMGBB_UPLOAD_URL, data={"key": IMGBB_API_KEY, "image": encoded_image, "name": file.filename}, timeout=30)
        result = response.json()
        return jsonify({"status": "success", "image_url": result["data"]["url"]}) if result.get("success") else jsonify({"error": "Upload failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# OTP and Email Management
otp_store = {}
def send_otp_email(email, otp, purpose="signup"):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG["user"]
        msg['To'] = email
        msg['Subject'] = "ListFast.ai Verification Code" if purpose == "signup" else "ListFast.ai Password Reset OTP"
        
        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center;">
                    <h1 style="color: white; margin: 0;">ListFast.ai</h1>
                </div>
                <div style="padding: 40px 30px; background: #f9f9f9;">
                    <h2 style="color: #333; margin-bottom: 20px;">{'Welcome to ListFast.ai!' if purpose == 'signup' else 'Password Reset Request'}</h2>
                    <p style="color: #666; font-size: 16px; line-height: 1.6;">
                        {'Thank you for signing up. Please use the verification code below to complete your registration:' if purpose == 'signup' else 'Use the OTP below to reset your password:'}
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
                        If you didn't request this code, please ignore this email or contact <a href="mailto:rahul@listfast.ai" style="color:#667eea;">rahul@listfast.ai</a>.
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
        server = smtplib.SMTP(EMAIL_CONFIG["host"], EMAIL_CONFIG["port"])
        server.starttls()
        server.login(EMAIL_CONFIG["user"], EMAIL_CONFIG["password"])
        server.sendmail(EMAIL_CONFIG["user"], email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json() or {}
    email, password = data.get("email", "").strip().lower(), data.get("password", "")
    if not email or not password or len(password) < 6:
        return jsonify({"error": "Valid email and password (6+ chars) required"}), 400
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT id FROM users WHERE email = %s", (email,))
            if c.fetchone():
                return jsonify({"error": "Email already exists"}), 400
            otp = str(random.randint(100000, 999999))
            otp_store[email] = {'otp': otp, 'password': password, 'timestamp': datetime.now(), 'attempts': 0}
            return jsonify({'message': 'Verification code sent'}) if send_otp_email(email, otp) else jsonify({"error": "Failed to send verification email"}), 500
    except psycopg2.Error:
        return jsonify({"error": "Registration failed"}), 500
    finally:
        close_db_connection(conn)

@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.get_json() or {}
    email, otp = data.get('email', '').strip().lower(), data.get('otp', '')
    if not email or not otp:
        return jsonify({'error': 'Email and OTP required'}), 400
    
    if email not in otp_store:
        return jsonify({'error': 'No verification code found'}), 400
    
    otp_data = otp_store[email]
    if datetime.now() - otp_data['timestamp'] > timedelta(minutes=10):
        del otp_store[email]
        return jsonify({'error': 'Verification code expired'}), 400
    
    if otp_data['attempts'] >= 5:
        del otp_store[email]
        return jsonify({'error': 'Too many incorrect attempts'}), 400
    
    if otp != otp_data['otp']:
        otp_data['attempts'] += 1
        return jsonify({'error': 'Invalid verification code'}), 400
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("INSERT INTO users (email, password_hash, is_active) VALUES (%s, %s, TRUE) RETURNING id",
                      (email, generate_password_hash(otp_data['password'])))
            user_id = c.fetchone()[0]
            conn.commit()
            del otp_store[email]
            session["user_id"], session["email"] = user_id, email
            return jsonify({'message': 'Email verified successfully!', 'user_id': user_id})
    except psycopg2.Error:
        conn.rollback()
        return jsonify({'error': 'Account creation failed'}), 500
    finally:
        close_db_connection(conn)

@app.route('/resend-otp', methods=['POST'])
def resend_otp():
    data = request.get_json() or {}
    email = data.get('email', '').strip().lower()
    if not email or email not in otp_store:
        return jsonify({'error': 'No pending verification for this email'}), 400
    
    if datetime.now() - otp_store[email]['timestamp'] < timedelta(minutes=1):
        return jsonify({'error': 'Please wait before requesting a new code'}), 429
    
    otp = str(random.randint(100000, 999999))
    otp_store[email].update({'otp': otp, 'timestamp': datetime.now(), 'attempts': 0})
    return jsonify({'message': 'New verification code sent'}) if send_otp_email(email, otp) else jsonify({'error': 'Failed to send verification email'}), 500

@app.route("/send-password-change-otp", methods=["POST"])
def send_password_change_otp():
    if not session.get("user_id") or not is_user_active(session["user_id"]):
        return jsonify({"error": "Please log in or account inactive"}), 401
    
    email = session.get("email")
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            otp = str(random.randint(100000, 999999))
            expires_at = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(_now() + 600))
            c.execute("DELETE FROM otps WHERE user_id = %s", (session["user_id"],))
            c.execute("INSERT INTO otps (user_id, otp, expires_at) VALUES (%s, %s, %s)", (session["user_id"], otp, expires_at))
            conn.commit()
            return jsonify({"status": "success", "message": "Verification code sent"}) if send_otp_email(email, otp, purpose="password_reset") else jsonify({"error": "Failed to send verification code"}), 500
    except psycopg2.Error:
        conn.rollback()
        return jsonify({"error": "Failed to generate OTP"}), 500
    finally:
        close_db_connection(conn)

@app.route("/change-password", methods=["POST"])
def change_password():
    if not session.get("user_id") or not is_user_active(session["user_id"]):
        return jsonify({"error": "Please log in or account inactive"}), 401
    
    data = request.get_json() or {}
    otp, new_password = data.get("otp", "").strip(), data.get("new_password", "").strip()
    if not otp.isdigit() or len(otp) != 6 or len(new_password) < 6:
        return jsonify({"error": "Invalid OTP or password (6+ chars)"}), 400
    
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute("SELECT otp, expires_at FROM otps WHERE user_id = %s AND otp = %s", (session["user_id"], otp))
            otp_record = c.fetchone()
            if not otp_record or _now() > time.mktime(otp_record[1].timetuple()):
                return jsonify({"error": "Invalid or expired verification code"}), 400
            c.execute("UPDATE users SET password_hash = %s WHERE id = %s", (generate_password_hash(new_password), session["user_id"]))
            c.execute("DELETE FROM otps WHERE user_id = %s", (session["user_id"],))
            conn.commit()
            return jsonify({"status": "success", "message": "Password updated successfully"})
    except psycopg2.Error:
        conn.rollback()
        return jsonify({"error": "Failed to update password"}), 500
    finally:
        close_db_connection(conn)

@app.errorhandler(404)
def not_found(error):
    return send_from_directory(app.template_folder, 'index.html')

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    if not all([EBAY_CONFIG["client_id"], EBAY_CONFIG["client_secret"], EBAY_CONFIG["ru_name"], DB_URL]):
        print("Missing required environment variables")
        exit(1)
    app.run(host="127.0.0.1", port=5000, debug=True)
