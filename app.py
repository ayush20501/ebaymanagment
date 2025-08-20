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

db_pool = psycopg2.pool.SimpleConnectionPool(1, 20, DB_URL)

SCOPES = " ".join([
    "https://api.ebay.com/oauth/api_scope",
    "https://api.ebay.com/oauth/api_scope/sell.inventory",
    "https://api.ebay.com/oauth/api_scope/sell.account",
])

# Schemas
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
        "country": {"type": "string", "enum": ["GB"]}
    }
}

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

MAX_LEN = 30

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
            conn.commit()
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        raise
    finally:
        db_pool.putconn(conn)

init_db()

# Utility functions
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
            c.execute("SELECT * FROM user_profiles WHERE user_id = %s", (user_id,))
            profile = c.fetchone()
            if profile:
                return {
                    "user_id": profile[0],
                    "address_line1": profile[1],
                    "city": profile[2],
                    "postal_code": profile[3],
                    "country": profile[4],
                    "created_at": profile[5],
                    "updated_at": profile[6]
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
                (user_id, address_line1, city, postal_code, country, updated_at)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id) DO UPDATE
                SET address_line1 = EXCLUDED.address_line1,
                    city = EXCLUDED.city,
                    postal_code = EXCLUDED.postal_code,
                    country = EXCLUDED.country,
                    updated_at = CURRENT_TIMESTAMP
            """, (user_id, profile_data['address_line1'], profile_data['city'],
                  profile_data['postal_code'], profile_data['country']))
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
    small_words = {"a", "an", "the", "and", "or", "but", "for", "to", "in", "on", "at", "by"}
    words = s.strip().split()
    out = []
    for i, w in enumerate(words):
        if i > 0 and i < len(words) - 1 and w.lower() in small_words:
            out.append(w.lower())
        else:
            out.append(w[:1].upper() + w[1:].lower())
    return " ".join(out)

def _clean_text(t: str, limit=6000) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()[:limit]

def _https_only(urls):
    return [u for u in (urls or []) if isinstance(u, str) and u.startswith("https://")]

def _gen_sku(prefix="ITEM"):
    ts = str(int(time.time() * 1000))
    return f"{prefix}-{ts[-8:]}"

def _strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</(p|li|h[1-6])>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(re.sub(r"\n{3,}", "\n\n", s)).strip()

def call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    if not OPENAI_API_KEY:
        raise NotImplementedError("OPENAI_API_KEY not set; LLM features disabled.")
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        sys_p = (system_prompt or "").strip()
        usr_p = (user_prompt or "").strip()
        if "json" not in sys_p.lower():
            sys_p = "Return a JSON object only. " + sys_p
        if "json" not in usr_p.lower():
            usr_p = usr_p + "\n\nReturn a JSON object only."
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
            ],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_p + "\nReturn only valid JSON. No prose."},
                    {"role": "user", "content": usr_p + "\nOnly valid JSON. No prose."},
                ],
                temperature=0.0,
            )
            txt = (resp.choices[0].message.content or "").strip()
            start = txt.find("{")
            end = txt.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise RuntimeError(f"LLM did not return JSON: {txt[:200]}")
            return json.loads(txt[start:end+1])
        except Exception as e2:
            raise RuntimeError(f"LLM JSON call failed: {e}\nFallback failed: {e2}")

def build_description_simple_from_raw(raw_text: str, html_mode: bool = True) -> Dict[str, str]:
    if html_mode:
        prompt = (
            "Return HTML only. Use ONLY <p>, <ul>, <li>, <br>, <strong>, <em> tags. "
            "No headings (h1–h6), no tables, no images, no scripts. "
            f"Write eBay product description for this product: {raw_text}"
        )
    else:
        prompt = (
            "Write eBay product description for this product (plain text only, "
            f"no headings, no bullet points, no bold): {raw_text}"
        )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        out = (resp.choices[0].message.content or "").strip()[:6000]
        if html_mode:
            html_desc = out
            text_desc = _strip_html(html_desc)
            return {"html": html_desc, "text": text_desc}
        return {"html": out, "text": out}
    except Exception:
        fallback = _clean_text(raw_text, limit=2000)
        return {"html": fallback if not html_mode else f"<p>{fallback}</p>", "text": fallback}

def _prompt_for_aspects(raw_text: str, images: List[str], cat_name: str):
    img_lines = ""
    if images:
        img_lines = "Image URLs provided:\n" + "\n".join(f"- {u}" for u in images) + "\n"
    return f"""
You are helping fill eBay listing item specifics (aspects) for category "{cat_name}".
Use ONLY the information from the product text below and the image URL filenames if helpful.
If a value is not present, leave it blank (do not invent). Prefer concise, canonical values.

{img_lines}
Product text (verbatim):
---
{raw_text}
---
Output a JSON object with keys = aspect names and values = either a string or array of strings.
Do not add extra keys. If unsure, set the value to an empty string "".
"""

def _postprocess_aspects(filled: Dict[str, Any]) -> Dict[str, List[str]]:
    out = {}
    for k, v in (filled or {}).items():
        if v is None:
            continue
        if isinstance(v, str):
            val = v.strip()
            if val:
                out[k] = [val]
        elif isinstance(v, list):
            vals = [str(x).strip() for x in v if str(x).strip()]
            if vals:
                out[k] = vals
    return out

def _constraint_map(aspects_raw: list) -> Dict[str, Dict[str, Any]]:
    out = {}
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

def apply_aspect_constraints(filled: Dict[str, List[str]], aspects_raw: list):
    cmap = _constraint_map(aspects_raw)
    adjusted = {}
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

def clean_keywords(keywords):
    cleaned = []
    for kw in keywords:
        kw = kw.strip()
        if len(kw) > MAX_LEN:
            kw = kw[:MAX_LEN].rsplit(" ", 1)[0]
        cleaned.append(kw)
    return cleaned

def _aspect_name(x: Any) -> Optional[str]:
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
    id_key = f"{kind}Policy vadaId"
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
    suggestions = r.json().get("categorySuggestions", [])
    for node in suggestions:
        cat = node.get("category") or {}
        if node.get("categoryTreeNodeLevel", 0) > 0 and node.get("leafCategoryTreeNode", True):
            return cat["categoryId"], cat["categoryName"]
    if suggestions:
        cat = suggestions[0]["category"]
        return cat["categoryId"], cat["categoryName"]
    raise RuntimeError("No category suggestions found")

def browse_majority_category(query: str, user_id: int):
    access = ensure_access_token(user_id)
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

def get_required_and_recommended_aspects(tree_id: str, category_id: str, user_id: int):
    access = ensure_access_token(user_id)
    url = f"{API}/commerce/taxonomy/v1/category_tree/{tree_id}/get_item_aspects_for_category"
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

def _fallback_title(raw_text: str) -> str:
    t = (raw_text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t[:80] if t else "Untitled Item"

# Routes
@app.route("/")
def index():
    return send_from_directory(app.template_folder, 'index.html')

@app.route("/signup.html")
def signup_page():
    return send_from_directory(app.template_folder, 'signup.html')

@app.route("/login.html")
def login_page():
    return send_from_directory(app.template_folder, 'login.html')

@app.route("/profile.html")
def profile_page():
    return send_from_directory(app.template_folder, 'profile.html')

@app.route("/ebay-auth.html")
def ebay_auth_page():
    return send_from_directory(app.template_folder, 'ebay-auth.html')

@app.route("/dashboard.html")
def dashboard_page():
    return send_from_directory(app.template_folder, 'dashboard.html')

@app.route("/success.html")
def success_page():
    return send_from_directory(app.template_folder, 'success.html')

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
            c.execute("INSERT INTO users (email, password_hash, is_active) VALUES (%s, %s, TRUE) RETURNING id",
                      (email, generate_password_hash(password)))
            user_id = c.fetchone()[0]
            conn.commit()
            session["user_id"] = user_id
            session["email"] = email
            return jsonify({"status": "success", "message": "User registered successfully"})
    except psycopg2.errors.UnAccountingError:
        conn.rollback()
        return jsonify({"error": "Email already exists"}), 400
    except psycopg2.Error:
        conn.rollback()
        return jsonify({"error": "Registration failed"}), 500
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
                return jsonify({"status": "success", "message": "Logged in successfully"})
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
        "country": "GB"
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

    return jsonify({
        "is_logged_in": True,
        "is_active": True,
        "email": session.get("email"),
        "has_profile": bool(profile),
        "has_ebay_auth": bool(tokens["refresh"]),
        "access_exp_in": max(0, int(tokens["exp"] - _now())) if tokens["access"] else 0
    })

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

    raw_text = _clean_text(body.get("raw_text"), limit=8000)
    images = _https_only(body.get("images"))
    price = body.get("price")
    quantity = int(body.get("quantity", 1))
    condition = (body.get("condition") or "NEW").upper()
    include_debug = bool(body.get("include_debug", False))
    use_html = bool(body.get("use_html_description", True))

    try:
        marketplace_id = MARKETPLACE_ID
        user_id = session["user_id"]

        # Step 1: Extract keywords
        system_prompt = (
            "You extract concise keywords for eBay category selection and search. "
            "Return STRICT JSON per the schema. Use ONLY facts present in the input. "
            "Do NOT invent identifiers; if absent, omit the field. "
            "Lowercase all keywords. No punctuation, no duplicates."
            "search_keywords must be ≤ 30 characters"
        )
        user_prompt = f"""MARKETPLACE: {marketplace_id}

RAW_TEXT:
{raw_text}

IMAGE_URLS (for context only, do not copy text from them):
{chr(10).join(images) if images else "(none)"}

OUTPUT RULES:
- category_keywords: 1–5 short phrases (2–3 words) that best describe the product category.
- search_keywords: 3–12 search terms buyers would type (mix of unigrams/bigrams/trigrams), all lowercase.
- All search_keywords must be ≤ 30 characters.
- normalized_title: <=80 chars, clean and factual (no emojis/promo).
- brand: only if explicitly present in RAW_TEXT.
- identifiers: only if explicitly present (isbn/ean/gtin/mpn)."""
        s1 = call_llm_json(system_prompt, user_prompt)
        s1["search_keywords"] = clean_keywords(s1.get("search_keywords", []))
        validate(instance=s1, schema=KEYWORDS_SCHEMA)

        normalized_title = s1.get("normalized_title") or _fallback_title(raw_text)
        category_keywords = s1.get("category_keywords") or []

        # Step 2: Category and aspects
        query = (" ".join(category_keywords)).strip() or normalized_title
        tree_id = get_category_tree_id()
        try:
            cat_id, cat_name = suggest_leaf_category(tree_id, query)
        except Exception:
            cat_id, cat_name = browse_majority_category(query, user_id)
            if not cat_id:
                return jsonify({"error": "No category found", "query": query}), 404

        aspects_info = get_required_and_recommended_aspects(tree_id, cat_id, user_id)
        req_aspects = aspects_info.get("required", [])
        rec_aspects = aspects_info.get("recommended", [])
        req_names = [n for n in (_aspect_name(x) for x in req_aspects) if n]
        rec_names = [n for n in (_aspect_name(x) for x in rec_aspects) if n]

        # Step 3: Fill aspects
        system_prompt2 = (
            "You fill eBay item aspects from provided text/images. NEVER leave required aspects empty; "
            "extract when explicit, infer when reasonable, otherwise use 'Does not apply'/'Unknown' where acceptable."
        )
        user_prompt2 = _prompt_for_aspects(normalized_title, images, cat_name)
        s3 = call_llm_json(system_prompt2, user_prompt2)
        validate(instance=s3, schema=ASPECTS_FILL_SCHEMA)

        allowed = set(req_names + rec_names)
        filled_aspects = {}
        for k, vals in (s3.get("filled") or {}).items():
            if k in allowed and isinstance(vals, list):
                clean_vals = list(dict.fromkeys([str(v).strip() for v in vals if str(v).strip()]))
                if clean_vals:
                    filled_aspects[k] = clean_vals

        filled_aspects = apply_aspect_constraints(filled_aspects, aspects_info.get("raw"))
        if "Book Title" in filled_aspects:
            filled_aspects["Book Title"] = [_trim_to_limit(v, 65) for v in filled_aspects["Book Title"]]

        # Step 4: Generate description
        desc_bundle = build_description_simple_from_raw(raw_text, html_mode=use_html)
        description_text = desc_bundle["text"]
        description_html = desc_bundle["html"] if use_html else description_text

        # Step 5: List item
        access = ensure_access_token(user_id)
        headers = {
            "Authorization": f"Bearer {access}",
            "Content-Type": "application/json",
            "Content-Language": LANG,
            "Accept-Language": LANG,
            "X-EBAY-C-MARKETPLACE-ID": marketplace_id,
        }

        sku = _gen_sku()
        title = smart_titlecase(normalized_title[:80])

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
            return jsonify({"error": "Failed to create inventory item", "details": r.text, "step": "inventory_item"}), 500

        fulfillment_policy_id = get_first_policy_id("fulfillment", access, marketplace_id)
        payment_policy_id = get_first_policy_id("payment", access, marketplace_id)
        return_policy_id = get_first_policy_id("return", access, marketplace_id)
        merchant_location_key = get_or_create_location(access, marketplace_id, profile)

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
        offer_url = f"{BASE}/sell/inventory/v1/offer"
        r = requests.post(offer_url, headers=headers, json=offer_payload, timeout=30)
        if r.status_code not in (200, 201):
            return jsonify({"error": "Failed to create offer", "details": r.text, "step": "create_offer"}), 500

        offer_id = r.json().get("offerId")
        pub_url = f"{BASE}/sell/inventory/v1/offer/{offer_id}/publish"
        r = requests.post(pub_url, headers=headers, timeout=30)
        if r.status_code not in (200, 201):
            return jsonify({"error": "Failed to publish listing", "details": r.text, "step": "publish", "offerId": offer_id}), 500

        pub = r.json()
        listing_id = pub.get("listingId")
        view_url = f"https://www.ebay.co.uk/itm/{listing_id}" if marketplace_id == "EBAY_GB" else None
        update_listing_count()

        result = {
            "status": "published",
            "offerId": offer_id,
            "listingId": listing_id,
            "viewItemUrl": view_url,
            "sku": sku,
            "title": title,
            "categoryName": cat_name,
            "aspects": filled_aspects
        }
        if include_debug:
            result["debug"] = {
                "step1_extract_keywords": s1,
                "step2_category_and_aspects": {
                    "queryUsed": query,
                    "categoryTreeId": tree_id,
                    "categoryId": cat_id,
                    "categoryName": cat_name,
                    "requiredAspects": req_aspects,
                    "recommendedAspects": rec_aspects
                },
                "step3_fill_aspects": s3,
                "description": {
                    "text": description_text,
                    "html": description_html,
                    "used_html": use_html
                }
            }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": "Publishing failed", "details": str(e)}), 500

@app.route("/total-listings")
def get_total_listings_route():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_user_active(session["user_id"]):
        return jsonify({"error": "Account is inactive"}), 403

    total = get_total_listings()
    return jsonify({"total_listings": total})

@app.route("/my-listings")
def get_my_listings():
    if "user_id" not in session:
        return jsonify({"error": "Please log in first"}), 401
    if not is_user_active(session["user_id"]):
        return jsonify({"error": "Account is inactive"}), 403

    return jsonify({"listings": []})

@app.errorhandler(404)
def not_found(error):
    return send_from_directory(app.template_folder, 'index.html')

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    if not all([CLIENT_ID, CLIENT_SECRET, RU_NAME, DB_URL, OPENAI_API_KEY]):
        print("Missing required environment variables (CLIENT_ID, CLIENT_SECRET, RU_NAME, SUPABASE_DB_URL, or OPENAI_API_KEY)")
        exit(1)

    app.run(host="127.0.0.1", port=5000, debug=True)
