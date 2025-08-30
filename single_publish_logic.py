import json
import requests
import time
from jsonschema import validate
from typing import Dict, Any
from openai import OpenAI
from collections import Counter
import re
import html
import os
import hashlib
import base64

from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE = os.getenv("EBAY_BASE")
API = os.getenv("EBAY_API")
MARKETPLACE_ID = os.getenv("EBAY_MARKETPLACE_ID")
LANG = os.getenv("EBAY_LANG")
CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")


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
        "condition": {"type": "string", "enum": ["NEW", "USED", "REFURBISHED"]}
    },
    "additionalProperties": False
}

KEYWORDS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["category_keywords", "search_keywords", "normalized_title"],
    "properties": {
        "category_keywords": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string", "minLength": 1, "maxLength": 40}},
        "search_keywords": {"type": "array", "minItems": 3, "maxItems": 12, "items": {"type": "string", "minLength": 1, "maxLength": 30}},
        "brand": {"type": "string"},
        "normalized_title": {"type": "string", "maxLength": 80}
    },
    "additionalProperties": False
}

ASPECTS_FILL_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["filled"],
    "properties": {
        "filled": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1
            }
        }
    },
    "additionalProperties": False
}

SMALL_WORDS = {"a", "an", "the", "and", "or", "nor", "but", "for", "so", "yet", "at", "by", "in", "of", "on", "to", "up", "off", "as", "if", "per", "via", "vs", "vs."}
MAX_LEN = 30


def _clean_text(t: str, limit: int = 6000) -> str:
    """Clean and truncate text."""
    return re.sub(r"\s+", " ", (t or "")).strip()[:limit]

def _https_only(urls: list) -> list:
    """Filter URLs to include only HTTPS."""
    return [u for u in (urls or []) if isinstance(u, str) and u.startswith("https://")]

def _gen_sku(prefix: str = "ITEM") -> str:
    """Generate a unique SKU."""
    ts = str(int(time.time() * 1000))
    h = hashlib.sha1(ts.encode()).hexdigest()[:6].upper()
    return f"{prefix}-{ts[-6:]}-{h}"

def _b64_basic() -> str:
    """Generate Base64-encoded Basic Auth header."""
    return "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()

def clean_keywords(keywords: list) -> list:
    """Clean and truncate keywords to MAX_LEN."""
    return [kw.strip()[:MAX_LEN].rsplit(" ", 1)[0] if len(kw.strip()) > MAX_LEN else kw.strip() for kw in keywords]

def smart_titlecase(s: str) -> str:
    """Title-case string, preserving acronyms and small words."""
    if not s:
        return s
    words = s.strip().split()
    out = []
    for i, w in enumerate(words):
        if re.search(r"[A-Z]{2,}|\d[A-Za-z]|[A-Za-z]\d", w):
            out.append(w)
            continue
        lower = w.lower()
        if 0 < i < len(words) - 1 and lower in SMALL_WORDS and not re.search(r"[:–—-]$", out[-1] if out else ""):
            out.append(lower)
        else:
            parts = re.split(r"(-|/)", w)
            out.append("".join(p[:1].upper() + p[1:].lower() if p not in ("-", "/") else p for p in parts))
    if out:
        out[0] = out[0][:1].upper() + out[0][1:]
        out[-1] = out[-1][:1].upper() + out[-1][1:]
    return " ".join(out)

def _fallback_title(raw_text: str) -> str:
    """Generate fallback title from raw text."""
    t = _clean_text(raw_text)
    return t[:80] or "Untitled Item"

def call_llm_json(system_prompt: str, user_prompt: str) -> Dict:
    """Call OpenAI in JSON mode."""
    if not OPENAI_API_KEY:
        raise NotImplementedError("OPENAI_API_KEY not set; LLM features disabled.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    sys_p = f"Return a JSON object only. {(system_prompt or '').strip()}"
    usr_p = f"{(user_prompt or '').strip()}\n\nReturn a JSON object only."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        raise RuntimeError(f"LLM JSON call failed: {e}")

def call_llm_text(user_prompt: str, system_prompt: str = None) -> str:
    """Call OpenAI for text output."""
    if not OPENAI_API_KEY:
        raise NotImplementedError("OPENAI_API_KEY not set; LLM features disabled.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = [{"role": "user", "content": user_prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
    return (resp.choices[0].message.content or "").strip()[:6000]

def build_description(raw_text: str) -> Dict[str, str]:
    """Generate HTML and text descriptions."""
    system_prompt = (
        "Return HTML only. Use ONLY <p>, <ul>, <li>, <br>, <strong>, <em> tags. "
        "No headings (h1–h6), no tables, no images, no scripts."
    )
    user_prompt = f"Write eBay product description for this product: {raw_text}"
    try:
        html_desc = call_llm_text(user_prompt, system_prompt)
        text_desc = re.sub(r"<br\s*/?>", "\n", html_desc, flags=re.I)
        text_desc = re.sub(r"</(p|li|h[1-6])>", "\n", text_desc, flags=re.I)
        text_desc = re.sub(r"<[^>]+>", "", text_desc)
        text_desc = html.unescape(re.sub(r"\n{3,}", "\n\n", text_desc)).strip()
        return {"html": html_desc, "text": text_desc}
    except Exception:
        fallback = _clean_text(raw_text, 2000)
        return {"html": f"<p>{fallback}</p>", "text": fallback}

def get_category_tree_id(access: str) -> str:
    """Fetch eBay category tree ID."""
    r = requests.get(
        f"{API}/commerce/taxonomy/v1/get_default_category_tree_id",
        params={"marketplace_id": MARKETPLACE_ID},
        headers={"Authorization": f"Bearer {access}"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["categoryTreeId"]

def suggest_leaf_category(tree_id: str, query: str, access: str) -> tuple[str, str]:
    """Suggest a leaf category for the item."""
    r = requests.get(
        f"{API}/commerce/taxonomy/v1/category_tree/{tree_id}/get_category_suggestions",
        params={"q": query},
        headers={"Authorization": f"Bearer {access}"},
        timeout=30,
    )
    r.raise_for_status()
    suggestions = (r.json() or {}).get("categorySuggestions", [])
    for node in suggestions:
        cat = node.get("category") or {}
        if node.get("categoryTreeNodeLevel", 0) > 0 and node.get("leafCategoryTreeNode", True):
            return cat["categoryId"], cat["categoryName"]
    if suggestions:
        cat = suggestions[0]["category"]
        return cat["categoryId"], cat["categoryName"]
    raise RuntimeError("No category suggestions found")

def browse_majority_category(query: str, access: str) -> tuple[str, None]:
    """Determine category by browsing similar items."""
    r = requests.get(
        f"{API}/buy/browse/v1/item_summary/search",
        params={"q": query, "limit": 50},
        headers={"Authorization": f"Bearer {access}", "X-EBAY-C-MARKETPLACE-ID": MARKETPLACE_ID},
        timeout=30,
    )
    r.raise_for_status()
    items = (r.json() or {}).get("itemSummaries", [])
    cats = [it.get("categoryId") for it in items if it.get("categoryId")]
    if not cats:
        return None, None
    return Counter(cats).most_common(1)[0][0], None

def get_aspects(tree_id: str, category_id: str, access: str) -> Dict:
    """Fetch required and recommended aspects for a category."""
    url = f"{API}/commerce/taxonomy/v1/category_tree/{tree_id}/get_item_aspects_for_category"
    r = requests.get(
        url,
        params={"category_id": category_id},
        headers={"Authorization": f"Bearer {access}", "Accept-Language": LANG, "X-EBAY-C-MARKETPLACE-ID": MARKETPLACE_ID},
        timeout=30,
    )
    r.raise_for_status()
    aspects = r.json().get("aspects", [])
    required, recommended = [], []
    for a in aspects:
        name = a.get("localizedAspectName") or (a.get("aspect", {}) or {}).get("name")
        if not name:
            continue
        if a.get("aspectConstraint", {}).get("aspectRequired"):
            required.append(name)
        elif a.get("aspectConstraint", {}).get("aspectUsage") == "RECOMMENDED":
            recommended.append(name)
    return {"required": required, "recommended": recommended, "raw": aspects}

def apply_aspect_constraints(filled: Dict[str, list], aspects_raw: list) -> Dict[str, list]:
    """Apply eBay aspect constraints."""
    constraint_map = {
        a.get("localizedAspectName") or (a.get("aspect", {}) or {}).get("name"): {
            "max_len": (a.get("aspectConstraint", {}) or {}).get("aspectValueMaxLength"),
            "mode": (a.get("aspectConstraint", {}) or {}).get("aspectMode")
        }
        for a in aspects_raw or []
        if a.get("localizedAspectName") or (a.get("aspect", {}) or {}).get("name")
    }
    adjusted = {}
    for k, vals in filled.items():
        vlist = []
        max_len = constraint_map.get(k, {}).get("max_len")
        mode = constraint_map.get(k, {}).get("mode")
        for v in vals or []:
            nv = str(v).strip()
            if mode == "FREE_TEXT" and isinstance(max_len, int) and max_len > 0 and len(nv) > max_len:
                nv = nv[:max_len].rsplit(" ", 1)[0] if " " in nv[:max_len] else nv[:max_len]
            vlist.append(nv)
        if vlist:
            adjusted[k] = vlist
    return adjusted

def get_policy_id(kind: str, access: str, marketplace: str) -> str:
    """Fetch the first policy ID of a given kind."""
    r = requests.get(
        f"{BASE}/sell/account/v1/{kind}_policy",
        headers={"Authorization": f"Bearer {access}", "Accept-Language": LANG, "Content-Language": LANG, "X-EBAY-C-MARKETPLACE-ID": marketplace},
        params={"marketplace_id": marketplace},
        timeout=30,
    )
    r.raise_for_status()
    policies = r.json().get(f"{kind}Policies", [])
    if not policies:
        raise RuntimeError(f"No {kind} policies found in {marketplace}.")
    return policies[0][f"{kind}PolicyId"]

def get_or_create_location(access: str, marketplace: str, profile: Dict) -> str:
    """Get or create a merchant location."""
    headers = {"Authorization": f"Bearer {access}", "Accept-Language": LANG, "Content-Language": LANG, "X-EBAY-C-MARKETPLACE-ID": marketplace}
    r = requests.get(f"{BASE}/sell/inventory/v1/location", headers=headers, timeout=30)
    r.raise_for_status()
    if locs := r.json().get("locations", []):
        return locs[0]["merchantLocationKey"]
    
    merchant_location_key = "PRIMARY_LOCATION"
    payload = {
        "name": "Primary Warehouse",
        "location": {"address": {"addressLine1": profile['address_line1'], "city": profile['city'], "postalCode": profile['postal_code'], "country": profile['country']}},
        "locationType": "WAREHOUSE",
        "merchantLocationStatus": "ENABLED",
    }
    r = requests.post(f"{BASE}/sell/inventory/v1/location/{merchant_location_key}", headers={**headers, "Content-Type": "application/json"}, json=payload, timeout=30)
    r.raise_for_status()
    return merchant_location_key

def parse_ebay_error(response_text: str) -> str:
    """Parse eBay API error response."""
    try:
        error_data = json.loads(response_text)
        if errors := error_data.get('errors', []):
            error = errors[0]
            error_id = error.get('errorId')
            message = error.get('message', '')
            if error_id == 25002 and 'identical items' in message.lower():
                return "This item already exists in your eBay listings."
            if error_id == 25001:
                return "Issue with product category. Try a different description."
            if error_id == 25003:
                return "Issue with eBay selling policies. Check your account settings."
            if 'listing policies' in message.lower():
                return "Missing required selling policies. Set up payment, return, and shipping policies."
            if 'inventory item' in message.lower():
                return "Failed to create product listing. Check details and retry."
            return message
        return f"eBay API error: {response_text}"
    except (json.JSONDecodeError, KeyError, TypeError):
        return f"Unknown eBay error: {response_text}"

def extract_keywords(raw_text: str, images: list) -> tuple[str, list, str]:
    """Extract keywords and metadata using AI or fallback."""
    if not OPENAI_API_KEY:
        return smart_titlecase(_clean_text(raw_text, 80)) or _fallback_title(raw_text), [], None
    
    system_prompt = (
        "Extract concise keywords for eBay category and search. Return JSON per schema. "
        "Use only input facts. Lowercase keywords, no punctuation, no duplicates."
    )
    user_prompt = f"""MARKETPLACE: {MARKETPLACE_ID}
RAW_TEXT: {raw_text}
IMAGE_URLS: {chr(10).join(images) if images else '(none)'}
OUTPUT RULES:
- category_keywords: 1–5 phrases (2–3 words) for product category.
- search_keywords: 3–12 terms (unigrams/bigrams/trigrams), ≤30 chars.
- normalized_title: ≤80 chars, clean, factual.
- brand: only if explicit in RAW_TEXT."""
    try:
        data = call_llm_json(system_prompt, user_prompt)
        validate(instance=data, schema=KEYWORDS_SCHEMA)
        return (
            data.get("normalized_title") or _fallback_title(raw_text),
            clean_keywords(data.get("category_keywords", [])),
            data.get("brand")
        )
    except Exception as e:
        print(f"[AI Keywords Error] {e}")
        return smart_titlecase(_clean_text(raw_text, 80)) or _fallback_title(raw_text), [], None

def fill_aspects(normalized_title: str, images: list, required: list, recommended: list, aspects_raw: list) -> Dict[str, list]:
    """Fill eBay item aspects using AI or fallback."""
    if not OPENAI_API_KEY or not (required or recommended):
        return {name: ["Unknown"] for name in required}
    
    system_prompt = (
        "Fill eBay item aspects from text/images. Never leave required aspects empty; "
        "extract explicit values, infer reasonably, or use 'Unknown'."
    )
    user_prompt = f"""
INPUT TEXT: {normalized_title}
IMAGE_URLS: {chr(10).join(images) if images else '(none)'}
ASPECTS:
- REQUIRED: {required}
- RECOMMENDED: {recommended}
OUTPUT: {{"filled": {{"AspectName": ["value1", "value2"]}}}}"""
    try:
        data = call_llm_json(system_prompt, user_prompt)
        validate(instance=data, schema=ASPECTS_FILL_SCHEMA)
        filled = {k: list(dict.fromkeys([str(v).strip() for v in vals if str(v).strip()]))
                  for k, vals in (data.get("filled") or {}).items()
                  if k in set(required + recommended) and isinstance(vals, list)}
        filled = apply_aspect_constraints(filled, aspects_raw)
        if "Book Title" in filled:
            filled["Book Title"] = [v[:65] for v in filled["Book Title"]]
        return filled or {name: ["Unknown"] for name in required}
    except Exception as e:
        print(f"[AI Aspects Error] {e}")
        return {name: ["Unknown"] for name in required}

def is_user_active(user_id: str, db_pool) -> bool:
    """Check if user is active."""
    conn = db_pool.getconn()
    try:
        with conn.cursor() as c:
            c.execute("SELECT is_active FROM users WHERE id = %s", (user_id,))
            user = c.fetchone()
            return bool(user and user[0])
    finally:
        db_pool.putconn(conn)

def publish_item(session, body: Dict, db_pool, get_user_profile, get_user_tokens, ensure_access_token, save_user_listing, update_listing_count) -> tuple[Dict, int]:
    """Publish an item to eBay."""
    if "user_id" not in session:
        return {"error": "Please log in first"}, 401
    if not is_user_active(session["user_id"], db_pool):
        return {"error": "Account is inactive"}, 403

    profile = get_user_profile(session["user_id"])
    if not profile:
        return {"error": "Please create your profile first"}, 400

    if not get_user_tokens(session["user_id"])["refresh"]:
        return {"error": "Please authenticate with eBay first"}, 400

    try:
        validate(instance=body, schema=LISTING_SCHEMA)
    except Exception as e:
        return {"error": "Invalid input", "details": str(e)}, 400

    raw_text = _clean_text(body.get("raw_text"), 8000)
    images = _https_only(body.get("images"))
    if not raw_text and not images:
        return {"error": "Raw text or images required"}, 400

    price = body.get("price")
    quantity = int(body.get("quantity", 1))
    condition = (body.get("condition") or "NEW").upper()
    marketplace_id = MARKETPLACE_ID
    lang = "en-GB" if marketplace_id == "EBAY_GB" else "en-US"

    try:
        # Extract keywords
        normalized_title, category_keywords, brand = extract_keywords(raw_text, images)

        # Find category
        access = ensure_access_token(session["user_id"])
        tree_id = get_category_tree_id(access)
        query = " ".join(category_keywords).strip() or normalized_title
        try:
            cat_id, cat_name = suggest_leaf_category(tree_id, query, access)
        except Exception:
            cat_id, cat_name = browse_majority_category(query, access)
            if not cat_id:
                return {"error": f"No category found for query: {query}"}, 404

        # Get aspects
        aspects_info = get_aspects(tree_id, cat_id, access)
        filled_aspects = fill_aspects(normalized_title, images, aspects_info["required"], aspects_info["recommended"], aspects_info["raw"])

        # Generate description
        desc_bundle = build_description(raw_text)
        description_text, description_html = desc_bundle["text"], desc_bundle["html"]

        # Create eBay listing
        headers = {
            "Authorization": f"Bearer {access}",
            "Content-Type": "application/json",
            "Content-Language": lang,
            "Accept-Language": lang,
            "X-EBAY-C-MARKETPLACE-ID": marketplace_id,
        }
        sku = _gen_sku("RAW")
        title = smart_titlecase(normalized_title)[:80]

        # Create inventory item
        r = requests.put(
            f"{BASE}/sell/inventory/v1/inventory_item/{sku}",
            headers=headers,
            json={
                "product": {"title": title, "description": description_text, "aspects": filled_aspects, "imageUrls": images},
                "condition": condition,
                "availability": {"shipToLocationAvailability": {"quantity": quantity}}
            },
            timeout=30,
        )
        if r.status_code not in (200, 201, 204):
            return {"error": parse_ebay_error(r.text), "step": "inventory_item"}, 400

        # Get policies and location
        try:
            fulfillment_policy_id = get_policy_id("fulfillment", access, marketplace_id)
            payment_policy_id = get_policy_id("payment", access, marketplace_id)
            return_policy_id = get_policy_id("return", access, marketplace_id)
            merchant_location_key = get_or_create_location(access, marketplace_id, profile)
        except RuntimeError as e:
            return {"error": f"Missing eBay policies: {str(e)}"}, 400

        # Create offer
        r = requests.post(
            f"{BASE}/sell/inventory/v1/offer",
            headers=headers,
            json={
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
            },
            timeout=30,
        )
        if r.status_code not in (200, 201):
            return {"error": parse_ebay_error(r.text), "step": "create_offer"}, 400

        offer_id = r.json().get("offerId")

        # Publish listing
        r = requests.post(f"{BASE}/sell/inventory/v1/offer/{offer_id}/publish", headers=headers, timeout=30)
        if r.status_code not in (200, 201):
            return {"error": parse_ebay_error(r.text), "step": "publish"}, 400

        pub = r.json()
        listing_id = pub.get("listingId") or (pub.get("listingIds") or [None])[0]
        view_url = f"https://www.ebay.co.uk/itm/{listing_id}" if marketplace_id == "EBAY_GB" else None

        # Save listing and update stats
        update_listing_count()
        save_user_listing(session["user_id"], {
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
            'marketplace_id': marketplace_id,
            'view_url': view_url
        })

        return {
            "status": "published",
            "offerId": offer_id,
            "listingId": listing_id,
            "viewItemUrl": view_url,
            "sku": sku,
            "marketplaceId": marketplace_id,
            "categoryId": cat_id,
            "categoryName": cat_name,
            "title": title,
            "aspects": filled_aspects
        }, 200

    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}"}, 500
    except RuntimeError as e:
        return {"error": str(e)}, 400
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}, 500
