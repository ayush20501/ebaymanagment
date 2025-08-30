# publish_logic.py

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

# Import environment variables
from dotenv import load_dotenv
load_dotenv()

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE = os.getenv("EBAY_BASE")
AUTH = os.getenv("EBAY_AUTH")
TOKEN = os.getenv("EBAY_TOKEN_URL")
API = os.getenv("EBAY_API")
MARKETPLACE_ID = os.getenv("EBAY_MARKETPLACE_ID")
LANG = os.getenv("EBAY_LANG")
CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")
RU_NAME = os.getenv("EBAY_RU_NAME")

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
        "use_simple_prompt_description": {"type": "boolean"},
        "include_debug": {"type": "boolean"}
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

SMALL_WORDS = {
    "a", "an", "the", "and", "or", "nor", "but", "for", "so", "yet",
    "at", "by", "in", "of", "on", "to", "up", "off", "as", "if",
    "per", "via", "vs", "vs."
}

MAX_LEN = 30

SCOPES = " ".join([
    "https://api.ebay.com/oauth/api_scope",
    "https://api.ebay.com/oauth/api_scope/sell.inventory",
    "https://api.ebay.com/oauth/api_scope/sell.account",
])

# Helper functions
def _now():
    return time.time()

def _b64_basic():
    return "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()

def clean_keywords(keywords):
    """Clean and truncate keywords to MAX_LEN"""
    cleaned = []
    for kw in keywords:
        kw = kw.strip()
        if len(kw) > MAX_LEN:
            kw = kw[:MAX_LEN].rsplit(" ", 1)[0]
        cleaned.append(kw)
    return cleaned

def call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    """Call OpenAI in JSON mode and return a dict."""
    if not OPENAI_API_KEY:
        raise NotImplementedError("OPENAI_API_KEY not set; LLM features disabled.")

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
            start = txt.find("{")
            end = txt.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise RuntimeError(f"LLM did not return JSON: {txt[:200]}")
            payload = txt[start:end+1]
            return json.loads(payload)
        except Exception as e2:
            raise RuntimeError(f"LLM JSON call failed: {e}\nFallback failed: {e2}")

def call_llm_text_simple(user_prompt: str, system_prompt: str = None) -> str:
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

def _aspect_name(x: Any) -> str:
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

def apply_aspect_constraints(filled: Dict[str, list], aspects_raw: list):
    """Apply eBay aspect constraints like max length"""
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

def _fallback_title(raw_text: str) -> str:
    """Generate fallback title from raw text"""
    t = (raw_text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return (t[:80]) if t else "Untitled Item"

def smart_titlecase(s: str) -> str:
    """Title-case words while keeping acronyms, numbers, and small words sane."""
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

def get_category_tree_id(access):
    r = requests.get(
        f"{API}/commerce/taxonomy/v1/get_default_category_tree_id",
        params={"marketplace_id": MARKETPLACE_ID},
        headers={"Authorization": f"Bearer {access}"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["categoryTreeId"]

def suggest_leaf_category(tree_id: str, query: str, access: str):
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
        if node.get("categoryTreeNodeLevel", 0) > 0 and node.get("leafCategoryTreeNode", True):
            return cat["categoryId"], cat["categoryName"]
    if suggestions:
        cat = suggestions[0]["category"]
        return cat["categoryId"], cat["categoryName"]
    raise RuntimeError("No category suggestions found")

def browse_majority_category(query: str, access: str):
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

def get_required_and_recommended_aspects(tree_id: str, category_id: str, access: str):
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

def parse_ebay_error(response_text):
    """Parse eBay API error response and return user-friendly message"""
    try:
        error_data = json.loads(response_text)
        if 'errors' in error_data:
            errors = error_data['errors']
            if errors and len(errors) > 0:
                first_error = errors[0]
                error_id = first_error.get('errorId')
                message = first_error.get('message', '')
                if error_id == 25002:
                    if 'identical items' in message.lower():
                        return "This item already exists in your eBay listings. eBay doesn't allow identical items from the same seller."
                elif error_id == 25001:
                    return "There was an issue with the product category. Please try with a different product description."
                elif error_id == 25003:
                    return "There's an issue with your eBay selling policies. Please check your eBay account settings."
                elif 'listing policies' in message.lower():
                    return "Your eBay account is missing required selling policies. Please set up payment, return, and shipping policies in your eBay account."
                elif 'inventory item' in message.lower():
                    return "Failed to create the product listing. Please check your product details and try again."
                else:
                    return message
        return f"eBay API error: {response_text}"
    except (json.JSONDecodeError, KeyError, TypeError):
        return f"Unknown eBay error: {response_text}"

def publish_item(session, body, db_pool, get_user_profile, get_user_tokens, ensure_access_token, save_user_listing, update_listing_count):
    """Main logic for publishing an item to eBay"""
    if "user_id" not in session:
        return {"error": "Please log in first"}, 401
    if not is_user_active(session["user_id"], db_pool):
        return {"error": "Account is inactive"}, 403

    profile = get_user_profile(session["user_id"])
    if not profile:
        return {"error": "Please create your profile first"}, 400

    tokens = get_user_tokens(session["user_id"])
    if not tokens["refresh"]:
        return {"error": "Please authenticate with eBay first"}, 400

    try:
        validate(instance=body, schema=LISTING_SCHEMA)
    except Exception as e:
        return {"error": "Invalid input", "details": str(e)}, 400
    
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
        return {"error": "Raw text or images required"}, 400
    
    try:
        # Step 1: AI Extract Keywords & Metadata
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

        # Step 2: Find eBay Category
        access = ensure_access_token(session["user_id"])
        tree_id = get_category_tree_id(access)
        query = (" ".join(category_keywords)).strip() or normalized_title
        try:
            cat_id, cat_name = suggest_leaf_category(tree_id, query, access)
        except Exception:
            cat_id, cat_name = browse_majority_category(query, access)
            if not cat_id:
                return {"error": "No category found from taxonomy or browse", "query": query}, 404

        # Step 3: Get Required/Recommended Aspects
        aspects_info = get_required_and_recommended_aspects(tree_id, cat_id, access)
        req_in = aspects_info.get("required", [])
        rec_in = aspects_info.get("recommended", [])
        req_names = [n for n in (_aspect_name(x) for x in req_in) if n]
        rec_names = [n for n in (_aspect_name(x) for x in rec_in) if n]

        # Step 4: AI Fill Aspects
        filled_aspects = {}
        if OPENAI_API_KEY and (req_names or rec_names):
            system_prompt2 = (
                "You fill eBay item aspects from provided text/images. NEVER leave required aspects empty; "
                "extract when explicit, infer when reasonable, otherwise use 'Does not apply'/'Unknown' where acceptable."
            )
            user_prompt2 = f"""
INPUT TEXT:
{normalized_title}

IMAGE_URLS (context only, do not OCR):
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

        # Step 5: AI Generate Description
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

        # Step 6: Create eBay Listing
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
            return {"error": error_msg, "step": "inventory_item"}, 400

        # Get policies
        try:
            fulfillment_policy_id = get_first_policy_id("fulfillment", access, marketplaceId)
            payment_policy_id = get_first_policy_id("payment", access, marketplaceId)
            return_policy_id = get_first_policy_id("return", access, marketplaceId)
        except RuntimeError as e:
            return {"error": f"Missing eBay policies: {str(e)}. Please set up your selling policies in your eBay account."}, 400
        
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
            return {"error": error_msg, "step": "create_offer"}, 400

        offer_id = r.json().get("offerId")
        
        # Publish listing
        pub_url = f"{BASE}/sell/inventory/v1/offer/{offer_id}/publish"
        r = requests.post(pub_url, headers=headers, timeout=30)
        if r.status_code not in (200, 201):
            error_msg = parse_ebay_error(r.text)
            return {"error": error_msg, "step": "publish"}, 400

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

        return result, 200

    except requests.exceptions.RequestException as e:
        return {"error": f"Network error while communicating with eBay: {str(e)}"}, 500
    except RuntimeError as e:
        return {"error": str(e)}, 400
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}, 500

def is_user_active(user_id, db_pool):
    conn = db_pool.getconn()
    try:
        with conn.cursor() as c:
            c.execute("SELECT is_active FROM users WHERE id = %s", (user_id,))
            user = c.fetchone()
            return user and user[0] == True
    finally:
        db_pool.putconn(conn)
