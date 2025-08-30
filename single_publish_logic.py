import os
import json
import re
import requests
import base64
from jsonschema import validate
from typing import Optional, Dict, Any, List
from collections import Counter
from openai import OpenAI
import time
import html
from dotenv import load_dotenv

load_dotenv()

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

SCOPES = " ".join([
    "https://api.ebay.com/oauth/api_scope",
    "https://api.ebay.com/oauth/api_scope/sell.inventory",
    "https://api.ebay.com/oauth/api_scope/sell.account",
])

LISTING_SCHEMA = {
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
    }
}

KEYWORDS_SCHEMA = {
    "type": "object",
    "required": ["category_keywords", "search_keywords"],
    "properties": {
        "category_keywords": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string", "minLength": 1, "maxLength": 40}},
        "search_keywords": {"type": "array", "minItems": 3, "maxItems": 12, "items": {"type": "string", "minLength": 1, "maxLength": 30}},
        "brand": {"type": "string"},
        "normalized_title": {"type": "string", "maxLength": 80}
    }
}

ASPECTS_FILL_SCHEMA = {
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
        }
    }
}

SMALL_WORDS = {
    "a", "an", "the", "and", "or", "nor", "but", "for", "so", "yet",
    "at", "by", "in", "of", "on", "to", "up", "off", "as", "if",
    "per", "via", "vs", "vs."
}

MAX_LEN = 30

def _b64_basic():
    return "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()

def _now():
    return time.time()

def clean_keywords(keywords):
    cleaned = []
    for kw in keywords:
        kw = kw.strip()
        if len(kw) > MAX_LEN:
            kw = kw[:MAX_LEN].rsplit(" ", 1)[0]
        cleaned.append(kw)
    return cleaned

def call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    if not OPENAI_API_KEY:
        raise NotImplementedError("OPENAI_API_KEY not set; LLM features disabled.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    sys_p = "Return a JSON object only. " + system_prompt.strip()
    usr_p = user_prompt.strip() + "\n\nReturn a JSON object only."
    try:
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
            resp2 = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_p + "\nReturn only valid JSON."},
                    {"role": "user", "content": usr_p + "\nOnly valid JSON."},
                ],
                temperature=0.0,
            )
            txt = resp2.choices[0].message.content.strip()
            start, end = txt.find("{"), txt.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise RuntimeError(f"LLM did not return JSON: {txt[:200]}")
            return json.loads(txt[start:end+1])
        except Exception as e2:
            raise RuntimeError(f"LLM JSON call failed: {e}\nFallback failed: {e2}")

def call_llm_text_simple(user_prompt: str, system_prompt: Optional[str] = None) -> str:
    if not OPENAI_API_KEY:
        raise NotImplementedError("OPENAI_API_KEY not set; LLM features disabled.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    msgs = [{"role": "system", "content": system_prompt}] if system_prompt else []
    msgs.append({"role": "user", "content": user_prompt})
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def build_description_simple_from_raw(raw_text: str) -> Dict[str, str]:
    prompt = (
        "Return HTML only. Use ONLY <p>, <ul>, <li>, <br>, <strong>, <em> tags. "
        "No headings, tables, images, or scripts. "
        f"Write eBay product description for: {raw_text}"
    )
    try:
        html_desc = call_llm_text_simple(prompt)[:6000].strip()
        text_desc = _strip_html(html_desc)
        return {"html": html_desc, "text": text_desc}
    except Exception:
        fallback = _clean_text(raw_text, limit=2000)
        return {"html": f"<p>{fallback}</p>", "text": fallback}

def _strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</(p|li|h[1-6])>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(re.sub(r"\n{3,}", "\n\n", s)).strip()

def _aspect_name(x: Any) -> Optional[str]:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("aspectName") or x.get("localizedAspectName") or x.get("name") or (x.get("aspect") or {}).get("name")
    return None

def apply_aspect_constraints(filled: Dict[str, List[str]], aspects_raw: list):
    cmap = {a.get("localizedAspectName") or (a.get("aspect") or {}).get("name"): {
        "max_len": (a.get("aspectConstraint", {}) or {}).get("aspectValueMaxLength"),
        "mode": a.get("aspectMode")
    } for a in aspects_raw or [] if a.get("localizedAspectName") or (a.get("aspect") or {}).get("name")}
    adjusted = {}
    for k, vals in filled.items():
        vlist = []
        max_len = cmap.get(k, {}).get("max_len")
        mode = cmap.get(k, {}).get("mode")
        for v in vals or []:
            nv = str(v).strip()
            if mode == "FREE_TEXT" and isinstance(max_len, int) and max_len > 0 and len(nv) > max_len:
                nv = nv[:max_len].rsplit(" ", 1)[0] if " " in nv[:max_len] else nv[:max_len]
            vlist.append(nv)
        if vlist:
            adjusted[k] = vlist
    return adjusted

def _fallback_title(raw_text: str) -> str:
    t = re.sub(r"\s+", " ", raw_text or "").strip()
    return t[:80] if t else "Untitled Item"

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

def _clean_text(t: str, limit=6000) -> str:
    return re.sub(r"\s+", " ", t or "").strip()[:limit]

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

def get_category_tree_id(access: str):
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

def parse_ebay_error(response_text):
    try:
        error_data = json.loads(response_text)
        if 'errors' in error_data:
            errors = error_data['errors']
            if errors and len(errors) > 0:
                first_error = errors[0]
                error_id = first_error.get('errorId')
                message = first_error.get('message', '')
                if error_id == 25002:
                    return "This item already exists in your eBay listings."
                elif error_id == 25001:
                    return "Issue with product category."
                elif error_id == 25003:
                    return "Issue with eBay selling policies."
                elif 'listing policies' in message.lower():
                    return "Missing required eBay selling policies."
                elif 'inventory item' in message.lower():
                    return "Failed to create product listing."
                return message
        return f"eBay API error: {response_text}"
    except (json.JSONDecodeError, KeyError, TypeError):
        return f"Unknown eBay error: {response_text}"

def publish_item(user_id, body, profile, tokens, save_user_listing, update_listing_count):
    if not profile:
        return {"error": "Please create your profile first"}, 400
    if not tokens["refresh"]:
        return {"error": "Please authenticate with eBay first"}, 400
    try:
        validate(instance=body, schema=LISTING_SCHEMA)
    except Exception as e:
        return {"error": "Invalid input", "details": str(e)}, 400
    
    raw_text_in = _clean_text(body.get("raw_text"), limit=8000)
    images = _https_only(body.get("images"))
    marketplace_id = MARKETPLACE_ID
    price = body.get("price")
    quantity = int(body.get("quantity", 1))
    condition = (body.get("condition") or "NEW").upper()
    
    if not raw_text_in and not images:
        return {"error": "Raw text or images required"}, 400
    
    try:
        access = ensure_access_token(user_id, tokens)
        if not OPENAI_API_KEY:
            normalized_title = smart_titlecase(raw_text_in[:80]) or _fallback_title(raw_text_in)
            category_keywords = []
            brand = None
        else:
            system_prompt = (
                "Extract concise keywords for eBay category selection and search. "
                "Return JSON per schema. Use ONLY input facts. "
                "Lowercase keywords, no punctuation, no duplicates."
            )
            user_prompt = f"""MARKETPLACE: {marketplace_id}
RAW_TEXT: {raw_text_in}
IMAGE_URLS: {chr(10).join(images) if images else "(none)"}
OUTPUT RULES:
- category_keywords: 1–5 short phrases (2–3 words) for product category.
- search_keywords: 3–12 search terms, all lowercase, ≤30 chars.
- normalized_title: ≤80 chars, clean, factual.
- brand: only if in RAW_TEXT."""
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

        tree_id = get_category_tree_id(access)
        query = (" ".join(category_keywords)).strip() or normalized_title
        try:
            cat_id, cat_name = suggest_leaf_category(tree_id, query, access)
        except Exception:
            cat_id, cat_name = browse_majority_category(query, access)
            if not cat_id:
                return {"error": "No category found", "query": query}, 404

        aspects_info = get_required_and_recommended_aspects(tree_id, cat_id, access)
        req_in = aspects_info.get("required", [])
        rec_in = aspects_info.get("recommended", [])
        req_names = [n for n in (_aspect_name(x) for x in req_in) if n]
        rec_names = [n for n in (_aspect_name(x) for x in rec_in) if n]

        filled_aspects = {}
        if OPENAI_API_KEY and (req_names or rec_names):
            system_prompt2 = (
                "Fill eBay item aspects from text/images. "
                "Never leave required aspects empty; use 'Unknown' if needed."
            )
            user_prompt2 = f"""
INPUT TEXT: {normalized_title}
IMAGE URLS: {chr(10).join(images) if images else "(none)"}
ASPECTS:
- REQUIRED: {req_names}
- RECOMMENDED: {rec_names}
OUTPUT: {{"filled": {{"AspectName": ["value1"]}}, "missing_required": []}}
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

        try:
            desc_bundle = build_description_simple_from_raw(raw_text_in)
            description_text = desc_bundle["text"]
            description_html = desc_bundle["html"]
        except Exception as e:
            print(f"[AI Description Error] {e}")
            description_text = raw_text_in[:2000]
            description_html = f"<p>{description_text}</p>"

        lang = "en-GB" if marketplace_id == "EBAY_GB" else "en-US"
        headers = {
            "Authorization": f"Bearer {access}",
            "Content-Type": "application/json",
            "Content-Language": lang,
            "Accept-Language": lang,
            "X-EBAY-C-MARKETPLACE-ID": marketplace_id,
        }

        sku = _gen_sku("RAW")
        title = smart_titlecase(normalized_title)[:80]

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
            return {"error": parse_ebay_error(r.text), "step": "inventory_item"}, 400

        try:
            fulfillment_policy_id = get_first_policy_id("fulfillment", access, marketplace_id)
            payment_policy_id = get_first_policy_id("payment", access, marketplace_id)
            return_policy_id = get_first_policy_id("return", access, marketplace_id)
        except RuntimeError as e:
            return {"error": f"Missing eBay policies: {str(e)}."}, 400
        
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
            return {"error": parse_ebay_error(r.text), "step": "create_offer"}, 400

        offer_id = r.json().get("offerId")
        pub_url = f"{BASE}/sell/inventory/v1/offer/{offer_id}/publish"
        r = requests.post(pub_url, headers=headers, timeout=30)
        if r.status_code not in (200, 201):
            return {"error": parse_ebay_error(r.text), "step": "publish"}, 400

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
            'category_id': cat_id,
            'category_name': cat_name,
            'marketplace_id': marketplace_id,
            'view_url': view_url
        }
        save_user_listing(user_id, listing_data)
        
        return {
            "result": {
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
            }
        }, 200

    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}"}, 500
    except RuntimeError as e:
        return {"error": str(e)}, 400
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}, 500

def ensure_access_token(user_id, tokens):
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
    from .app import save_user_tokens
    save_user_tokens(user_id, data["access_token"], data.get("refresh_token"), data["expires_in"])
    return data["access_token"]
