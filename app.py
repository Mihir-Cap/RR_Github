import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import re
import json
import time
import html
import base64
import difflib
import datetime
import urllib.parse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# 1. PAGE CONFIG + THEME
# =====================================================================
st.set_page_config(page_title="SupeRR Selector | Explorer", layout="wide")

NAVY = "#0C1130"
PINK = "#E91E8C"
TEAL = "#10B9A0"
BG = "#F1F3F9"
CARD_BORDER = "#E5E8F2"
MUTED = "#64748B"


# ---- optional brand assets: put files in an "assets" folder next to this app ----
#   assets/rr_logo.png   (or .svg/.jpg/.webp)  -> official Rajasthan Royals logo
#   assets/rr_font.ttf   (or .otf/.woff2)      -> official Rajasthan Royals font
def _find_asset(names):
    for folder in (os.path.join(APP_DIR, "assets"), APP_DIR):
        for n in names:
            p = os.path.join(folder, n)
            if os.path.isfile(p):
                return p
    return None


def _data_uri(path):
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp",
            "svg": "image/svg+xml", "ttf": "font/ttf", "otf": "font/otf", "woff": "font/woff",
            "woff2": "font/woff2"}.get(ext, "application/octet-stream")
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


_font_path = _find_asset(["rr_font.ttf", "rr_font.otf", "rr_font.woff2", "rr_font.woff"])
if _font_path:
    RR_FONT_STACK = "'RRBrand', 'Cinzel', serif"
    _font_face = f"@font-face {{ font-family: 'RRBrand'; src: url({_data_uri(_font_path)}); }}"
else:
    RR_FONT_STACK = "'Cinzel', serif"
    _font_face = ""

st.markdown(
    "<style>@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&display=swap');"
    + _font_face + "</style>",
    unsafe_allow_html=True,
)

CSS = """
<style>
    .stApp { background-color: __BG__; color: __NAVY__; font-family: 'Inter', sans-serif; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.4rem; max-width: 1220px; }
    h1, h2, h3, h4, h5, h6 { color: __NAVY__; }
    [data-testid="stMarkdownContainer"] p { color: __NAVY__; }

    /* ---------- Hero banner ---------- */
    .hero { background: __NAVY__; border-radius: 16px; padding: 22px 28px 24px 28px; }
    .hero-top { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
    .hero-left { display: flex; align-items: center; gap: 14px; }
    .rr-logo { background: #FFFFFF; border-radius: 50%; width: 60px; height: 60px; display: flex;
               align-items: center; justify-content: center; overflow: hidden; flex-shrink: 0; }
    .rr-logo img { width: 100%; height: 100%; object-fit: cover; object-position: center; border-radius: 50%; }
    .rr-logo-fallback { background: __PINK__; color: #fff; font-weight: 700; font-size: 20px; font-family: __RRFONT__; }
    .rr-name { font-family: __RRFONT__; font-size: 26px; font-weight: 700; color: #FFFFFF; letter-spacing: .6px; line-height: 1.1; }
    .hero-right { text-align: right; }
    .hero-brand { color: #FFFFFF; font-weight: 700; font-size: 19px; line-height: 1.15; }
    .hero-sub { color: #94A3B8; font-size: 12px; letter-spacing: .6px; margin-top: 2px; }
    .stApp .hero h1 { color: #FFFFFF; margin: 0 0 6px 0; padding: 0; font-size: 30px; font-weight: 700; }
    .stApp .hero p { color: #C7CEE6; margin: 0; font-size: 14px; }

    /* ---------- Metric cards ---------- */
    .metric-container { display: flex; gap: 14px; margin: 16px 0 6px 0; }
    .metric-card { background: #FFFFFF; border-radius: 14px; padding: 18px 20px; flex: 1; border: 1px solid __CARD_BORDER__; }
    .metric-val { font-size: 28px; font-weight: 700; margin-bottom: 4px; }
    .metric-val.magenta { color: __PINK__; }
    .metric-val.dark { color: __NAVY__; }
    .metric-val.teal { color: __TEAL__; }
    .metric-label { font-size: 13px; color: __MUTED__; }
    .footnote { font-size: 12px; color: __MUTED__; margin: 0 0 18px 4px; }

    /* ---------- About card ---------- */
    .info-card { background: #FFFFFF; border: 1px solid __CARD_BORDER__; border-radius: 14px; padding: 22px 26px; margin-bottom: 18px; }
    .info-card p { font-size: 14.5px; line-height: 1.65; color: #334155; margin: 0 0 8px 0; }
    .info-card b { color: __NAVY__; }

    /* ---------- Config panel ---------- */
    .st-key-config_card { background: #FFFFFF; border: 1px solid __CARD_BORDER__; border-radius: 14px;
                          padding: 22px 26px; margin-bottom: 22px; }
    .config-header { font-size: 19px; font-weight: 700; color: __NAVY__; margin-bottom: 2px; }
    .config-subheader { font-size: 13.5px; color: __MUTED__; margin-bottom: 14px; line-height: 1.55; }
    .ctl-label { font-size: 15px; font-weight: 700; color: __NAVY__; margin: 14px 0 3px 0; }
    .ctl-help { font-size: 13px; color: __MUTED__; line-height: 1.55; margin: 0 0 6px 0; }
    .legend { display: flex; justify-content: space-between; font-size: 12.5px; font-weight: 600; margin-top: -2px; }
    .explain-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 10px 0 10px 0; }
    .explain-box { background: #F6F7FC; border: 1.5px solid transparent; border-radius: 10px; padding: 12px 14px;
                   font-size: 13px; line-height: 1.55; color: #475569; }
    .explain-box b { color: __NAVY__; }
    .explain-box.active { border-color: __TEAL__; background: #E6F8F5; }

    [data-testid="stRadio"] div[role="radiogroup"] { display: flex; gap: 12px; width: 100%; }
    [data-testid="stRadio"] div[role="radiogroup"] > label {
        flex: 1; background: #F6F7FC; border: 1.5px solid #E1E5F0; border-radius: 12px;
        padding: 12px 16px; margin: 0; cursor: pointer; }
    [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child { display: none; }
    [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        border-color: __TEAL__; background: #E6F8F5; box-shadow: 0 0 0 3px rgba(16,185,160,.15); }
    [data-testid="stRadio"] label p { color: __NAVY__; font-weight: 600; font-size: 14px; }

    [data-testid="stTickBar"] { display: none; }
    [data-testid="stSliderThumbValue"] { color: __PINK__ !important; font-weight: 700; }
    [data-baseweb="slider"] [role="slider"] { background-color: __PINK__ !important;
        box-shadow: 0 0 0 4px rgba(233,30,140,.18) !important; border: 2px solid #FFFFFF !important; }

    /* ---------- Player rows (bigger) ---------- */
    [data-testid="stExpander"] { background: #FFFFFF; border: 1px solid __CARD_BORDER__ !important;
                                 border-radius: 14px; margin-bottom: 12px; }
    [data-testid="stExpander"] summary { padding: 1.1rem 1.3rem !important; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span { color: __NAVY__ !important; font-weight: 700; font-size: 19px !important; }
    .list-title { font-size: 27px; font-weight: 800; margin: 6px 0 12px 0; }

    /* ---------- Player profile content ---------- */
    .p-head { display: flex; gap: 20px; margin-bottom: 12px; align-items: center; }
    .p-photo { width: 120px; height: 120px; border-radius: 16px; object-fit: cover; object-position: center 18%; flex-shrink: 0; background: #F6F7FC; }
    .p-name { margin: 0; font-size: 24px; font-weight: 800; color: __NAVY__; }
    .p-role { color: __MUTED__; font-size: 14px; margin: 2px 0 7px 0; }
    .selected-badge { background-color: __TEAL__; color: #FFFFFF; padding: 3px 11px; border-radius: 999px;
                      font-size: 11.5px; font-weight: 700; display: inline-block; }
    .p-bio { font-size: 13px; color: #475569; font-style: italic; line-height: 1.55; margin: 4px 0 8px 0; }
    .stat-row { display: flex; gap: 10px; margin: 10px 0 6px 0; }
    .stat-tile { flex: 1; background: #F6F7FC; border-radius: 10px; padding: 11px 6px; text-align: center; }
    .stat-val { font-size: 21px; font-weight: 700; }
    .stat-lbl { font-size: 11.5px; color: __MUTED__; }
    .profile-table { width: 100%; border-collapse: collapse; margin: 8px 0 14px 0; }
    .profile-table th { text-align: left; padding: 9px 8px; border-bottom: 1px solid __CARD_BORDER__;
                        color: __NAVY__; font-size: 13px; font-weight: 700; background: transparent; }
    .profile-table td { padding: 8px 8px; border-bottom: 1px solid #EEF0F6; color: #475569; font-size: 13.5px; }
    .profile-table td:last-child { text-align: right; color: __NAVY__; font-weight: 600; }
    .profile-table tr:last-child td { border-bottom: none; }
    .p-src { font-size: 11.5px; color: __MUTED__; line-height: 1.5; }
    .p-src a { color: __PINK__; }

    .pill-green { background: #D9F6E5; color: #0B8F4D; font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 999px; }
    .sec-row { display: flex; justify-content: space-between; align-items: center; margin: 4px 0 10px 0; }
    .sec-title { font-size: 20px; font-weight: 700; color: __NAVY__; }
    .disclaimer { margin-top: 28px; padding: 18px 24px; border-top: 1px solid __CARD_BORDER__; color: __MUTED__; font-size: 12.5px; line-height: 1.7; text-align: center; }
    .disclaimer b { color: __NAVY__; }
    hr { border-color: __CARD_BORDER__; }
    [data-testid="stProgressBar"] [role="progressbar"] > div {
        background-color: __PINK__ !important;
    }
</style>
"""
for _k, _v in {"__NAVY__": NAVY, "__PINK__": PINK, "__TEAL__": TEAL, "__BG__": BG,
               "__CARD_BORDER__": CARD_BORDER, "__MUTED__": MUTED, "__RRFONT__": RR_FONT_STACK}.items():
    CSS = CSS.replace(_k, _v)
st.markdown(CSS, unsafe_allow_html=True)


# =====================================================================
# 2. LIVE INTERNET LAYER (CRICKET SOURCES + INDEPENDENT IMAGE RETRIEVAL)
# =====================================================================
# No Wikipedia or Wikimedia endpoints are used anywhere in this section.
# Player biography/profile retrieval and photograph retrieval are independent:
# a missing biography does not prevent the image engine from running.

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS = None
    DDGS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    fuzz = None
    RAPIDFUZZ_AVAILABLE = False

from concurrent.futures import as_completed
from io import BytesIO
import hashlib
import threading

IPL_TEAM_KEYS = [
    "super kings", "mumbai indians", "royal challengers", "knight riders",
    "delhi capitals", "delhi daredevils", "punjab kings", "kings xi",
    "rajasthan royals", "sunrisers", "deccan chargers", "gujarat titans",
    "gujarat lions", "super giants", "supergiant", "pune warriors",
    "kochi tuskers",
]

# Domains are used to rank search results, not to scrape their websites.
TRUSTED_CRICKET_DOMAINS = (
    "iplt20.com", "bcci.tv", "icc-cricket.com", "espncricinfo.com",
    "cricbuzz.com", "wisden.com", "cricketarchive.com", "howstat.com",
)
IMAGE_SOURCE_DOMAINS = (
    "iplt20.com", "bcci.tv", "icc-cricket.com", "espncricinfo.com",
    "cricbuzz.com", "wisden.com",
)
IMAGE_MAX_BYTES = 5 * 1024 * 1024
IMAGE_MAX_SIDE = 512
IMAGE_MIN_SCORE = 58
IMAGE_VERIFIED_SCORE = 75
PROFILE_CACHE_VERSION = "cricket_profile_images_v3"
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "").strip()

def _is_ipl_team(club):
    c = str(club or "").lower()
    return any(k in c for k in IPL_TEAM_KEYS)


def _clean_search_text(value):
    if value is None:
        return ""
    value = html.unescape(str(value))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _domain_of(url):
    try:
        return urllib.parse.urlparse(url or "").netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _is_blocked_image_source(url):
    domain = _domain_of(url)
    return (
        domain == "wikipedia.org" or domain.endswith(".wikipedia.org")
        or domain == "wikimedia.org" or domain.endswith(".wikimedia.org")
    )


def _trusted_cricket_result(item):
    url = item.get("url") or item.get("href") or ""
    if _is_blocked_image_source(url):
        return False
    domain = _domain_of(url)
    return any(domain == d or domain.endswith("." + d) for d in TRUSTED_CRICKET_DOMAINS)


def _ddgs_text(query, max_results=6):
    if not DDGS_AVAILABLE:
        return []
    try:
        searcher = DDGS(timeout=9)
        results = searcher.text(
            query, region="in-en", safesearch="moderate",
            max_results=max_results,
        )
        normalized = []
        for item in results or []:
            url = item.get("href") or item.get("url") or ""
            if not url or _is_blocked_image_source(url):
                continue
            normalized.append({
                "title": _clean_search_text(item.get("title")),
                "url": url,
                "snippet": _clean_search_text(item.get("body") or item.get("snippet")),
                "source": _domain_of(url),
            })
        return normalized
    except Exception:
        return []


def _ddgs_extract(url):
    if not DDGS_AVAILABLE or not url or _is_blocked_image_source(url):
        return ""
    try:
        searcher = DDGS(timeout=9)
        result = searcher.extract(url, fmt="text_plain")
        return _clean_search_text(result.get("content", "")) if isinstance(result, dict) else ""
    except Exception:
        return ""


def _best_search_result(results, name):
    if not results:
        return None
    target = re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()
    scored = []
    for item in results:
        title = re.sub(r"[^a-z0-9 ]", "", item.get("title", "").lower())
        snippet = item.get("snippet", "").lower()
        ratio = difflib.SequenceMatcher(None, target, title).ratio()
        if target and target in title:
            ratio += 0.5
        if any(word in snippet for word in ("cricketer", "batter", "bowler", "ipl", "t20")):
            ratio += 0.15
        if _trusted_cricket_result(item):
            ratio += 0.2
        scored.append((ratio, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _extract_labelled_value(text, labels):
    text = _clean_search_text(text)
    for label in labels:
        match = re.search(
            rf"\b{label}\b\s*(?:[:=\-]|\s)\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            text, flags=re.I,
        )
        if match:
            return match.group(1)
    return None


def _find_debut(text, patterns):
    text = _clean_search_text(text)
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1).strip(" ,.;")
    return None


def _name_similarity(name, candidate_text):
    target = re.sub(r"[^a-z0-9 ]", " ", name.lower()).strip()
    candidate = re.sub(r"[^a-z0-9 ]", " ", (candidate_text or "").lower()).strip()
    if not target or not candidate:
        return 0.0
    if RAPIDFUZZ_AVAILABLE:
        return float(fuzz.token_set_ratio(target, candidate)) / 100.0
    return difflib.SequenceMatcher(None, target, candidate).ratio()


def _image_candidates_ddgs(name, query):
    if not DDGS_AVAILABLE:
        return []
    try:
        searcher = DDGS(timeout=9)
        try:
            results = searcher.images(
                query, region="in-en", safesearch="moderate",
                max_results=10, type_image="photo",
            )
        except TypeError:
            results = searcher.images(
                query, region="in-en", safesearch="moderate",
                max_results=10,
            )
        candidates = []
        for item in results or []:
            image_url = item.get("image") or item.get("original")
            page_url = item.get("url") or item.get("link") or ""
            if not image_url or _is_blocked_image_source(image_url) or _is_blocked_image_source(page_url):
                continue
            candidates.append({
                "image": image_url,
                "page_url": page_url,
                "title": _clean_search_text(item.get("title")),
                "source": item.get("source") or _domain_of(page_url),
                "width": item.get("width"),
                "height": item.get("height"),
                "provider": "DuckDuckGo Images",
            })
        return candidates
    except Exception:
        return []


def _image_candidates_serpapi(name):
    # Optional fallback. No SerpApi request is made unless the user configures a key.
    if not SERPAPI_KEY:
        return []
    try:
        params = urllib.parse.urlencode({
            "engine": "google_images",
            "q": f'"{name}" cricketer official player portrait',
            "api_key": SERPAPI_KEY,
            "ijn": "0",
        })
        req = urllib.request.Request(
            "https://serpapi.com/search.json?" + params,
            headers={"User-Agent": "SupeRRSelectorExplorer/2.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        candidates = []
        for item in payload.get("images_results", [])[:10]:
            image_url = item.get("original") or item.get("thumbnail")
            page_url = item.get("link") or ""
            if not image_url or _is_blocked_image_source(image_url) or _is_blocked_image_source(page_url):
                continue
            candidates.append({
                "image": image_url,
                "page_url": page_url,
                "title": _clean_search_text(item.get("title")),
                "source": item.get("source") or _domain_of(page_url),
                "width": item.get("original_width"),
                "height": item.get("original_height"),
                "provider": "Google Images (SerpApi)",
            })
        return candidates
    except Exception:
        return []


def _score_image_candidate(name, candidate):
    title = candidate.get("title", "")
    page_url = candidate.get("page_url", "")
    domain = _domain_of(page_url)
    combined = f"{title} {page_url} {candidate.get('source', '')}".lower()
    tokens = [t for t in re.findall(r"[a-z0-9]+", name.lower()) if len(t) > 1]
    title_low = title.lower()
    token_coverage = (
        sum(token in title_low for token in tokens) / len(tokens)
        if tokens else 0.0
    )

    exact_name = 30 if name.lower() in title_low else (24 if token_coverage >= 0.99 else 0)
    identity_similarity = 25 * max(
        _name_similarity(name, title),
        _name_similarity(name, page_url.replace("/", " ").replace("-", " ")),
    )
    cricket_terms = ("cricket", "cricketer", "ipl", "batter", "bowler", "t20", "bcci", "icc")
    cricket_relevance = 15 if any(term in combined for term in cricket_terms) else 0
    if any(domain == d or domain.endswith("." + d) for d in IMAGE_SOURCE_DOMAINS):
        cricket_relevance = 15

    team_context = 15 if any(team in combined for team in IPL_TEAM_KEYS) else (
        7.5 if any(term in combined for term in ("india", "indian", "cricket", "ipl")) else 0
    )

    try:
        width, height = int(candidate.get("width") or 0), int(candidate.get("height") or 0)
    except (TypeError, ValueError):
        width, height = 0, 0
    if width >= 400 and height >= 400:
        quality = 10
    elif width >= 200 and height >= 200:
        quality = 7
    elif width >= 120 and height >= 120:
        quality = 4
    else:
        quality = 2

    metadata = 5 if title and (candidate.get("source") or page_url) else (
        2 if title or page_url else 0
    )
    score = round(exact_name + identity_similarity + cricket_relevance + team_context + quality + metadata, 1)
    return {
        "score": min(score, 100.0),
        "exact_name": exact_name,
        "identity": round(identity_similarity, 1),
        "cricket_relevance": cricket_relevance,
        "team_context": team_context,
        "quality": quality,
        "metadata": metadata,
    }


def _download_validate_image(candidate):
    image_url = candidate.get("image")
    if not image_url or _is_blocked_image_source(image_url):
        return None, None, "Blocked or missing image URL"
    try:
        request = urllib.request.Request(
            image_url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; SupeRRSelectorExplorer/2.0)",
                "Accept": "image/avif,image/webp,image/png,image/jpeg,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(request, timeout=7) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            raw = response.read(IMAGE_MAX_BYTES + 1)
        if len(raw) > IMAGE_MAX_BYTES:
            return None, None, "Image exceeds 5 MB limit"
        if content_type and not content_type.startswith("image/"):
            return None, None, "URL did not return an image"
        if not PIL_AVAILABLE:
            return None, None, "Pillow is required for image validation"
        with Image.open(BytesIO(raw)) as probe:
            probe.verify()
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
            if width < 120 or height < 120:
                return None, None, "Image dimensions are too small"
            if width * height > 30_000_000:
                return None, None, "Image dimensions are excessive"
            image = image.convert("RGB")
            image.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
            output = BytesIO()
            image.save(output, format="JPEG", quality=82, optimize=True)
            optimized = output.getvalue()
        digest = hashlib.sha256(optimized).hexdigest()
        data_uri = "data:image/jpeg;base64," + base64.b64encode(optimized).decode("ascii")
        return data_uri, digest, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def _fetch_player_image(name):
    started = time.perf_counter()
    out = {
        "photo": None, "photo_source_url": None, "photo_source": None,
        "photo_provider": None, "photo_score": 0, "photo_verified": False,
        "photo_hash": None, "image_duration": 0.0, "image_fallback_used": False,
        "image_error": None, "image_score_details": {},
    }
    # One targeted search first; a single alternate query is used only if needed.
    candidate_batches = [
        ("primary", _image_candidates_ddgs(name, f'"{name}" cricketer official player portrait')),
    ]
    ranked = []
    for stage, candidates in candidate_batches:
        for candidate in candidates:
            candidate["score_details"] = _score_image_candidate(name, candidate)
            candidate["score"] = candidate["score_details"]["score"]
            ranked.append(candidate)
        ranked.sort(key=lambda item: item["score"], reverse=True)
        for candidate in ranked[:5]:
            if candidate["score"] < IMAGE_MIN_SCORE:
                continue
            photo, digest, error = _download_validate_image(candidate)
            if photo:
                domain = _domain_of(candidate.get("page_url"))
                trusted = any(domain == d or domain.endswith("." + d) for d in IMAGE_SOURCE_DOMAINS)
                out.update({
                    "photo": photo,
                    "photo_source_url": candidate.get("page_url"),
                    "photo_source": domain or candidate.get("source") or "Image search",
                    "photo_provider": candidate.get("provider", "DuckDuckGo Images"),
                    "photo_score": candidate["score"],
                    "photo_verified": candidate["score"] >= IMAGE_VERIFIED_SCORE and (
                        trusted or candidate["score_details"]["exact_name"] >= 24
                    ),
                    "photo_hash": digest,
                    "image_score_details": candidate["score_details"],
                })
                out["image_duration"] = round(time.perf_counter() - started, 2)
                return out
            out["image_error"] = error

    # One alternate DuckDuckGo query, only after the primary query yields no usable image.
    fallback_candidates = _image_candidates_ddgs(
        name, f'"{name}" cricket player headshot photograph'
    )
    out["image_fallback_used"] = True
    for candidate in fallback_candidates:
        candidate["score_details"] = _score_image_candidate(name, candidate)
        candidate["score"] = candidate["score_details"]["score"]
        ranked.append(candidate)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    for candidate in ranked[:8]:
        if candidate["score"] < IMAGE_MIN_SCORE:
            continue
        photo, digest, error = _download_validate_image(candidate)
        if photo:
            domain = _domain_of(candidate.get("page_url"))
            trusted = any(domain == d or domain.endswith("." + d) for d in IMAGE_SOURCE_DOMAINS)
            out.update({
                "photo": photo,
                "photo_source_url": candidate.get("page_url"),
                "photo_source": domain or candidate.get("source") or "Image search",
                "photo_provider": candidate.get("provider", "DuckDuckGo Images"),
                "photo_score": candidate["score"],
                "photo_verified": candidate["score"] >= IMAGE_VERIFIED_SCORE and (
                    trusted or candidate["score_details"]["exact_name"] >= 24
                ),
                "photo_hash": digest,
                "image_score_details": candidate["score_details"],
                "image_error": None,
            })
            out["image_duration"] = round(time.perf_counter() - started, 2)
            return out
        out["image_error"] = error

    # Optional Google Images fallback through SerpApi; never used without a configured key.
    if SERPAPI_KEY:
        serp_ranked = _image_candidates_serpapi(name)
        for candidate in serp_ranked:
            candidate["score_details"] = _score_image_candidate(name, candidate)
            candidate["score"] = candidate["score_details"]["score"]
        serp_ranked.sort(key=lambda item: item["score"], reverse=True)
        for candidate in serp_ranked[:8]:
            if candidate["score"] < IMAGE_MIN_SCORE:
                continue
            photo, digest, error = _download_validate_image(candidate)
            if photo:
                domain = _domain_of(candidate.get("page_url"))
                trusted = any(domain == d or domain.endswith("." + d) for d in IMAGE_SOURCE_DOMAINS)
                out.update({
                    "photo": photo,
                    "photo_source_url": candidate.get("page_url"),
                    "photo_source": domain or candidate.get("source") or "Image search",
                    "photo_provider": candidate.get("provider"),
                    "photo_score": candidate["score"],
                    "photo_verified": candidate["score"] >= IMAGE_VERIFIED_SCORE and (
                        trusted or candidate["score_details"]["exact_name"] >= 24
                    ),
                    "photo_hash": digest,
                    "image_score_details": candidate["score_details"],
                    "image_error": None,
                })
                break
            out["image_error"] = error

    out["image_duration"] = round(time.perf_counter() - started, 2)
    if not out["photo"] and not out["image_error"]:
        out["image_error"] = "No suitable non-Wikipedia cricket portrait passed validation."
    return out


def _fetch_profile_raw(name):
    """Fetch profile text and photograph independently; safe to run in a worker."""
    out = {
        "ok": False, "found": False, "error": None, "sources": [],
        "photo": None, "bio": None, "role": None, "batting": None,
        "bowling": None, "ipl_teams": None, "t20_debut": None,
        "t20i_debut": None, "ipl_debut": None, "ipl_matches": None,
        "ipl_runs": None, "ipl_wickets": None, "t20i_matches": None,
        "t20i_runs": None, "t20i_wickets": None,
    }

    # Image retrieval is deliberately independent from the biography search.
    try:
        image_data = _fetch_player_image(name)
        out.update(image_data)
        if out.get("photo"):
            out["sources"].append(out.get("photo_provider") or "Image search")
    except Exception as exc:
        out["image_error"] = f"{type(exc).__name__}: {exc}"

    if not DDGS_AVAILABLE:
        out["error"] = "Missing dependency: install ddgs."
        out["found"] = bool(out.get("photo"))
        out["ok"] = bool(out["found"])
        return out

    try:
        queries = [
            f'"{name}" cricketer profile batting style bowling style teams',
            f'"{name}" IPL career statistics matches runs wickets',
            f'"{name}" T20I career statistics matches runs wickets',
        ]
        all_results = []
        for query in queries:
            all_results.extend(_ddgs_text(query, max_results=5))

        deduped, seen_urls = [], set()
        for item in all_results:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                deduped.append(item)
        all_results = deduped

        best = _best_search_result(all_results, name)
        trusted_results = [item for item in all_results if _trusted_cricket_result(item)]
        page_texts = []
        for item in trusted_results[:3]:
            extracted = _ddgs_extract(item["url"])
            if extracted:
                page_texts.append((item, extracted))
        if not page_texts:
            page_texts = [(item, item.get("snippet", "")) for item in all_results[:8]]

        combined_text = _clean_search_text(" ".join(
            f"{item.get('title', '')}. {item.get('snippet', '')}. {page_text}"
            for item, page_text in page_texts
        ))

        if best:
            out["ddg_title"] = best.get("title")
            out["ddg_url"] = best.get("url")
            out["sources"].append(best.get("source") or "DuckDuckGo search")

        bio_source = _best_search_result(trusted_results or all_results, name)
        if bio_source and bio_source.get("snippet"):
            out["bio"] = bio_source["snippet"][:480]
            out["bio_source_url"] = bio_source.get("url")

        batting_match = re.search(
            r"\b(Right[- ]hand(?:ed)?|Left[- ]hand(?:ed)?)\s+(?:batsman|batter|bat)\b",
            combined_text, re.I,
        )
        if batting_match:
            out["batting"] = batting_match.group(1).replace("-", " ").title()
        else:
            batting_match = re.search(
                r"\bBatting\s*(?:style)?\s*[:\-]\s*([^.;]{3,45})",
                combined_text, re.I,
            )
            if batting_match:
                out["batting"] = batting_match.group(1).strip()

        bowling_match = re.search(
            r"\b((?:Right|Left)[- ]arm\s+(?:fast|medium|slow|off[- ]break|leg[- ]break|orthodox|wrist[- ]spin)[^.;]{0,35})",
            combined_text, re.I,
        )
        if bowling_match:
            out["bowling"] = bowling_match.group(1).strip()
        else:
            bowling_match = re.search(
                r"\bBowling\s*(?:style)?\s*[:\-]\s*([^.;]{3,60})",
                combined_text, re.I,
            )
            if bowling_match:
                out["bowling"] = bowling_match.group(1).strip()

        role_match = re.search(
            r"\b((?:right|left)[- ]handed\s+)?(wicketkeeper[- ]batter|wicketkeeper|all[- ]rounder|batter|batsman|bowler)\b",
            combined_text, re.I,
        )
        if role_match:
            out["role"] = " ".join(x for x in role_match.groups() if x).strip().title()

        ipl_texts, t20i_texts = [], []
        for item, page_text in page_texts:
            context = f"{item.get('title', '')} {item.get('snippet', '')} {page_text}"
            low = context.lower()
            if "ipl" in low or "indian premier league" in low:
                ipl_texts.append(context)
            if "t20i" in low or "twenty20 international" in low:
                t20i_texts.append(context)
        ipl_blob, t20i_blob = " ".join(ipl_texts), " ".join(t20i_texts)
        out["ipl_matches"] = _extract_labelled_value(ipl_blob, ["matches", "mat"])
        out["ipl_runs"] = _extract_labelled_value(ipl_blob, ["runs"])
        out["ipl_wickets"] = _extract_labelled_value(ipl_blob, ["wickets", "wkts"])
        out["t20i_matches"] = _extract_labelled_value(t20i_blob, ["matches", "mat"])
        out["t20i_runs"] = _extract_labelled_value(t20i_blob, ["runs"])
        out["t20i_wickets"] = _extract_labelled_value(t20i_blob, ["wickets", "wkts"])

        out["t20_debut"] = _find_debut(combined_text, [
            r"(?:T20|Twenty20)\s+debut\s*[:\-]?\s*([^.;]{4,35})",
            r"made (?:his|her) T20 debut (?:on|against)?\s*([^.;]{4,35})",
        ])
        out["t20i_debut"] = _find_debut(combined_text, [
            r"T20I debut\s*[:\-]?\s*([^.;]{4,35})",
            r"Twenty20 International debut\s*[:\-]?\s*([^.;]{4,35})",
        ])
        out["ipl_debut"] = _find_debut(combined_text, [
            r"IPL debut\s*[:\-]?\s*([^.;]{4,35})",
            r"made (?:his|her) IPL debut (?:for [^.;]{1,35}? )?(?:in|on)\s+([^.;]{4,25})",
        ])

        team_mentions = [team.title() for team in IPL_TEAM_KEYS if team in combined_text.lower()]
        out["ipl_teams"] = ", ".join(dict.fromkeys(team_mentions[:5])) or None
        out["ok"] = True
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"

    out["found"] = bool(out.get("photo") or out.get("bio") or out.get("ddg_title"))
    if not out["found"] and not out.get("error"):
        out["error"] = out.get("image_error") or "No matching cricket profile or portrait was found."
    return out


# Reuse the application's existing profile-cache file. The version marker causes
# old Wikipedia/Wikidata cache entries to be ignored and replaced, not reused.
CACHE_FILE = os.path.join(APP_DIR, "player_profile_cache_v2.json")


@st.cache_resource
def _profile_store():
    store = {}
    try:
        with open(CACHE_FILE, encoding="utf-8") as fh:
            raw = json.load(fh)
        if raw.get("_cache_version") == PROFILE_CACHE_VERSION:
            now = time.time()
            for key, value in raw.items():
                if key != "_cache_version" and isinstance(value, list) and len(value) == 2 and value[0] > now:
                    store[key] = value
    except Exception:
        pass
    return store


def _save_store(store):
    try:
        payload = {"_cache_version": PROFILE_CACHE_VERSION}
        payload.update({
            key: value for key, value in store.items()
            if isinstance(value, list) and len(value) == 2 and value[1].get("found")
        })
        with open(CACHE_FILE, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
    except Exception:
        pass


def get_profiles(names):
    """Fetch uncached profiles with four workers and update progress on the UI thread."""
    unique_names = [name for name in dict.fromkeys(names) if name and name != "Unknown"]
    store, now = _profile_store(), time.time()
    cached_names = [name for name in unique_names if name in store and store[name][0] > now]
    todo = [name for name in unique_names if name not in cached_names]

    total = len(unique_names)
    completed = len(cached_names)
    cache_hits = sum(1 for name in cached_names if (store[name][1] or {}).get("photo"))
    image_successes = sum(1 for name in cached_names if (store[name][1] or {}).get("photo"))
    trusted_successes = sum(
        1 for name in cached_names
        if (store[name][1] or {}).get("photo")
        and any(
            (store[name][1] or {}).get("photo_source", "").lower() == domain
            or (store[name][1] or {}).get("photo_source", "").lower().endswith("." + domain)
            for domain in IMAGE_SOURCE_DOMAINS
        )
    )
    fallback_count = 0
    duplicate_count = 0
    durations = [
        float((store[name][1] or {}).get("image_duration") or 0)
        for name in cached_names
    ]
    scores = [
        float((store[name][1] or {}).get("photo_score") or 0)
        for name in cached_names if (store[name][1] or {}).get("photo")
    ]

    progress = st.progress(
        completed / total if total else 1.0,
        text=f"Preparing player profiles: {completed}/{total}",
    )
    metrics = st.empty()

    def show_metrics():
        with metrics.container():
            cols = st.columns(6)
            cols[0].metric("Image cache hits", f"{(cache_hits / max(completed, 1)) * 100:.0f}%")
            cols[1].metric("Photo success", f"{image_successes}/{completed}")
            cols[2].metric("Trusted-source rate", f"{(trusted_successes / max(image_successes, 1)) * 100:.0f}%")
            cols[3].metric("Avg. image time", f"{(sum(durations) / len(durations)):.1f}s" if durations else "—")
            cols[4].metric("Avg. image score", f"{(sum(scores) / len(scores)):.0f}/100" if scores else "—")
            cols[5].metric("Remaining", max(total - completed, 0))
        st.caption(
            f"Fallback searches: {fallback_count} · Duplicate portraits rejected: {duplicate_count} · "
            f"Processed: {completed}/{total}"
        )

    show_metrics()
    if todo:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_fetch_profile_raw, name): name for name in todo}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"found": False, "photo": None, "error": f"{type(exc).__name__}: {exc}"}

                digest = result.get("photo_hash")
                if digest:
                    duplicate_of = next((
                        existing_name for existing_name, existing_value in store.items()
                        if existing_name != name and isinstance(existing_value, list)
                        and len(existing_value) == 2
                        and (existing_value[1] or {}).get("photo_hash") == digest
                    ), None)
                    if duplicate_of:
                        result["photo"] = None
                        result["photo_hash"] = None
                        result["photo_verified"] = False
                        result["image_error"] = f"Duplicate portrait matched {duplicate_of}; image rejected."
                        duplicate_count += 1

                ttl = 7 * 86400 if result.get("found") else 6 * 3600
                store[name] = [time.time() + ttl, result]
                completed += 1
                if result.get("photo"):
                    image_successes += 1
                    scores.append(float(result.get("photo_score") or 0))
                    source_domain = (result.get("photo_source") or "").lower()
                    if any(source_domain == domain or source_domain.endswith("." + domain)
                           for domain in IMAGE_SOURCE_DOMAINS):
                        trusted_successes += 1
                if result.get("image_fallback_used"):
                    fallback_count += 1
                if result.get("image_duration") is not None:
                    durations.append(float(result.get("image_duration") or 0))
                progress.progress(
                    min(completed / max(total, 1), 1.0),
                    text=f"Retrieving player profiles and photographs: {completed}/{total}",
                )
                show_metrics()
        _save_store(store)

    progress.progress(1.0, text=f"Retrieval complete: {total}/{total} players processed")
    show_metrics()
    return {name: store[name][1] for name in names if name in store}


LOGO_TRUSTED = ("iplt20.com", "rajasthanroyals.com", "bcci.tv", "espncricinfo.com", "cricbuzz.com")
LOGO_NAMES = [ "rr_logo.jpeg"]


def _fetch_and_cache_rr_logo():
    """One-time lookup of the Rajasthan Royals logo. Saved to assets/rr_logo.png so
    later runs use the local file and make no network request."""
    if not (DDGS_AVAILABLE and PIL_AVAILABLE):
        return None
    try:
        results = DDGS(timeout=9).images(
            "Rajasthan Royals logo png", region="in-en", safesearch="moderate", max_results=15
        ) or []
    except Exception:
        return None
    ranked = []
    for item in results:
        url = item.get("image") or item.get("original") or ""
        page = item.get("url") or item.get("link") or ""
        if not url or _is_blocked_image_source(url) or _is_blocked_image_source(page):
            continue
        try:
            w, h = int(float(item.get("width") or 0)), int(float(item.get("height") or 0))
        except (TypeError, ValueError):
            w, h = 0, 0
        sc = 0
        dom = _domain_of(url) + " " + _domain_of(page)
        if any(d in dom for d in LOGO_TRUSTED):
            sc += 40
        if "logo" in (item.get("title") or "").lower():
            sc += 20
        if "rajasthan" in (item.get("title") or "").lower():
            sc += 15
        if url.lower().split("?")[0].endswith(".png"):
            sc += 15
        if w and h and 0.8 <= w / h <= 1.25:
            sc += 10
        if w and w >= 200:
            sc += 5
        ranked.append((sc, url))
    ranked.sort(reverse=True)
    for sc, url in ranked[:6]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; SupeRRSelectorExplorer/2.0)"})
            with urllib.request.urlopen(req, timeout=8) as r:
                raw = r.read(IMAGE_MAX_BYTES + 1)
            if len(raw) > IMAGE_MAX_BYTES:
                continue
            with Image.open(BytesIO(raw)) as im:
                if im.size[0] < 120 or im.size[1] < 120:
                    continue
                im = im.convert("RGBA")          # keep transparency
                im.thumbnail((400, 400))
                folder = os.path.join(APP_DIR, "assets")
                os.makedirs(folder, exist_ok=True)
                out = os.path.join(folder, "rr_logo.png")
                im.save(out, format="PNG", optimize=True)
                return out
        except Exception:
            continue
    return None


def get_rr_logo():
    """1) local assets/rr_logo.* if present  2) otherwise fetch once and cache  3) else None (RR badge)."""
    path = _find_asset(LOGO_NAMES)
    if not path:
        try:
            path = _fetch_and_cache_rr_logo()
        except Exception:
            path = None
    return _data_uri(path) if path else None


# =====================================================================
# 3. CACHED MODEL DATA
# =====================================================================
@st.cache_resource(show_spinner="Initializing Analytics Terminal...")
def load_base_data():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    try:
        import main
        from src.ranking.ranker import apply_dynamic_eb_shrinkage, safe_zscore, estimate_beta_binomial_priors

        bat_base = main.batting_deltas.copy()
        bowl_base = main.bowling_deltas.copy()
        flags = main.flags_df.copy()

        return bat_base, bowl_base, flags, apply_dynamic_eb_shrinkage, safe_zscore, estimate_beta_binomial_priors
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


batters_base, bowlers_base, flags_df, apply_eb, safe_z, est_priors = load_base_data()

# =====================================================================
# 4. HERO + METRICS + ABOUT
# =====================================================================
_logo = get_rr_logo()
_logo_html = (f"<div class='rr-logo'><img src='{_logo}' alt='Rajasthan Royals'></div>" if _logo
              else "<div class='rr-logo rr-logo-fallback'>RR</div>")

st.markdown(
    "<div class='hero'>"
    "<div class='hero-top'>"
    f"<div class='hero-left'>{_logo_html}<div class='rr-name'>Rajasthan Royals</div></div>"
    "<div class='hero-right'><div class='hero-brand'>SupeRR Selector</div>"
    "<div class='hero-sub'>SEASON 2 · ANALYTICS TERMINAL</div></div>"
    "</div>"
    "<h1>Pressure Performance Explorer</h1>"
    "<p>Explore player reliability, adjust scoring weights and understand selection methodology.</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<div class='metric-container'>"
    f"<div class='metric-card'><div class='metric-val magenta'>194,014</div><div class='metric-label'>Deliveries</div></div>"
    f"<div class='metric-card'><div class='metric-val dark'>{len(batters_base)}</div><div class='metric-label'>Eligible batters*</div></div>"
    f"<div class='metric-card'><div class='metric-val teal'>{len(bowlers_base)}</div><div class='metric-label'>Eligible bowlers*</div></div>"
    f"</div>"
    f"<div class='footnote'>*Eligible = players who have faced enough pressure moments in the data to be judged fairly.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='info-card'>"
    "<div class='config-header'>What is this?</div>"
    "<p><b>SupeRR Selector</b> is Rajasthan Royals' talent-hunt challenge. Season 2 asks participants to use "
    "real cricket data to pick the players who deserve a place in the squad for the <b>IPL 2027</b> auction.</p>"
    "<p>This explorer is our answer to one question: <b>who stays calm and delivers when the pressure is on?</b> "
    "Anyone can score runs or take wickets when the match is easy. We look at the tight moments &ndash; close chases, "
    "wickets falling, overs that decide the game &ndash; across <b>194,014 deliveries</b>, and score every batter and "
    "bowler on how well and how steadily they perform in those moments.</p>"
    "<p>The ten best batters and ten best bowlers appear at the bottom of the page. Nothing here is a black box: "
    "you can change the settings below and watch the lists update straight away.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# =====================================================================
# 5. MODEL CONFIGURATION
# =====================================================================
SHRINK_STRICT = "Strict EB shrinkage (Official Methodology)"
SHRINK_RAW = "Raw VOA (Experimental Sensitivity Analysis)"


def _config_container():
    for kwargs in ({"border": False, "key": "config_card"}, {"border": True}, {}):
        try:
            return st.container(**kwargs)
        except TypeError:
            continue


def control_head(tag, title, help_text):
    # the invisible marker lets the CSS find the slider that comes right after this block
    st.markdown(
        f"<div class='ctl-label'>{title}<i class='mk-{tag}'></i></div><div class='ctl-help'>{help_text}</div>",
        unsafe_allow_html=True,
    )


def legend(left, lp, right, rp):
    st.markdown(
        f"<div class='legend'><span style='color:{PINK}'>{left} {lp*100:.0f}%</span>"
        f"<span style='color:{TEAL}'>{right} {rp*100:.0f}%</span></div>",
        unsafe_allow_html=True,
    )


with _config_container():
    st.markdown(
        "<div class='config-header'>Model configuration</div>"
        "<div class='config-subheader'>These dials decide how each player's final score is put together. "
        "Move them to see how the top 10 batters and bowlers change.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='ctl-label'>Shrinkage mode</div>", unsafe_allow_html=True)
    eb_mode = st.radio("Shrinkage mode", options=[SHRINK_STRICT, SHRINK_RAW], horizontal=True,
                       label_visibility="collapsed", key="shrink_mode")
    bypass_eb = (eb_mode == SHRINK_RAW)
    st.markdown(
        "<div class='explain-grid'>"
        f"<div class='explain-box{'' if bypass_eb else ' active'}'><b>The careful, fair way.</b> A player who has faced "
        "only a few pressure moments can look brilliant (or terrible) just by luck. This mode gently pulls such "
        "players back towards the average, so to rank high a player has to prove it over many matches.</div>"
        f"<div class='explain-box{' active' if bypass_eb else ''}'><b>No safety net.</b> Every player's numbers are used "
        "exactly as they are, so someone with just a few great innings can jump to the top. Use it only to test how "
        "much the careful method changes the list.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    col_conf1, col_conf2 = st.columns(2, gap="large")
    with col_conf1:
        control_head("b_perf", "Batting: Performance vs consistency",
                     "<b>Performance</b> is how much a batter helps the team win in pressure moments. "
                     "<b>Consistency</b> is how steady they are from one innings to the next, without wild swings. "
                     "Slide right to reward match-winners, left to reward dependable players.")
        b_perf_val = st.slider("Batting: Performance vs consistency", 0.0, 1.0, 0.70, 0.05,
                               key="b_perf", label_visibility="collapsed")
        legend("Performance", b_perf_val, "Consistency", 1.0 - b_perf_val)

        control_head("b_voa", "Batting: VOA vs survival",
                     "<b>VOA</b> is the model's measure of the extra value a batter creates compared with an average "
                     "player in pressure moments. <b>Survival</b> is how rarely they get out when the heat is on. "
                     "Slide right to favour scoring impact, left to favour staying at the crease.")
        b_voa_val = st.slider("Batting: VOA vs survival", 0.0, 1.0, 0.60, 0.05,
                              key="b_voa", label_visibility="collapsed")
        legend("VOA", b_voa_val, "Survival", 1.0 - b_voa_val)

    with col_conf2:
        control_head("bw_perf", "Bowling: Performance vs consistency",
                     "<b>Performance</b> is how well a bowler keeps the runs in check under pressure compared with an "
                     "average bowler. <b>Consistency</b> is how steady they are from match to match. "
                     "Slide right to reward the biggest impact, left to reward the steady, reliable bowlers.")
        bw_perf_val = st.slider("Bowling: Performance vs consistency", 0.0, 1.0, 0.70, 0.05,
                                key="bw_perf", label_visibility="collapsed")
        legend("Performance", bw_perf_val, "Consistency", 1.0 - bw_perf_val)

# Slider track: pink up to the value, green for the rest (replaces the grey)
_slider_css = ""
for _tag, _val in (("b_perf", b_perf_val), ("b_voa", b_voa_val), ("bw_perf", bw_perf_val)):
    _p = int(round(_val * 100))
    _grad = f"linear-gradient(to right, {PINK} 0%, {PINK} {_p}%, {TEAL} {_p}%, {TEAL} 100%)"
    _decl = f"background: {_grad} !important; height: 8px !important; border-radius: 999px !important;"
    for _pre in (".element-container", "[data-testid='stElementContainer']"):
        _base = f"{_pre}:has(.mk-{_tag}) + {_pre}"
        _slider_css += f"{_base} [data-baseweb='slider'] > div > div:not([role='slider']):not([data-testid]) {{ {_decl} }}\n"
        _slider_css += f"{_base} [data-baseweb='slider'] div[style*='linear-gradient'] {{ {_decl} }}\n"
    _slider_css += f".st-key-{_tag} [data-baseweb='slider'] div[style*='linear-gradient'] {{ {_decl} }}\n"
st.markdown(f"<style>{_slider_css}</style>", unsafe_allow_html=True)

b_cons_val = 1.0 - b_perf_val
b_surv_val = 1.0 - b_voa_val
bw_cons_val = 1.0 - bw_perf_val


# =====================================================================
# 6. RANKING MATH ENGINE (unchanged logic)
# =====================================================================
def recalculate_rankings(bat, bwl, b_perf, b_cons, b_voa, b_surv, bw_perf, bw_cons, bypass):
    bat = bat.copy()
    bwl = bwl.copy()

    alpha, beta = est_priors(bat, 'pressure_dismissals', 'pressure_opportunities')
    if pd.isna(alpha) or pd.isna(beta):
        bat['z_survival'] = 0.0
    else:
        bat['z_survival'] = safe_z((bat['pressure_dismissals'] + alpha) / (bat['pressure_opportunities'] + alpha + beta)) * -1

    if bypass:
        bat['z_batting_voa'] = safe_z(bat['batting_voa'])
        bwl['z_bowling_performance_PROVISIONAL'] = safe_z(bwl['bowling_voa_PROVISIONAL'])
    else:
        bat['z_batting_voa'] = safe_z(apply_eb(bat, 'batting_voa', 'batting_consistency', 'valid_batting_innings'))
        bwl['z_bowling_performance_PROVISIONAL'] = safe_z(apply_eb(bwl, 'bowling_voa_PROVISIONAL', 'bowling_consistency_PROVISIONAL', 'valid_bowling_innings'))

    bat['Final_Score'] = (b_perf * ((b_voa * bat['z_batting_voa']) + (b_surv * bat['z_survival']))) + (b_cons * (safe_z(bat['batting_consistency']) * -1))
    bwl['Final_Score_PROVISIONAL'] = (bw_perf * bwl['z_bowling_performance_PROVISIONAL']) + (bw_cons * (safe_z(bwl['bowling_consistency_PROVISIONAL']) * -1))

    top_batters = bat.dropna(subset=['Final_Score']).sort_values(by='Final_Score', ascending=False).head(10).reset_index(drop=True)
    top_bowlers = bwl.dropna(subset=['Final_Score_PROVISIONAL']).sort_values(by='Final_Score_PROVISIONAL', ascending=False).head(10).reset_index(drop=True)

    if 'striker_name' in flags_df.columns:
        b_map = dict(zip(flags_df['striker_id'].dropna().astype(float), flags_df.dropna(subset=['striker_id'])['striker_name']))
        top_batters['Player_Name'] = top_batters['eligible_batter_id'].astype(float).map(b_map)
    if 'bowler_name' in flags_df.columns:
        bw_map = dict(zip(flags_df['bowler_id'].dropna().astype(float), flags_df.dropna(subset=['bowler_id'])['bowler_name']))
        top_bowlers['Player_Name'] = top_bowlers['bowler_id'].astype(float).map(bw_map)

    return top_batters, top_bowlers


top_batters, top_bowlers = recalculate_rankings(batters_base, bowlers_base, b_perf_val, b_cons_val, b_voa_val, b_surv_val, bw_perf_val, bw_cons_val, bypass_eb)


# =====================================================================
# 7. PLAYER PROFILE RENDERING
# =====================================================================
def safe_val(val):
    try:
        if val is None or pd.isna(val) or val == -1 or val == "":
            return "—"
    except Exception:
        pass
    return str(val)


def fnum(val, digits=4):
    try:
        return f"{float(val):.{digits}f}"
    except Exception:
        return "—"


def player_name(row, col="Player_Name"):
    v = row.get(col)
    return v.strip() if isinstance(v, str) and v.strip() else "Unknown"


def rows_html(rows):
    return "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(safe_val(v))}</td></tr>" for k, v in rows
    )


def render_player(idx, name, prof, kind, accent, avatar_bg, model_rows, innings_val):
    prof = prof or {}
    avatar = (f"https://ui-avatars.com/api/?name={urllib.parse.quote(name)}"
              f"&background={avatar_bg}&color=FFFFFF&size=256&bold=true")
    photo = prof.get("photo") or avatar
    role_line = prof.get("role") or kind
    bio = prof.get("bio") or ("Bio information currently unavailable." if not prof.get("found") else "")

    source_url = prof.get("bio_source_url") or prof.get("ddg_url") or prof.get("photo_source_url")
    source_label = prof.get("ddg_title") or prof.get("photo_source") or name
    if source_url:
        src = (f"Source: <a href='{html.escape(source_url, quote=True)}' target='_blank' "
               f"rel='noopener noreferrer'>{html.escape(source_label)}</a>. "
               f"Photo confidence: {float(prof.get('photo_score') or 0):.0f}/100. "
               f"{'High-confidence metadata match.' if prof.get('photo_verified') else 'Identity match needs manual review.'}")
    elif prof.get("error") or prof.get("image_error"):
        message = prof.get("error") or prof.get("image_error")
        src = f"Could not retrieve live data ({html.escape(str(message))}). It will retry automatically."
    else:
        src = "No matching cricket profile or portrait was found in the configured sources."

    def tile(label, val, color):
        return (f"<div class='stat-tile'><div class='stat-val' style='color:{color}'>"
                f"{html.escape(safe_val(val))}</div><div class='stat-lbl'>{label}</div></div>")

    parts = [
        "<div class='p-head'>",
        f"<img class='p-photo' src='{html.escape(photo, quote=True)}' alt='{html.escape(name, quote=True)}' "
        f"style='border:2px solid {accent};'>",
        "<div>",
        f"<h3 class='p-name'>{html.escape(name)}</h3>",
        f"<div class='p-role'>{html.escape(role_line)}</div>",
        f"<div class='selected-badge'>Selected Rank: #{idx + 1}</div>",
        "</div></div>",
        f"<p class='p-bio'>{html.escape(bio)}</p>" if bio else "",
        "<div class='stat-row'>",
        tile("IPL matches", prof.get("ipl_matches"), NAVY),
        tile("IPL runs scored", prof.get("ipl_runs"), PINK),
        tile("IPL wickets", prof.get("ipl_wickets"), TEAL),
        "</div>",
        "<table class='profile-table'><tr><th colspan='2'>Career milestones</th></tr>",
        rows_html([
            ("First T20 (debut)", prof.get("t20_debut")),
            ("T20I debut (international)", prof.get("t20i_debut")),
            ("IPL debut / first season", prof.get("ipl_debut")),
        ]),
        "</table>",
        "<table class='profile-table'><tr><th colspan='2'>Profile</th></tr>",
        rows_html([
            ("Primary role", kind),
            ("Batting hand", prof.get("batting")),
            ("Bowling style", prof.get("bowling")),
            ("IPL franchises", prof.get("ipl_teams")),
            ("Pressure innings", innings_val),
        ]),
        "</table>",
        "<table class='profile-table'><tr><th colspan='2'>Pressure statistics (model data)</th></tr>",
        rows_html(model_rows),
        "</table>",
        f"<div class='p-src'>{src}</div>",
    ]
    st.markdown("".join(parts), unsafe_allow_html=True)


bat_names = [player_name(r) for _, r in top_batters.iterrows()]
bowl_names = [player_name(r) for _, r in top_bowlers.iterrows()]
with st.spinner("Searching cricket sources for player portraits and profile information…"):
    profiles = get_profiles(bat_names + bowl_names)

st.markdown(
    "<div class='sec-row'><div class='sec-title'>Selected players</div>"
    "<span class='pill-green'>Top 10 + Top 10</span></div>",
    unsafe_allow_html=True,
)

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown(f"<div class='list-title' style='color:{TEAL};'>Top 10 Batters</div>", unsafe_allow_html=True)
    for idx, row in top_batters.iterrows():
        name = bat_names[idx]
        with st.expander(f"#{idx + 1}  {name}  (Score: {fnum(row.get('Final_Score'))})"):
            render_player(
                idx, name, profiles.get(name), "Batter", TEAL, "E91E8C",
                [("VOA", fnum(row.get('batting_voa'))),
                 ("Consistency Var", fnum(row.get('batting_consistency'))),
                 ("Dismissals", safe_val(row.get('pressure_dismissals')))],
                row.get('pressure_opportunities'),
            )

with col_right:
    st.markdown(f"<div class='list-title' style='color:{PINK};'>Top 10 Bowlers</div>", unsafe_allow_html=True)
    for idx, row in top_bowlers.iterrows():
        name = bowl_names[idx]
        with st.expander(f"#{idx + 1}  {name}  (Score: {fnum(row.get('Final_Score_PROVISIONAL'))})"):
            render_player(
                idx, name, profiles.get(name), "Bowler", PINK, "10B9A0",
                [("VOA (Economy)", fnum(row.get('bowling_voa_PROVISIONAL'))),
                 ("Consistency Var", fnum(row.get('bowling_consistency_PROVISIONAL')))],
                row.get('valid_bowling_innings'),
            )

# =====================================================================
# 7b. EXPORT TOP 10 LISTS
# =====================================================================
def _build_export(top_df, names, score_col, model_cols, kind):
    df = pd.DataFrame({"Rank": range(1, len(top_df) + 1), "Player": names, "Role": kind})
    df["Final score"] = top_df[score_col].round(4).values
    for label, col in model_cols.items():
        if col in top_df.columns:
            df[label] = pd.to_numeric(top_df[col], errors="coerce").round(4).values
    for label, key in (("IPL matches", "ipl_matches"), ("IPL runs", "ipl_runs"),
                       ("IPL wickets", "ipl_wickets"), ("IPL franchises", "ipl_teams")):
        df[label] = [(profiles.get(n) or {}).get(key) or "" for n in names]
    return df


export_bat = _build_export(top_batters, bat_names, "Final_Score", {
    "Batting VOA": "batting_voa", "Batting consistency (variance)": "batting_consistency",
    "Pressure dismissals": "pressure_dismissals", "Pressure opportunities": "pressure_opportunities",
    "Z batting VOA": "z_batting_voa", "Z survival": "z_survival"}, "Batter")
export_bowl = _build_export(top_bowlers, bowl_names, "Final_Score_PROVISIONAL", {
    "Bowling VOA (economy, provisional)": "bowling_voa_PROVISIONAL",
    "Bowling consistency (provisional)": "bowling_consistency_PROVISIONAL",
    "Valid bowling innings": "valid_bowling_innings",
    "Z bowling performance": "z_bowling_performance_PROVISIONAL"}, "Bowler")
export_settings = pd.DataFrame([
    ("Shrinkage mode", eb_mode),
    ("Batting: performance weight", b_perf_val), ("Batting: consistency weight", round(b_cons_val, 2)),
    ("Batting: VOA weight", b_voa_val), ("Batting: survival weight", round(b_surv_val, 2)),
    ("Bowling: performance weight", bw_perf_val), ("Bowling: consistency weight", round(bw_cons_val, 2)),
    ("Exported on", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
], columns=["Setting", "Value"])

st.markdown("<div style='height:10px'></div><div class='sec-row'><div class='sec-title'>Export the lists</div></div>",
            unsafe_allow_html=True)
st.caption("Downloads reflect the current slider settings above.")
_c1, _c2, _c3 = st.columns(3)
_c1.download_button("Download Top 10 Batters (CSV)", export_bat.to_csv(index=False).encode("utf-8-sig"),
                    "superr_top10_batters.csv", "text/csv", use_container_width=True)
_c2.download_button("Download Top 10 Bowlers (CSV)", export_bowl.to_csv(index=False).encode("utf-8-sig"),
                    "superr_top10_bowlers.csv", "text/csv", use_container_width=True)
try:
    _buf = BytesIO()
    with pd.ExcelWriter(_buf, engine="openpyxl") as _xw:
        export_bat.to_excel(_xw, sheet_name="Top 10 Batters", index=False)
        export_bowl.to_excel(_xw, sheet_name="Top 10 Bowlers", index=False)
        export_settings.to_excel(_xw, sheet_name="Settings used", index=False)
    _c3.download_button("Download both (Excel)", _buf.getvalue(), "superr_top10_selection.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True)
except ImportError:
    _c3.caption("Excel export needs `openpyxl` (add it to requirements.txt).")

# =====================================================================
# 8. DATA CHECK (what was retrieved for every selected player)
# =====================================================================
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
with st.expander("Data check: what was retrieved for each player"):
    rows = []
    for n in dict.fromkeys(bat_names + bowl_names):
        p = profiles.get(n) or {}
        rows.append({
            "Player": n,
            "Search result": p.get("ddg_title") or "—",
            "Photo": "Yes" if p.get("photo") else "No",
            "High-confidence match": "Yes" if p.get("photo_verified") else "Review",
            "Image score": f"{float(p.get('photo_score') or 0):.0f}/100",
            "Image source": p.get("photo_source") or "—",
            "T20 debut": p.get("t20_debut") or "—",
            "IPL debut": p.get("ipl_debut") or "—",
            "IPL runs": p.get("ipl_runs") or "—",
            "IPL wickets": p.get("ipl_wickets") or "—",
            "Sources": ", ".join(p.get("sources", [])) or "—",
            "Note": p.get("image_error") or p.get("error") or ("" if p.get("found") else "No cricket profile or portrait found"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if st.button("Re-fetch all player data"):
        _store = _profile_store()
        _store.clear()
        _save_store(_store)
        (getattr(st, "rerun", None) or st.experimental_rerun)()

# =====================================================================
# 9. FOOTER / DISCLAIMER
# =====================================================================
# =====================================================
# 9. FOOTER / DISCLAIMER
# =====================================================
# =====================================================
# 9. FOOTER / DISCLAIMER
# =====================================================
st.markdown(
    "<div class='disclaimer'>"
    "<div style='text-align:center; line-height:1.4; margin-bottom:1rem;'>"
    "<div style='font-size:1.1rem;'>Built by</div>"
    "<div style='font-size:2.2rem; font-weight:700; color:#E91E8C;'>"
    "Mihir Madkaikar</div>"
    "<div style='font-size:1.1rem;'>"
    "with assistance from Gemini Pro, Claude and ChatGPT Go, combining "
    "engineering and data analysis experience with cricket analytics and "
    "data-driven evaluation."
    "</div>"
    "</div>"
    "This is an independent, unofficial, educational project created for "
    "SupeRR Selector Season 2. It is not affiliated with, endorsed by, or operated "
    "by Rajasthan Royals or any other team, league or organisation. It does not "
    "contain official, confidential, internal or secret team data. Analysis is "
    "based on publicly available cricket data and the methodology described in "
    "this explorer.<br><br>"
    "All images, logos, player photos, team names and trademarks shown here belong "
    "to their respective copyright owners. I do not own or claim any rights to them. "
    "They appear for educational and illustrative purposes only. If any rights holder "
    "would like content credited differently or removed, please contact me.<br><br>"
    "This project is not monetized in any way and was built purely for learning "
    "and competition purposes."
    "</div>",
    unsafe_allow_html=True,
)