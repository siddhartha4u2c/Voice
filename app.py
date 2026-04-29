# pip install streamlit openai gTTS edge-tts deep-translator python-dotenv requests
# export EURI_API_KEY="your-key"
# Optional: export SERPAPI_KEY="your-key"  (strongly recommended for social search)

import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
import asyncio
import re
import json
import requests
from urllib.parse import quote_plus, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from gtts import gTTS
import edge_tts
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════
load_dotenv()
EURI_API_KEY    = os.getenv("EURI_API_KEY") or os.getenv("OPENAI_API_KEY")
SERPAPI_KEY     = os.getenv("SERPAPI_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
STT_MODEL       = os.getenv("STT_MODEL",       "gpt-4o-mini-transcribe")
CHAT_MODEL      = os.getenv("CHAT_MODEL",      "gpt-4.1-mini")
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "512"))
TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", "gpt-4o-mini")

SOCIAL_PLATFORMS = ["linkedin", "instagram", "facebook"]

# Sources that may show multi-result disambiguation UI
DISAMBIGUATION_SOURCES = SOCIAL_PLATFORMS + ["youtube"]

# All sources (legacy aggregate); runtime picks a subset per question type
ALL_SEARCH_SOURCES = ["wikipedia", "web", "youtube", "linkedin", "instagram", "facebook"]

# Identity / profile / personal-info questions → social only (user request)
PERSON_PROFILE_SOURCES = ["linkedin", "instagram", "facebook"]

# Public-figure identity/profile questions → broader factual sources + social
PUBLIC_FIGURE_PROFILE_SOURCES = ["wikipedia", "web", "youtube", "linkedin", "instagram", "facebook"]

# General knowledge → web + video + encyclopedia + light social + news links (no LinkedIn-only sweep)
GENERAL_SOURCES = ["wikipedia", "web", "youtube", "facebook", "instagram"]

# ════════════════════════════════════════════════════════════
#  MULTI-RESULT SEARCH LAYER
# ════════════════════════════════════════════════════════════

def _serpapi_search(query: str, num: int = 5) -> list:
    """Returns up to `num` organic results from SerpAPI (Google backend).
    Each result: {title, snippet, url}"""
    if not SERPAPI_KEY:
        return []
    try:
        r = requests.get(
            "https://serpapi.com/search",
            params={"q": query, "api_key": SERPAPI_KEY, "num": num, "output": "json"},
            timeout=10,
        )
        r.raise_for_status()
        return [
            {"title": x.get("title", ""), "snippet": x.get("snippet", ""), "url": x.get("link", "")}
            for x in r.json().get("organic_results", [])[:num]
        ]
    except Exception:
        return []


def _ddg_search(query: str) -> list:
    """Returns 0–5 results from the free DuckDuckGo Instant Answer API."""
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1, "skip_disambig": 1},
            timeout=8,
            headers={"User-Agent": "VoiceAIBot/1.0"},
        )
        r.raise_for_status()
        data = r.json()
        results = []
        if data.get("AbstractText"):
            results.append({
                "title":   data.get("Heading", query),
                "snippet": data["AbstractText"],
                "url":     data.get("AbstractURL", ""),
            })
        for t in data.get("RelatedTopics", []):
            if isinstance(t, dict) and t.get("Text") and len(results) < 5:
                results.append({
                    "title":   t.get("Text", "")[:60],
                    "snippet": t["Text"],
                    "url":     t.get("FirstURL", ""),
                })
        return results
    except Exception:
        return []


ENTITY_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "for", "to", "in", "on", "at", "by", "with", "from", "as", "or", "and",
    "but", "if", "so", "it", "its", "this", "that", "these", "those", "my", "your",
    "about", "who", "what", "which", "when", "where", "how", "why",
    "hai", "hain", "ho", "tha", "thi", "the", "kaun", "kya", "please", "plz",
})


def extract_entity_for_search(user_text: str) -> str:
    """Strip question filler so social search uses a proper name / topic phrase."""
    raw = (user_text or "").strip()
    if not raw:
        return raw
    t = re.sub(r"[\?\!\.…]+", " ", raw)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(
        r"\s+(kaun\s+(hai|hain|ho|tha|thi|the|thi|hoga|hogi|honge|hongi)|"
        r"kya\s+hai|kaun|kya|hai|hain|please|plz)\s*$",
        "",
        t,
        flags=re.IGNORECASE,
    ).strip()
    t = re.sub(r"[\s]*(कौन\s+हैं?|क्या\s+है|हैं?|कौन)\s*$", "", t).strip()
    t = re.sub(
        r"^(who\s+is|who\'?s|what\s+is|what\'?s|what\s+are|"
        r"tell\s+me\s+about|give\s+me\s+information\s+on|"
        r"can\s+you\s+tell\s+me\s+about|i\s+want\s+to\s+know\s+about|"
        r"do\s+you\s+know|about|define|describe|explain)\s+",
        "",
        t,
        flags=re.IGNORECASE,
    ).strip()
    t = re.sub(r"^(the|a|an)\s+", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(
        r"^(kaun\s+(hai|hain)|kya\s+hai|batao|bataiye)\s+",
        "",
        t,
        flags=re.IGNORECASE,
    ).strip()
    t = re.sub(r"^(कौन\s+हैं?|क्या\s+है|बताओ)\s*", "", t).strip()
    if len(t) < 2:
        return re.sub(r"\s+", " ", raw)
    return t


def _entity_significant_tokens(entity: str) -> list[str]:
    """Words used to verify a social hit is about the requested person/topic."""
    if not entity:
        return []
    parts = re.findall(r"[A-Za-z][A-Za-z]+|[\u0900-\u097F]+", entity)
    out: list[str] = []
    for p in parts:
        if len(p) < 2:
            continue
        pl = p.lower() if p.isascii() else p
        if pl in ENTITY_STOP_WORDS:
            continue
        out.append(pl)
    return out


def _social_hit_entity_score(blob_lower: str, tokens: list[str]) -> int:
    return sum(1 for tok in tokens if tok in blob_lower)


def _social_hit_matches_entity(blob_lower: str, tokens: list[str]) -> bool:
    if not tokens:
        return True
    n = len(tokens)
    sc = _social_hit_entity_score(blob_lower, tokens)
    if n == 1:
        return sc >= 1
    if n == 2:
        return sc >= 2
    return sc >= max(2, n - 1)


def filter_social_candidates_by_entity(hits: list, entity: str) -> list:
    """Drop irrelevant profiles (e.g. another person when the query names someone)."""
    tokens = _entity_significant_tokens(entity)
    if not tokens or not hits:
        return hits
    kept: list[tuple[int, dict]] = []
    for h in hits:
        blob = f"{h.get('title', '')} {h.get('snippet', '')}".lower()
        if _social_hit_matches_entity(blob, tokens):
            kept.append((_social_hit_entity_score(blob, tokens), h))
    if kept:
        kept.sort(key=lambda x: -x[0])
        return [h for _, h in kept]
    # Strict filter removed everyone — keep only hits that mention at least one token
    relaxed: list[tuple[int, dict]] = []
    for h in hits:
        blob = f"{h.get('title', '')} {h.get('snippet', '')}".lower()
        sc = _social_hit_entity_score(blob, tokens)
        if sc >= 1:
            relaxed.append((sc, h))
    if relaxed:
        relaxed.sort(key=lambda x: -x[0])
        return [h for _, h in relaxed]
    # Devanagari entity vs English-only titles — do not drop all SERP results
    if any("\u0900" <= c <= "\u097f" for c in entity):
        return hits
    return []


def search_multi_candidates(query: str, platform: str, max_results: int = 5) -> list:
    """
    Runs 4 query variations for `platform` in parallel and returns
    a de-duplicated ranked list of candidate results.

    Uses a cleaned **entity** (name/topic) for search, not raw questions like
    "who is X", so results match the intended person. Results are filtered so
    title+snippet must contain the significant name tokens.
    """
    site_map = {
        "linkedin":  "site:linkedin.com/in OR site:linkedin.com/pub",
        "instagram": "site:instagram.com",
        "facebook":  "site:facebook.com",
    }
    site_filter = site_map.get(platform, "")

    entity = extract_entity_for_search(query)
    q = entity if entity else query

    variations = list(dict.fromkeys([
        f"{site_filter} {q}".strip(),
        f"{site_filter} \"{q}\"".strip(),
        f"{site_filter} {q} profile".strip(),
        f"{site_filter} {q} official".strip(),
    ]))

    all_results = []
    seen_urls: set = set()

    def _fetch(variation: str) -> list:
        hits = _serpapi_search(variation, num=max_results) if SERPAPI_KEY else _ddg_search(variation)
        for h in hits:
            h["query_variation"] = variation
            h["platform"]        = platform.capitalize()
        return hits

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_fetch, v): v for v in variations}
        for fut in as_completed(futures):
            for hit in fut.result():
                url = hit.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(hit)

    q_lower = q.lower()
    all_results.sort(key=lambda x: (0 if q_lower in x.get("title", "").lower() else 1))
    all_results = filter_social_candidates_by_entity(all_results, q)

    for i, r in enumerate(all_results[:max_results], 1):
        r["rank"] = i
    return all_results[:max_results]


def search_wikipedia(query: str, sentences: int = 5) -> list:
    """Returns up to 3 Wikipedia candidate pages."""
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "format": "json", "srlimit": 3},
            timeout=8,
        )
        r.raise_for_status()
        raw = r.json().get("query", {}).get("search", [])
        results = []
        for i, item in enumerate(raw, 1):
            title   = item["title"]
            url     = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
            snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            results.append({"rank": i, "title": title, "snippet": snippet,
                             "url": url, "platform": "Wikipedia"})
        return results
    except Exception:
        return []


def search_web(query: str, num: int = 6) -> list:
    hits = _serpapi_search(query, num=num) if SERPAPI_KEY else _ddg_search(query)
    for i, h in enumerate(hits, 1):
        h["rank"]     = i
        h["platform"] = "Web"
    return hits


def search_youtube(query: str, max_results: int = 5, order_by_date: bool = False) -> list:
    """YouTube Data API v3 search for videos; optionally prioritize newest uploads."""
    if not YOUTUBE_API_KEY:
        return []
    try:
        params = {
            "part": "snippet",
            "q": query,
            "maxResults": max_results,
            "key": YOUTUBE_API_KEY,
            "type": "video",
        }
        if order_by_date:
            params["order"] = "date"
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params=params,
            timeout=12,
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("error"):
            return []
        results = []
        for item in payload.get("items", []):
            snippet = item.get("snippet") or {}
            title = snippet.get("title") or ""
            desc = (snippet.get("description") or "")[:400]
            pid = item.get("id") or {}
            url = ""
            if pid.get("videoId"):
                url = f"https://www.youtube.com/watch?v={pid['videoId']}"
            if not url:
                continue
            results.append({
                "title":   title,
                "snippet": desc,
                "url":     url,
                "platform": "YouTube",
                "publishedAt": snippet.get("publishedAt", ""),
                "channelTitle": snippet.get("channelTitle", ""),
            })
        if order_by_date:
            results.sort(key=lambda x: x.get("publishedAt", ""), reverse=True)
        for i, row in enumerate(results[:max_results], 1):
            row["rank"] = i
        return results[:max_results]
    except Exception:
        return []


def is_latest_video_query(text: str) -> bool:
    t = (text or "").lower()
    hints = [
        "latest video", "newest video", "recent video", "most recent video",
        "last video", "latest upload", "new upload",
    ]
    return any(h in t for h in hints)


def is_recency_sensitive_query(text: str) -> bool:
    """Queries where stale video evidence is risky; prefer newest videos."""
    t = (text or "").lower()
    keyword_hits = [
        "latest", "newest", "recent", "currently", "current", "today", "now", "as of",
        "who is the", "who's the",
    ]
    office_hits = [
        "prime minister", "chief minister", "deputy chief minister", "president",
        "governor", "finance minister", "home minister", "minister",
    ]
    return any(k in t for k in keyword_hits) or any(o in t for o in office_hits)


def is_news_like_query(text: str) -> bool:
    """Detect incident/breaking/development style queries and force news retrieval mode."""
    t = (text or "").lower()
    hints = [
        "latest", "breaking", "incident", "incidents", "development", "developments",
        "update", "updates", "situation", "unrest", "violence", "clashes",
        "what happened", "current status", "news",
    ]
    return any(h in t for h in hints)


def is_incident_tracker_query(text: str) -> bool:
    """Incident-style news asks that benefit from numbered 'latest confirmed incidents' format."""
    t = (text or "").lower()
    topic_hits = [
        "incident", "incidents", "clash", "clashes", "violence", "attack", "blast",
        "killed", "injured", "protest", "unrest", "curfew", "internet shutdown",
        "ethnic conflict", "latest incidents", "what happened in", "current situation",
    ]
    return any(h in t for h in topic_hits)


def is_office_holder_query(text: str) -> bool:
    t = (text or "").lower()
    hits = [
        "who is the", "prime minister", "chief minister", "deputy chief minister",
        "president", "governor", "home minister", "finance minister", "minister",
    ]
    return any(h in t for h in hits)


def is_bengal_topic_query(text: str) -> bool:
    t = (text or "").lower()
    hints = [
        "bengal", "west bengal", "kolkata", "wb election", "bengal election",
        "bangla", "বাংলা", "বঙ্গ", "पश्चिम बंगाल", "বিধানসভা",
    ]
    return any(h in t for h in hints)


def search_youtube_with_local_variants(
    base_query: str,
    user_text: str,
    max_results: int = 6,
    order_by_date: bool = False,
) -> list:
    """For Bengal topics, blend Bengali/Hindi/English query variants and dedupe URLs."""
    if not YOUTUBE_API_KEY:
        return []
    q = (base_query or "").strip()
    if not q:
        return []
    if not is_bengal_topic_query(user_text):
        return search_youtube(q, max_results=max_results, order_by_date=order_by_date)

    variants = list(dict.fromkeys([
        q,
        f"{q} bengali",
        f"{q} bangla",
        f"{q} বাংলা",
        f"{q} hindi",
        f"{q} english",
        f"West Bengal election latest {q}",
    ]))
    merged: list[dict] = []
    seen: set[str] = set()
    for v in variants:
        rows = search_youtube(v, max_results=max_results, order_by_date=order_by_date)
        for r in rows:
            u = (r.get("url") or "").strip()
            if not u or u in seen:
                continue
            seen.add(u)
            merged.append(r)
    if order_by_date:
        merged.sort(key=lambda x: x.get("publishedAt", ""), reverse=True)
    for i, row in enumerate(merged[:max_results], 1):
        row["rank"] = i
    return merged[:max_results]


def extract_channel_name_hint(text: str) -> str:
    """Best-effort channel-name extraction from queries like '... on channel XYZ'."""
    raw = (text or "").strip()
    if not raw:
        return ""
    m = re.search(r"(?:on\s+(?:his|her|their)?\s*channel\s+)(.+)$", raw, flags=re.I)
    if m:
        return m.group(1).strip(" .?!\"'")
    m2 = re.search(r"(?:channel\s+)(.+)$", raw, flags=re.I)
    if m2:
        return m2.group(1).strip(" .?!\"'")
    return ""


def search_latest_youtube_from_channel(query_text: str, max_results: int = 5) -> list:
    """For 'latest video from channel X' queries: resolve channel, then fetch videos ordered by date."""
    if not YOUTUBE_API_KEY:
        return []
    channel_hint = extract_channel_name_hint(query_text)
    if not channel_hint:
        return []
    try:
        ch = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": channel_hint,
                "type": "channel",
                "maxResults": 3,
                "key": YOUTUBE_API_KEY,
            },
            timeout=12,
        )
        ch.raise_for_status()
        items = ch.json().get("items") or []
        if not items:
            return []
        channel_id = items[0].get("id", {}).get("channelId")
        channel_title = items[0].get("snippet", {}).get("title", "")
        if not channel_id:
            return []

        vids = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "channelId": channel_id,
                "type": "video",
                "order": "date",
                "maxResults": max_results,
                "key": YOUTUBE_API_KEY,
            },
            timeout=12,
        )
        vids.raise_for_status()
        out = []
        for i, item in enumerate((vids.json().get("items") or [])[:max_results], 1):
            sn = item.get("snippet") or {}
            vid = item.get("id", {}).get("videoId")
            if not vid:
                continue
            out.append({
                "rank": i,
                "title": sn.get("title", ""),
                "snippet": (sn.get("description") or "")[:400],
                "url": f"https://www.youtube.com/watch?v={vid}",
                "platform": "YouTube",
                "publishedAt": sn.get("publishedAt", ""),
                "channelTitle": sn.get("channelTitle") or channel_title,
            })
        return out
    except Exception:
        return []


def linkify_http(text: str) -> str:
    """Turn bare http(s) URLs in plain text into markdown links for st.markdown."""
    if not text:
        return text

    def _repl(m: re.Match) -> str:
        u = m.group(0)
        u = u.rstrip(").,;]")
        return f"[{u}]({u})"

    return re.sub(r"https?://[^\s<>\)\"']+", _repl, text)


def md_link_title_url(title: str, url: str) -> str:
    """Markdown link; escape brackets in title so links render correctly."""
    if not url:
        return f"**{title}**"
    t = (title or "Link").replace("[", "\\[").replace("]", "\\]")
    return f"[{t}]({url})"


def is_supported_video_url(url: str) -> bool:
    """URLs Streamlit can embed as video players reliably."""
    if not url:
        return False
    return bool(re.search(r"(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)", url, flags=re.I))


def is_social_profile_url(url: str) -> bool:
    if not url:
        return False
    return bool(re.search(r"(facebook\.com|instagram\.com|linkedin\.com)", url, flags=re.I))


def render_social_link_actions(url: str, key_suffix: str) -> None:
    """Show direct-open and optional in-app preview controls for social links."""
    if not is_social_profile_url(url):
        return

    st.link_button("🔗 Open on site (login if needed)", url)

    if st.toggle("Try in-app preview (scrollable)", key=f"social_preview_{key_suffix}"):
        # Some social sites block iframe embedding (X-Frame-Options/CSP).
        # When blocked, the direct-open button remains the reliable fallback.
        components.html(
            f"""
            <div style="height:420px; overflow:auto; border:1px solid #ddd; border-radius:8px;">
                <iframe
                    src="{url}"
                    style="width:100%; height:800px; border:none;"
                    loading="lazy"
                    referrerpolicy="no-referrer-when-downgrade"
                ></iframe>
            </div>
            """,
            height=440,
            scrolling=True,
        )
        st.caption("If preview does not load, click ‘Open on site’ and sign in there.")


# ════════════════════════════════════════════════════════════
#  AI CROSS-REFERENCE — auto-pick best candidate
# ════════════════════════════════════════════════════════════

def ai_pick_best_candidate(client: OpenAI, user_query: str,
                            candidates: list, platform: str) -> dict:
    """Asks the LLM to score candidates and returns the best match dict."""
    if not candidates:
        return {}
    numbered = "\n".join(
        f"{c['rank']}. Title: {c['title']}\n   Snippet: {c['snippet'][:200]}\n   URL: {c['url']}"
        for c in candidates
    )
    prompt = (
        f"The user asked: \"{user_query}\"\n\n"
        f"Below are {len(candidates)} search results from {platform}.\n\n"
        f"{numbered}\n\n"
        "Which result number is MOST LIKELY the correct person/page the user is asking about? "
        "Reply with ONLY a JSON object: "
        "{\"best\": <number>, \"confidence\": \"high|medium|low\", \"reason\": \"<one sentence>\"}"
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
        )
        raw  = re.sub(r"```[a-z]*", "", resp.choices[0].message.content.strip()).strip("`").strip()
        data = json.loads(raw)
        idx  = int(data.get("best", 1)) - 1
        if 0 <= idx < len(candidates):
            candidates[idx]["ai_confidence"] = data.get("confidence", "low")
            candidates[idx]["ai_reason"]     = data.get("reason", "")
            return candidates[idx]
    except Exception:
        pass
    return candidates[0] if candidates else {}


# ════════════════════════════════════════════════════════════
#  BUILD LLM CONTEXT FROM CHOSEN CANDIDATES
# ════════════════════════════════════════════════════════════

def _wiki_title_from_candidate(cand: dict) -> str:
    t = (cand.get("title") or "").strip()
    if t:
        return t
    url = cand.get("url") or ""
    m = re.search(r"wikipedia\.org/wiki/([^?#]+)", url)
    if m:
        return unquote(m.group(1)).replace("_", " ")
    return ""


def enrich_wikipedia_pick(cand: dict) -> dict:
    """Wikipedia REST summary: neutral lead + thumbnail (live MediaWiki; not chat training data)."""
    title = _wiki_title_from_candidate(cand)
    if not title:
        return cand
    try:
        safe = quote_plus(title.replace(" ", "_"))
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe}",
            timeout=8,
            headers={"User-Agent": "VoiceAIBot/1.0 (https://github.com/)"},
        )
        if r.status_code != 200:
            return cand
        d = r.json()
        out = dict(cand)
        if d.get("extract"):
            out["rest_extract"] = d["extract"]
        thumb = (d.get("thumbnail") or {}).get("source")
        if thumb:
            out["thumbnail"] = thumb
        if d.get("description"):
            out["short_description"] = d["description"]
        return out
    except Exception:
        return cand


def extract_youtube_video_id(url: str) -> str | None:
    if not url:
        return None
    m = re.search(
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        url,
    )
    return m.group(1) if m else None


def enrich_youtube_pick(cand: dict) -> dict:
    """Official view counts & publish time via YouTube Data API (not web scraping)."""
    if not YOUTUBE_API_KEY:
        return cand
    vid = extract_youtube_video_id(cand.get("url", ""))
    if not vid:
        return cand
    try:
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={"part": "statistics,snippet", "id": vid, "key": YOUTUBE_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        items = r.json().get("items") or []
        if not items:
            return cand
        it = items[0]
        stats = it.get("statistics") or {}
        sn = it.get("snippet") or {}
        out = dict(cand)
        out["yt_view_count"] = stats.get("viewCount")
        out["yt_like_count"] = stats.get("likeCount")
        out["yt_published_at"] = sn.get("publishedAt")
        out["yt_channel_title"] = sn.get("channelTitle")
        return out
    except Exception:
        return cand


def search_google_news(query: str, num: int = 6) -> list[dict]:
    """SerpAPI Google News engine — dedicated news index, separate from generic web search."""
    if not SERPAPI_KEY:
        return []
    try:
        r = requests.get(
            "https://serpapi.com/search",
            params={
                "engine": "google_news",
                "q": query,
                "api_key": SERPAPI_KEY,
                "num": num,
                "output": "json",
            },
            timeout=12,
        )
        r.raise_for_status()
        rows = r.json().get("news_results") or []
        out = []
        for x in rows[:num]:
            out.append({
                "title":   x.get("title", ""),
                "snippet": x.get("snippet", "") or x.get("summary", ""),
                "url":     x.get("link", "") or x.get("url", ""),
                "source":  x.get("source", ""),
                "date":    x.get("date", ""),
            })
        return out
    except Exception:
        return []


def _overlap_token_set(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z]+", (text or "").lower())
    stop = ENTITY_STOP_WORDS | {"http", "https", "www", "com", "org", "inc", "ltd"}
    return {w for w in words if len(w) > 2 and w not in stop}


def cross_source_jaccard(selections: dict) -> tuple[float, int, int]:
    wiki = selections.get("wikipedia") or {}
    web = selections.get("web") or {}
    a = _overlap_token_set(wiki.get("rest_extract") or wiki.get("snippet", ""))
    b = _overlap_token_set(web.get("snippet", ""))
    if not a or not b:
        return 0.0, len(a), len(b)
    inter = len(a & b)
    uni = len(a | b)
    return (inter / uni if uni else 0.0), len(a), len(b)


def years_mentioned_in_selections(selections: dict) -> list[int]:
    parts = []
    for c in selections.values():
        parts.append(c.get("rest_extract") or "")
        parts.append(c.get("snippet") or "")
    text = " ".join(parts)
    ys = sorted({int(y) for y in re.findall(r"\b(19\d{2}|20[0-3]\d)\b", text)})
    return ys[-12:]


def format_google_news_for_llm(items: list[dict]) -> str:
    if not items:
        return ""
    lines = ["=== GOOGLE NEWS CLUSTER (SerpAPI live index — separate from generic web hit list) ==="]
    for i, n in enumerate(items, 1):
        lines.append(
            f"{i}. {n.get('title', '')}\n"
            f"   {str(n.get('snippet', ''))[:280]}\n"
            f"   URL: {n.get('url', '')}\n"
            f"   Outlet / date: {n.get('source', '')} {n.get('date', '')}"
        )
    return "\n".join(lines)


def format_general_links_for_llm(items: list[dict]) -> str:
    if not items:
        return ""
    lines = ["=== RELATED SOURCE LINKS (curated, up to 6) ==="]
    for i, it in enumerate(items, 1):
        lines.append(
            f"{i}. [{it.get('kind', 'link')}] {it.get('title', 'Source')}\n   {it.get('url', '')}"
        )
    return "\n".join(lines)


def build_general_reference_links(
    candidates_map: dict,
    ai_picks: dict,
    news_items: list[dict] | None,
    max_links: int = 6,
) -> list[dict]:
    """Pick diverse URLs (YouTube, websites, news, Wikipedia, social) for general questions."""
    seen: set[str] = set()
    out: list[dict] = []

    def push(title: str, url: str, kind: str) -> None:
        url = (url or "").strip()
        if not url or url in seen:
            return
        seen.add(url)
        out.append({"title": title or "Source", "url": url, "kind": kind})
        if len(out) >= max_links:
            return

    wp = ai_picks.get("wikipedia")
    if wp and wp.get("url"):
        push(wp.get("title", "Wikipedia"), wp["url"], "Wikipedia")
    for c in (candidates_map.get("youtube") or [])[:2]:
        push(c.get("title", "YouTube"), c.get("url", ""), "YouTube")
    for c in (candidates_map.get("web") or [])[:4]:
        push(c.get("title", "Website"), c.get("url", ""), "Website")
    for n in (news_items or [])[:3]:
        push(n.get("title", "News"), n.get("url", ""), "News")
    for plat, label in (("facebook", "Facebook"), ("instagram", "Instagram")):
        p = ai_picks.get(plat)
        if p and p.get("url"):
            push(p.get("title", label), p["url"], label)
    return out[:max_links]


def is_person_identity_question(client: OpenAI, user_text: str) -> bool:
    """True when the user mainly wants a person's identity, profile, or personal/professional info."""
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Reply with exactly YES or NO (English only).\n"
                    "YES if the user is primarily asking about a specific person's identity, biography, "
                    "job/role, professional profile, social-media presence, contact/reachability, or "
                    "other personal information about a named or clearly implied individual.\n"
                    "NO for news/events, definitions, countries/history without a person focus, science, "
                    "products, how-to, abstract topics, or questions not centered on an individual's profile."
                )},
                {"role": "user", "content": user_text[:900]},
            ],
            max_tokens=6,
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").strip().upper()
        return out.startswith("Y")
    except Exception:
        return False


def is_public_figure_query(client: OpenAI, query_text: str) -> bool:
    """True when the named person is a notable/public figure (not a private individual)."""
    q = (query_text or "").strip()
    if not q:
        return False

    # Fast evidence check via Wikipedia search presence for the queried name/topic.
    wiki_hits = search_wikipedia(q)
    if wiki_hits:
        top_title = (wiki_hits[0].get("title") or "").lower()
        toks = [t.lower() for t in re.findall(r"[A-Za-z]{3,}", q)]
        if toks and sum(1 for t in toks if t in top_title) >= min(len(toks), 2):
            return True

    # LLM fallback for cases where wiki matching is not obvious.
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Reply with exactly YES or NO.\n"
                    "YES if this appears to refer to a public figure (politician, celebrity, "
                    "athlete, public office holder, widely known author/executive/religious leader, etc.).\n"
                    "NO if it likely refers to a private individual or unknown person."
                )},
                {"role": "user", "content": q[:600]},
            ],
            max_tokens=6,
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").strip().upper()
        return out.startswith("Y")
    except Exception:
        return False


def build_augmented_search_context(
    selections: dict,
    google_news_items: list[dict] | None,
) -> str:
    base = build_context_from_selections(selections)
    blocks = []

    wp = selections.get("wikipedia") or {}
    if wp.get("rest_extract"):
        blocks.append(
            "=== WIKIPEDIA REST SUMMARY (live MediaWiki API — full lead paragraph) ===\n"
            + wp["rest_extract"][:1600]
        )

    if google_news_items:
        blocks.append(format_google_news_for_llm(google_news_items))

    yt = selections.get("youtube") or {}
    if yt.get("yt_view_count") is not None:
        blocks.append(
            "=== YOUTUBE DATA API — HARD STATS (views / publish time; not inferred) ===\n"
            f"Views: {yt.get('yt_view_count')} | Likes (if available): {yt.get('yt_like_count', 'n/a')} | "
            f"Published: {yt.get('yt_published_at', '')} | Channel: {yt.get('yt_channel_title', '')}"
        )

    jacc, na, nb = cross_source_jaccard(selections)
    years = years_mentioned_in_selections(selections)
    if selections.get("wikipedia") and selections.get("web"):
        blocks.append(
            "=== CROSS-SOURCE PROGRAMMATIC CHECK (transparent overlap; chat bots rarely show this) ===\n"
            f"Wikipedia lead/web snippet token overlap (Jaccard index): {jacc:.3f} "
            f"(rough token-set sizes {na} vs {nb}). "
            f"Years detected in retrieved text: {years if years else 'none'}."
        )

    if not blocks:
        return base
    return base + "\n\n" + "\n\n".join(blocks)


def generate_additional_news_response(
    client: OpenAI,
    user_text: str,
    search_context: str,
    target_lang_instr: str,
    previous_responses: list[str],
) -> str:
    """Generate a new news/event follow-up without repeating prior responses."""
    history = "\n\n".join(
        f"[Previous {i+1}]\n{txt[:1400]}"
        for i, txt in enumerate(previous_responses[-6:])
        if txt
    ) or "None"
    prompt = (
        search_context
        + f"\n\nUser question: {user_text}\n\n"
        + "Generate ONE additional follow-up response about this news/event.\n"
        + f"Reply only in {target_lang_instr}. "
        + "Do not transliterate Hindi into Latin letters.\n"
        + "Important: avoid repeating wording/facts already used in previous responses.\n"
        + "Prefer a fresh angle: timeline detail, stakeholder impact, what changed next, "
          "or uncertainty/what to watch.\n"
        + "Keep it concise (4-8 lines) in markdown.\n\n"
        + f"Previously shown responses:\n{history}\n"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=min(420, CHAT_MAX_TOKENS),
    )
    return (resp.choices[0].message.content or "").strip()


def build_context_from_selections(selections: dict) -> str:
    if not selections:
        return ""
    parts = [
        f"[{platform.capitalize()} – {cand.get('title', 'Unknown')}]\n"
        f"{cand.get('snippet', '')}\n"
        f"URL: {cand.get('url', '')}"
        for platform, cand in selections.items()
    ]
    return (
        "=== VERIFIED SEARCH CONTEXT ===\n"
        + "\n\n".join(parts)
        + "\n=== END OF CONTEXT ===\n\n"
        "Use the above verified context to answer accurately. "
        "Cite the source platform in your answer. "
        "If the context is insufficient, say so clearly."
    )


# ════════════════════════════════════════════════════════════
#  LANGUAGE UTILITIES
# ════════════════════════════════════════════════════════════

def ensure_target_language_text(text: str, language_code: str) -> bool:
    script_checks = {"hi-IN": r"[\u0900-\u097F]"}
    pattern = script_checks.get(language_code)
    if not pattern:
        return True
    return bool(re.search(pattern, text))


def contains_latin_letters(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def translate_to_target_language(client, text, language_code, target_lang_instr, model, max_tokens) -> str:
    script_hint = ""
    if language_code == "hi-IN":
        script_hint = "Use only Devanagari script. Do not use any English letters."
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                f"Translate the text into {target_lang_instr}. "
                "Return only translated text. Do not include English. " + script_hint)},
            {"role": "user", "content": text},
        ],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def answer_directly_in_target_language(client, user_text, target_lang_instr, model, max_tokens) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                f"Answer the question only in {target_lang_instr}. "
                "Do not use English. Keep response concise and natural.")},
            {"role": "user", "content": user_text},
        ],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def enforce_native_language_reply(client, user_text, draft_reply, language_code,
                                   target_lang_instr, max_tokens) -> str:
    if language_code == "en-US":
        return draft_reply
    model_candidates = list(dict.fromkeys([TRANSLATE_MODEL, CHAT_MODEL, "gpt-4o-mini"]))
    reply = draft_reply
    for model_name in model_candidates:
        try:
            t = translate_to_target_language(client, reply, language_code, target_lang_instr, model_name, max_tokens)
            if t: reply = t
        except Exception:
            pass
        if language_code == "hi-IN" and not ensure_target_language_text(reply, language_code):
            try:
                r = answer_directly_in_target_language(client, user_text, target_lang_instr, model_name, max_tokens)
                if r: reply = r
            except Exception:
                pass
        if language_code == "hi-IN":
            for _ in range(2):
                if ensure_target_language_text(reply, language_code) and not contains_latin_letters(reply):
                    break
                try:
                    s = translate_to_target_language(client, reply, language_code, target_lang_instr, model_name, max_tokens)
                    if s: reply = s
                except Exception:
                    break
        if language_code != "hi-IN":
            return reply
        if ensure_target_language_text(reply, language_code) and not contains_latin_letters(reply):
            return reply
    return reply


def fallback_google_translate(text: str, language_code: str) -> str:
    target_map = {"hi-IN": "hi"}
    target = target_map.get(language_code)
    if not target:
        return text
    try:
        return GoogleTranslator(source="auto", target=target).translate(text)
    except Exception:
        return text


def prepare_markdown_for_tts(markdown_text: str) -> str:
    """Strip markdown noise so TTS reads naturally (works for both formats)."""
    if not markdown_text:
        return ""
    t = markdown_text
    t = re.sub(r"^#+\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", t)
    t = re.sub(r"^\s*[-*]\s+", " — ", t, flags=re.MULTILINE)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def build_mixed_links_block(
    selections: dict,
    ref_links: list[dict],
    language_code: str,
    max_links: int = 6,
) -> str:
    """Return a compact link section with YouTube + non-YouTube + social (if available)."""
    def _norm(url: str) -> str:
        return (url or "").strip()

    buckets = {"youtube": [], "social": [], "other": []}
    seen: set[str] = set()

    def push(title: str, url: str, kind: str) -> None:
        u = _norm(url)
        if not u or u in seen:
            return
        seen.add(u)
        row = {"title": (title or "Source").strip(), "url": u}
        buckets[kind].append(row)

    # curated links first
    for it in ref_links or []:
        u = _norm(it.get("url", ""))
        t = it.get("title", "Source")
        if is_supported_video_url(u):
            push(t, u, "youtube")
        elif is_social_profile_url(u):
            push(t, u, "social")
        else:
            push(t, u, "other")

    # fallback from selected picks
    for _, cand in (selections or {}).items():
        u = _norm(cand.get("url", ""))
        t = cand.get("title", "Source")
        if is_supported_video_url(u):
            push(t, u, "youtube")
        elif is_social_profile_url(u):
            push(t, u, "social")
        else:
            push(t, u, "other")

    chosen: list[dict] = []
    if buckets["youtube"]:
        chosen.append(buckets["youtube"][0])
    if buckets["other"]:
        chosen.append(buckets["other"][0])
    if buckets["social"]:
        chosen.append(buckets["social"][0])

    for k in ("youtube", "other", "social"):
        for row in buckets[k]:
            if len(chosen) >= max_links:
                break
            if row not in chosen:
                chosen.append(row)

    if not chosen:
        return ""

    heading = "References" if language_code == "en-US" else "संदर्भ"
    lines = [f"\n\n{heading}:"]
    for row in chosen[:max_links]:
        lines.append(f"- {row['title']}: {row['url']}")
    return "\n".join(lines)


def synthesize_speech(text: str, language_code: str) -> str:
    voice_map = {"en-US": "en-US-AriaNeural", "hi-IN": "hi-IN-SwaraNeural"}
    voice       = voice_map.get(language_code, "en-US-AriaNeural")
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate="+0%")
        try:
            asyncio.run(communicate.save(output_file))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(communicate.save(output_file))
            loop.close()
    except Exception:
        lang_map = {"en-US": "en", "hi-IN": "hi"}
        gTTS(text=text, lang=lang_map.get(language_code, "en")).save(output_file)
    return output_file


# ════════════════════════════════════════════════════════════
#  OPENAI CLIENT
# ════════════════════════════════════════════════════════════

client = OpenAI(
    api_key=EURI_API_KEY,
    base_url="https://api.euron.one/api/v1/euri",
)

# ════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Voice AI Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    "<style>[data-testid='stSidebar'] { display: none; }</style>",
    unsafe_allow_html=True,
)

LANGUAGE_OPTIONS = {"English": "en-US", "Hindi": "hi-IN"}

# ── Session state ──────────────────────────────────────────
for key, default in [
    ("step", "record"), ("user_text", ""), ("candidates_map", {}),
    ("ai_picks", {}), ("user_selections", {}), ("final_answer", ""),
    ("language", "en-US"), ("active_sources", []),
    ("answer_format_mode", "General answer"),
    ("news_more_responses", []),
    ("person_profile_query", False),
    ("public_figure_query", False),
    ("cached_google_news", []),
    ("general_reference_links", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Sessions created before "Auto-detect" was removed
if st.session_state.get("answer_format_mode") == "Auto-detect":
    st.session_state.answer_format_mode = "General answer"


def reset_conversation() -> None:
    for k in [
        "step", "user_text", "candidates_map", "ai_picks", "user_selections",
        "final_answer", "active_sources", "news_more_responses",
        "cached_google_news", "general_reference_links",
    ]:
        st.session_state[k] = (
            {}
            if k in ("candidates_map", "ai_picks", "user_selections")
            else [] if k in (
                "active_sources", "news_more_responses",
                "cached_google_news", "general_reference_links",
            )
            else ""
        )
    st.session_state.person_profile_query = False
    st.session_state.public_figure_query = False
    st.session_state.step = "record"
    st.rerun()


# ── Top bar: title (left) · language (right) ─────────────────
_title_col, _lang_col = st.columns([4, 1])
with _title_col:
    st.title("🎤 Voice AI Assistant")
with _lang_col:
    _pad, _lang_inner = st.columns([1, 3])
    with _lang_inner:
        _selected_label = st.selectbox(
            "Language",
            list(LANGUAGE_OPTIONS.keys()),
            key="top_language_select",
        )
language = LANGUAGE_OPTIONS[_selected_label]
st.session_state.language = language

if not SERPAPI_KEY:
    st.warning(
        "⚠️ No `SERPAPI_KEY` found. Social searches use the free DuckDuckGo API "
        "(fewer results). Add `SERPAPI_KEY` for richer disambiguation."
    )

if not YOUTUBE_API_KEY:
    st.info("ℹ️ Set `YOUTUBE_API_KEY` in your environment to enable YouTube search results.")

if not EURI_API_KEY:
    st.error("❌ `EURI_API_KEY` missing.")
    st.stop()


def run_search_pipeline(user_text: str, lang_code: str) -> None:
    """Shared path after we have the question string (from speech or typing)."""
    user_text = user_text.strip()
    if not user_text:
        return

    st.session_state.user_text = user_text
    st.session_state.language = lang_code

    # Search with a clean entity phrase (e.g. "Sachin Tendulkar"), not "who is...".
    search_q = (extract_entity_for_search(user_text).strip() or user_text.strip())
    recency_sensitive = is_recency_sensitive_query(user_text)
    news_like = is_news_like_query(user_text)
    office_holder_query = is_office_holder_query(user_text)

    person_focus = is_person_identity_question(client, user_text)
    st.session_state.person_profile_query = person_focus

    if person_focus:
        public_figure = is_public_figure_query(client, search_q)
        st.session_state.public_figure_query = public_figure
        if public_figure:
            active_sources = list(PUBLIC_FIGURE_PROFILE_SOURCES)
        else:
            active_sources = list(PERSON_PROFILE_SOURCES)
            st.session_state.cached_google_news = []
            st.session_state.general_reference_links = []
    else:
        st.session_state.public_figure_query = False
        active_sources = list(GENERAL_SOURCES)

    st.session_state.active_sources = active_sources

    candidates_map = {}
    ai_picks = {}

    for src in active_sources:
        if src == "wikipedia":
            cands = search_wikipedia(search_q)
        elif src == "web":
            cands = search_web(search_q, num=6)
        elif src == "youtube":
            if is_latest_video_query(user_text):
                cands = search_latest_youtube_from_channel(user_text, max_results=5)
                if not cands:
                    cands = search_youtube_with_local_variants(
                        search_q, user_text, max_results=6, order_by_date=True
                    )
            else:
                cands = search_youtube_with_local_variants(
                    search_q, user_text, max_results=6, order_by_date=recency_sensitive
                )
        else:
            cands = search_multi_candidates(search_q, src, max_results=5)

        candidates_map[src] = cands

        if cands:
            if src == "youtube" and recency_sensitive and not news_like and not office_holder_query:
                # For "current/latest" style queries, trust newest YouTube upload first.
                ai_picks[src] = cands[0]
                ai_picks[src]["ai_reason"] = "Newest upload prioritized for recency-sensitive question."
                ai_picks[src]["ai_confidence"] = "high"
            else:
                ai_picks[src] = ai_pick_best_candidate(client, user_text, cands, src)

    # For incident/office-holder fact queries, avoid stale video becoming the primary evidence
    # unless user explicitly asked for latest video.
    if (news_like or office_holder_query) and not is_latest_video_query(user_text):
        ai_picks.pop("youtube", None)

    cached_news: list[dict] = []
    if not person_focus and SERPAPI_KEY:
        cached_news = search_google_news(search_q, num=6)
    st.session_state.cached_google_news = cached_news

    if not person_focus:
        st.session_state.general_reference_links = build_general_reference_links(
            candidates_map, ai_picks, cached_news, max_links=6
        )
    else:
        st.session_state.general_reference_links = []

    if ai_picks.get("wikipedia"):
        ai_picks["wikipedia"] = enrich_wikipedia_pick(ai_picks["wikipedia"])
    if ai_picks.get("youtube"):
        ai_picks["youtube"] = enrich_youtube_pick(ai_picks["youtube"])

    st.session_state.candidates_map = candidates_map
    st.session_state.ai_picks = ai_picks

    st.session_state.user_selections = {
        src: pick for src, pick in ai_picks.items() if pick
    }
    st.session_state.news_more_responses = []
    st.session_state.step = "answer"

    st.rerun()


# ════════════════════════════════════════════════════════════
#  STEP 1 — RECORD → TRANSCRIBE → MULTI-CANDIDATE SEARCH
# ════════════════════════════════════════════════════════════
if st.session_state.step == "record":

    st.radio(
        "Response format",
        ["News & events — sections", "General answer"],
        key="answer_format_mode",
        horizontal=False,
    )

    audio_input = st.audio_input("🎙️ Record your question")

    st.divider()
    typed_q = st.text_area(
        "Typed question",
        label_visibility="collapsed",
        placeholder=(
            "Examples: Who is Sachin Tendulkar? / सचिन तेंदुलकर कौन हैं? / "
            "Sachin Tendulkar kaun hai?"
        ),
        height=110,
        key="typed_question_input",
    )
    submit_typed = st.button("🔎 Search with typed question", type="secondary")

    if submit_typed:
        if not typed_q or not typed_q.strip():
            st.warning("Please type a question above, or use voice.")
        else:
            run_search_pipeline(typed_q.strip(), language)

    elif audio_input is not None:
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav.write(audio_input.getbuffer())
        temp_wav.close()

        stt_lang_map = {"en-US": "en", "hi-IN": "hi"}
        transcript = None
        transcription_errors = []
        for model_name in list(dict.fromkeys([STT_MODEL, "gpt-4o-mini-transcribe", "whisper-1"])):
            try:
                with open(temp_wav.name, "rb") as af:
                    transcript = client.audio.transcriptions.create(
                        model=model_name, file=af, language=stt_lang_map.get(language, "en"))
                break
            except Exception as err:
                transcription_errors.append(f"{model_name}: {err}")

        os.remove(temp_wav.name)

        if transcript is None:
            st.error("Transcription failed.")
            with st.expander("Details"):
                st.code("\n".join(transcription_errors))
            st.stop()

        user_text = transcript.text.strip()
        if not user_text:
            st.error("Could not understand audio.")
            st.stop()

        run_search_pipeline(user_text, language)


# ════════════════════════════════════════════════════════════
#  STEP 2 — DISAMBIGUATION
# ════════════════════════════════════════════════════════════
elif st.session_state.step == "disambiguate":

    st.markdown(
        f"🗣️ **You asked:** {linkify_http(st.session_state.user_text)}"
    )
    st.markdown("### 🔎 Multiple profiles found — please confirm the correct one")

    user_selections = {}

    # Auto-accept Wikipedia and Web silently
    for src in ["wikipedia", "web"]:
        if st.session_state.ai_picks.get(src):
            user_selections[src] = st.session_state.ai_picks[src]

    for src in DISAMBIGUATION_SOURCES:
        if src not in st.session_state.active_sources:
            continue

        cands = st.session_state.candidates_map.get(src, [])
        if not cands:
            st.markdown(f"**{src.capitalize()}** — no public results found.")
            continue

        ai_best     = st.session_state.ai_picks.get(src, {})
        ai_best_url = ai_best.get("url", "")

        st.markdown(f"---\n#### {src.capitalize()} — {len(cands)} candidate(s) found")

        # Clickable links in markdown (radio options are plain text only in Streamlit)
        for i, c in enumerate(cands):
            st.markdown(f"**Result {i + 1}**  \n{md_link_title_url(c.get('title') or 'Untitled', c.get('url', ''))}")
            if c.get("snippet"):
                sn = c["snippet"][:300] + ("…" if len(c.get("snippet", "")) > 300 else "")
                st.markdown(linkify_http(sn))
            if c.get("url") == ai_best_url:
                confidence = ai_best.get("ai_confidence", "")
                reason     = ai_best.get("ai_reason", "")
                badge      = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "")
                st.caption(
                    f"✅ AI best guess {badge} {confidence}" + (f" — {reason}" if reason else "")
                )
            st.markdown("")

        radio_options = [f"Use result {i + 1}" for i in range(len(cands))]
        radio_options.append("❌ None of these / skip this platform")

        default_idx = 0
        for i, c in enumerate(cands):
            if c.get("url") == ai_best_url:
                default_idx = i
                break

        chosen_label = st.radio(
            f"**Select the correct {src.capitalize()} result (matches numbered results above):**",
            radio_options,
            index=default_idx,
            key=f"radio_{src}",
        )

        chosen_idx = radio_options.index(chosen_label)
        if chosen_idx < len(cands):
            user_selections[src] = cands[chosen_idx]

    if st.button("✅ Confirm & get answer", type="primary"):
        st.session_state.user_selections = user_selections
        st.session_state.step = "answer"
        st.rerun()


# ════════════════════════════════════════════════════════════
#  STEP 3 — ANSWER + TTS
# ════════════════════════════════════════════════════════════
elif st.session_state.step == "answer":

    language   = st.session_state.language
    user_text  = st.session_state.user_text
    selections = st.session_state.user_selections

    st.markdown(f"🗣️ **You asked:** {linkify_http(user_text)}")

    search_q = extract_entity_for_search(user_text).strip() or user_text.strip()

    person_profile = bool(st.session_state.get("person_profile_query"))
    public_figure_profile = bool(st.session_state.get("public_figure_query"))
    if person_profile:
        if public_figure_profile:
            st.caption(
                "👤 **Profile-style question (public figure detected)** — using Wikipedia/Web/YouTube + social sources."
            )
        else:
            st.caption(
                "👤 **Profile-style question (non-public)** — search uses **LinkedIn, Facebook, and Instagram** only."
            )

    ref_links_ui = st.session_state.get("general_reference_links") or []

    # Prioritize inline playback for selected YouTube URL before link lists.
    yt_pick = selections.get("youtube") or {}
    yt_url = (yt_pick.get("url") or "").strip()
    if is_supported_video_url(yt_url):
        st.markdown("### ▶️ Now playing")
        st.video(yt_url)

    if not person_profile and ref_links_ui:
        with st.expander("🔗 Related sources (curated links)", expanded=False):
            for i, it in enumerate(ref_links_ui, 1):
                url = it.get("url") or ""
                st.markdown(
                    f"- **{it.get('kind', 'Link')}**: "
                    f"{md_link_title_url(it.get('title') or 'Source', url)}"
                )
                if is_supported_video_url(url):
                    st.video(url)
                render_social_link_actions(url, f"related_{i}")

    format_mode = st.session_state.get("answer_format_mode", "General answer")
    auto_news = is_news_like_query(user_text)
    is_news_or_event = (format_mode == "News & events — sections") or auto_news
    incident_tracker_mode = is_incident_tracker_query(user_text)
    if auto_news and format_mode != "News & events — sections":
        st.caption("📰 Auto-switched to news retrieval for a latest/incidents query.")

    google_news_items: list[dict] = []
    if person_profile and not public_figure_profile:
        google_news_items = []
    elif is_news_or_event and SERPAPI_KEY:
        google_news_items = search_google_news(search_q, num=6)
    else:
        google_news_items = list(st.session_state.get("cached_google_news") or [])

    if google_news_items:
        with st.expander("📰 Google News cluster (SerpAPI, same query)", expanded=False):
            for n in google_news_items[:6]:
                st.markdown(
                    f"**{n.get('title', '')}**  \n"
                    f"{md_link_title_url('Open', n.get('url', ''))} · "
                    f"{n.get('source', '')} {n.get('date', '')}"
                )

    search_context = build_augmented_search_context(
        selections, google_news_items or None
    )
    gl_block = format_general_links_for_llm(ref_links_ui if not person_profile else [])
    if gl_block:
        search_context = search_context + "\n\n" + gl_block

    profile_guard = ""
    if person_profile and not public_figure_profile:
        profile_guard = (
            "Retrieval is restricted to LinkedIn, Instagram, and Facebook snippets below. "
            "Do not claim facts from Wikipedia or generic web unless they appear in this context.\n\n"
        )

    language_instruction_map = {
        "en-US": "English",
        "hi-IN": "Hindi written in Devanagari script only",
    }
    target_lang_instr = language_instruction_map.get(language, "English")

    if is_news_or_event and incident_tracker_mode:
        system_prompt = (
            profile_guard
            + search_context
            + f"You are summarizing **latest confirmed incidents** from retrieved live context. "
            f"Reply only in {target_lang_instr}. "
            "Do not transliterate Hindi into Latin letters. "
            "Use only facts from the provided context and prioritize the most recent items. "
            "If dates are available, mention them; if not, use relative timing from context. "
            "Do not repeat the same incident in different words.\n\n"
            "Format exactly in this style (translated to reply language):\n"
            "## Latest confirmed incidents\n"
            "- 4 to 6 numbered points, each 2-4 lines.\n"
            "- Each point must describe a distinct incident/development.\n"
            "- Include location and impact (deaths/injuries/displacement/actions) only if stated in context.\n\n"
            "## Situation snapshot\n"
            "- 2 to 4 bullet points with broader status (curfew/internet/security/political response).\n"
            "- If an item is uncertain or conflicting, explicitly say 'not fully confirmed in sources'.\n\n"
            "## Sources\n"
            "- Provide 4 to 6 links from context.\n"
            "- Include at least 2 non-YouTube links when available.\n"
        )
    elif is_news_or_event:
        system_prompt = (
            profile_guard
            + search_context
            + f"You are summarizing **news or current events** from the verified context. "
            f"Reply only in {target_lang_instr}. "
            "Do not transliterate Hindi into Latin letters. "
            "Use facts only from the context; if something is missing, say so. "
            "Prioritize the most recent sources by date/time and avoid stale updates when newer evidence exists.\n\n"
            "For role/office-holder facts (e.g., chief minister, deputy CM), do NOT assert a name unless supported by current "
            "non-video sources in context; otherwise state uncertainty clearly.\n\n"
            "Format your reply as markdown with these sections (translate section titles into "
            f"the reply language; for Hindi use Devanagari for titles and body):\n\n"
            "## Headline\n"
            "One punchy line.\n\n"
            "## What happened\n"
            "2–5 sentences for someone catching up.\n\n"
            "## Key points\n"
            "Bullet list (— or -) with dates, places, numbers when the context includes them.\n\n"
            "## Context or outlook\n"
            "1–3 sentences if supported; otherwise a brief note that detail is limited.\n\n"
            "## Sources\n"
            "Give 3-6 source links where possible. Prefer at least 2 non-YouTube sources "
            "(news websites/newspapers/web reports) when they are available in context.\n"
        )
    else:
        system_prompt = (
            profile_guard
            + search_context
            + f"Reply only in {target_lang_instr}. "
            "Do not transliterate into English letters. "
            "Write a clear answer in 7-10 lines (not too short). "
            "For office-holder questions, avoid asserting outdated names; prefer the newest non-video sources in context, "
            "or explicitly mention uncertainty if evidence conflicts.\n"
            "After the answer body, add a final section titled 'References' (or the equivalent title in the reply language) "
            "and list 3-6 relevant source links from the provided context, including at least 2 non-YouTube links when available. "
            "When RELATED SOURCE LINKS are present, tie facts to those URLs/platforms where possible. "
            "Cite the source platform in your answer. "
            "If the context is insufficient, say so clearly."
        )

    with st.spinner("🤖 Generating answer…"):
        try:
            llm_resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_text},
                ],
                max_tokens=CHAT_MAX_TOKENS,
            )
            reply = llm_resp.choices[0].message.content.strip()
        except Exception as err:
            st.error(f"LLM request failed: {err}")
            st.stop()

    reply = enforce_native_language_reply(
        client, user_text, reply, language, target_lang_instr, CHAT_MAX_TOKENS)

    if language == "hi-IN" and not ensure_target_language_text(reply, language):
        reply = fallback_google_translate(reply, language)
        if not ensure_target_language_text(reply, language) or contains_latin_letters(reply):
            reply = "क्षमा करें, अभी हिंदी में उत्तर तैयार करने में समस्या हो रही है।"
            st.warning("Native-script enforcement failed.")

    mixed_links = build_mixed_links_block(
        selections=selections,
        ref_links=ref_links_ui,
        language_code=language,
        max_links=6,
    )
    if mixed_links:
        reply = reply.strip()
        if "References:" in reply or "संदर्भ" in reply:
            extra_head = "\n\nAdditional references:" if language == "en-US" else "\n\nअतिरिक्त संदर्भ:"
            body = mixed_links.split(":", 1)[-1].strip()
            if body:
                reply += f"{extra_head}\n{body}"
        else:
            reply += mixed_links

    if is_news_or_event:
        st.caption("📰 **News & events** — sections below. 🔊 Audio is the same story, optimized for listening.")
    answer_heading = "### 📰 News & events brief" if is_news_or_event else "### 🤖 Answer"
    st.markdown(f"{answer_heading}\n{reply}")

    if is_news_or_event:
        prev_list = [reply] + list(st.session_state.get("news_more_responses", []))
        if st.button("➕ More (new angle)", key="more_news_btn"):
            with st.spinner("🧠 Creating another perspective…"):
                try:
                    extra = generate_additional_news_response(
                        client=client,
                        user_text=user_text,
                        search_context=search_context,
                        target_lang_instr=target_lang_instr,
                        previous_responses=prev_list,
                    )
                    extra = enforce_native_language_reply(
                        client, user_text, extra, language, target_lang_instr, CHAT_MAX_TOKENS
                    )
                    if extra:
                        st.session_state.news_more_responses.append(extra)
                except Exception as err:
                    st.warning(f"Couldn't generate more response right now: {err}")

        for i, extra_text in enumerate(st.session_state.get("news_more_responses", []), 1):
            st.markdown(f"#### 🧩 More response {i}\n{extra_text}")

    tts_text = prepare_markdown_for_tts(reply)
    with st.spinner("🔊 Synthesizing speech…"):
        audio_file = synthesize_speech(tts_text, language)

    st.audio(audio_file, format="audio/mp3")
    os.remove(audio_file)

    st.divider()
    _btn_a, _btn_b = st.columns(2)
    with _btn_a:
        if st.button("🎙️ Ask another question", use_container_width=True):
            reset_conversation()
    with _btn_b:
        if st.button("🔄 Start over", use_container_width=True):
            reset_conversation()
