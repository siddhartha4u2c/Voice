# pip install streamlit openai gTTS edge-tts deep-translator python-dotenv requests
# export EURI_API_KEY="your-key"
# Optional: export SERPAPI_KEY="your-key"  (strongly recommended for social search)

import streamlit as st
import tempfile
import os
import asyncio
import re
import json
import requests
from urllib.parse import quote_plus
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
STT_MODEL       = os.getenv("STT_MODEL",       "gpt-4o-mini-transcribe")
CHAT_MODEL      = os.getenv("CHAT_MODEL",      "gpt-4.1-mini")
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "512"))
TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", "gpt-4o-mini")

SOCIAL_PLATFORMS = ["linkedin", "instagram", "facebook"]

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


def search_multi_candidates(query: str, platform: str, max_results: int = 5) -> list:
    """
    Runs 4 query variations for `platform` in parallel and returns
    a de-duplicated ranked list of candidate results.

    Variations are designed to surface DIFFERENT people/pages with
    the same name so the AI and user can pick the right one.
    """
    site_map = {
        "linkedin":  "site:linkedin.com/in OR site:linkedin.com/pub",
        "instagram": "site:instagram.com",
        "facebook":  "site:facebook.com",
    }
    site_filter = site_map.get(platform, "")

    variations = list(dict.fromkeys([
        f"{site_filter} {query}".strip(),
        f"{site_filter} \"{query}\"".strip(),          # exact name
        f"{site_filter} {query} profile".strip(),
        f"{site_filter} {query} official".strip(),
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

    # Exact-name title matches first
    query_lower = query.lower()
    all_results.sort(key=lambda x: (0 if query_lower in x.get("title", "").lower() else 1))

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


def search_web(query: str) -> list:
    hits = _serpapi_search(query, num=5) if SERPAPI_KEY else _ddg_search(query)
    for i, h in enumerate(hits, 1):
        h["rank"]     = i
        h["platform"] = "Web"
    return hits


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
    script_checks = {"hi-IN": r"[\u0900-\u097F]", "te-IN": r"[\u0C00-\u0C7F]"}
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
    elif language_code == "te-IN":
        script_hint = "Use only Telugu script. Do not use any English letters."
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
        if language_code in {"hi-IN", "te-IN"} and not ensure_target_language_text(reply, language_code):
            try:
                r = answer_directly_in_target_language(client, user_text, target_lang_instr, model_name, max_tokens)
                if r: reply = r
            except Exception:
                pass
        if language_code in {"hi-IN", "te-IN"}:
            for _ in range(2):
                if ensure_target_language_text(reply, language_code) and not contains_latin_letters(reply):
                    break
                try:
                    s = translate_to_target_language(client, reply, language_code, target_lang_instr, model_name, max_tokens)
                    if s: reply = s
                except Exception:
                    break
        if language_code not in {"hi-IN", "te-IN"}:
            return reply
        if ensure_target_language_text(reply, language_code) and not contains_latin_letters(reply):
            return reply
    return reply


def fallback_google_translate(text: str, language_code: str) -> str:
    target_map = {"hi-IN": "hi", "te-IN": "te", "es-ES": "es"}
    target = target_map.get(language_code)
    if not target:
        return text
    try:
        return GoogleTranslator(source="auto", target=target).translate(text)
    except Exception:
        return text


def synthesize_speech(text: str, language_code: str) -> str:
    voice_map = {"en-US": "en-US-AriaNeural", "hi-IN": "hi-IN-SwaraNeural",
                 "te-IN": "te-IN-ShrutiNeural", "es-ES": "es-ES-ElviraNeural"}
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
        lang_map = {"en-US": "en", "hi-IN": "hi", "te-IN": "te", "es-ES": "es"}
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

st.set_page_config(page_title="Voice AI Assistant", layout="centered")
st.title("🎤 Voice AI Assistant")
st.caption("Multilingual · Grounded · Multi-result disambiguation for same-name profiles")

# ── Session state ──────────────────────────────────────────
for key, default in [
    ("step", "record"), ("user_text", ""), ("candidates_map", {}),
    ("ai_picks", {}), ("user_selections", {}), ("final_answer", ""),
    ("language", "en-US"), ("active_sources", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    language_options    = {"English": "en-US", "Hindi": "hi-IN", "Telugu": "te-IN", "Spanish": "es-ES"}
    selected_lang_label = st.selectbox("Language", list(language_options.keys()))
    language            = language_options[selected_lang_label]

    st.markdown("---")
    st.markdown("**🔍 Search Sources**")
    use_wikipedia = st.checkbox("Wikipedia",                value=True)
    use_web       = st.checkbox("Web (DuckDuckGo / Google)", value=True)
    use_linkedin  = st.checkbox("LinkedIn (public)")
    use_instagram = st.checkbox("Instagram (public)")
    use_facebook  = st.checkbox("Facebook (public)")

    if (use_linkedin or use_instagram or use_facebook) and not SERPAPI_KEY:
        st.warning(
            "⚠️ No `SERPAPI_KEY` found. Social searches use the free DuckDuckGo API "
            "(fewer results). Add `SERPAPI_KEY` for richer disambiguation."
        )
    st.markdown("---")
    st.caption(
        "ℹ️ LinkedIn, Instagram & Facebook have **no public APIs**. "
        "Only publicly indexed pages are accessible."
    )
    if st.button("🔄 Start Over"):
        for k in ["step","user_text","candidates_map","ai_picks","user_selections","final_answer","active_sources"]:
            st.session_state[k] = {} if k in ("candidates_map","ai_picks","user_selections") \
                                   else [] if k == "active_sources" else ""
        st.session_state.step = "record"
        st.rerun()

if not EURI_API_KEY:
    st.error("❌ `EURI_API_KEY` missing.")
    st.stop()

# ════════════════════════════════════════════════════════════
#  STEP 1 — RECORD → TRANSCRIBE → MULTI-CANDIDATE SEARCH
# ════════════════════════════════════════════════════════════
if st.session_state.step == "record":

    audio_input = st.audio_input("🎙️ Record your question")

    if audio_input is not None:
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav.write(audio_input.getbuffer())
        temp_wav.close()

        stt_lang_map = {"en-US": "en", "hi-IN": "hi", "te-IN": "te", "es-ES": "es"}
        transcript   = None
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

        st.session_state.user_text  = user_text
        st.session_state.language   = language

        active_sources = (
            (["wikipedia"] if use_wikipedia else []) +
            (["web"]       if use_web       else []) +
            (["linkedin"]  if use_linkedin  else []) +
            (["instagram"] if use_instagram else []) +
            (["facebook"]  if use_facebook  else [])
        )
        st.session_state.active_sources = active_sources

        candidates_map = {}
        ai_picks       = {}

        with st.spinner("🔍 Running multi-candidate search across all sources…"):
            for src in active_sources:
                if src == "wikipedia":
                    cands = search_wikipedia(user_text)
                elif src == "web":
                    cands = search_web(user_text)
                else:
                    cands = search_multi_candidates(user_text, src, max_results=5)

                candidates_map[src] = cands

                if cands:
                    ai_picks[src] = ai_pick_best_candidate(client, user_text, cands, src)

        st.session_state.candidates_map = candidates_map
        st.session_state.ai_picks       = ai_picks

        # Only show disambiguation UI for social platforms with >1 result
        needs_disambiguation = any(
            len(candidates_map.get(src, [])) > 1
            for src in SOCIAL_PLATFORMS
            if src in active_sources
        )

        if needs_disambiguation:
            st.session_state.step = "disambiguate"
        else:
            st.session_state.user_selections = {
                src: pick for src, pick in ai_picks.items() if pick
            }
            st.session_state.step = "answer"

        st.rerun()


# ════════════════════════════════════════════════════════════
#  STEP 2 — DISAMBIGUATION
# ════════════════════════════════════════════════════════════
elif st.session_state.step == "disambiguate":

    st.info(f"🗣️ **You asked:** {st.session_state.user_text}")
    st.markdown("### 🔎 Multiple profiles found — please confirm the correct one")
    st.caption("The AI has pre-selected its best guess (✅ badge). Override if needed, then confirm.")

    user_selections = {}

    # Auto-accept Wikipedia and Web silently
    for src in ["wikipedia", "web"]:
        if st.session_state.ai_picks.get(src):
            user_selections[src] = st.session_state.ai_picks[src]

    for src in SOCIAL_PLATFORMS:
        if src not in st.session_state.active_sources:
            continue

        cands = st.session_state.candidates_map.get(src, [])
        if not cands:
            st.markdown(f"**{src.capitalize()}** — no public results found.")
            continue

        ai_best     = st.session_state.ai_picks.get(src, {})
        ai_best_url = ai_best.get("url", "")

        st.markdown(f"---\n#### {src.capitalize()} — {len(cands)} candidate(s) found")

        # Build radio option labels
        options = []
        for c in cands:
            lines = [f"**{c['title']}**"]
            if c.get("snippet"):
                lines.append(c["snippet"][:160] + ("…" if len(c["snippet"]) > 160 else ""))
            if c.get("url"):
                lines.append(f"🔗 `{c['url']}`")
            if c.get("url") == ai_best_url:
                confidence = ai_best.get("ai_confidence", "")
                reason     = ai_best.get("ai_reason", "")
                badge      = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "")
                lines.append(f"✅ *AI best guess* {badge} {confidence}" + (f" — {reason}" if reason else ""))
            options.append("\n\n".join(lines))

        options.append("❌ None of these / skip this platform")

        # Default index = AI best guess
        default_idx = 0
        for i, c in enumerate(cands):
            if c.get("url") == ai_best_url:
                default_idx = i
                break

        chosen_label = st.radio(
            f"Select the correct {src.capitalize()} profile:",
            options,
            index=default_idx,
            key=f"radio_{src}",
        )

        chosen_idx = options.index(chosen_label)
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

    st.info(f"🗣️ **You asked:** {user_text}")

    if selections:
        with st.expander("📌 Sources used for this answer", expanded=False):
            for platform, cand in selections.items():
                badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                    cand.get("ai_confidence", ""), "")
                st.markdown(
                    f"**{platform.capitalize()}** → [{cand.get('title','')}]({cand.get('url','')})\n\n"
                    f"> {cand.get('snippet','')[:250]}\n\n"
                    + (f"{badge} AI confidence: **{cand['ai_confidence']}** — {cand.get('ai_reason','')}"
                       if cand.get("ai_confidence") else "")
                )

    search_context = build_context_from_selections(selections)

    language_instruction_map = {
        "en-US": "English",
        "hi-IN": "Hindi written in Devanagari script only",
        "te-IN": "Telugu written in Telugu script only",
        "es-ES": "Spanish",
    }
    target_lang_instr = language_instruction_map.get(language, "English")

    system_prompt = (
        search_context
        + f"Reply only in {target_lang_instr}. "
        "Do not transliterate into English letters. "
        "Answer concisely and naturally. "
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

    if language in {"hi-IN", "te-IN"} and not ensure_target_language_text(reply, language):
        reply = fallback_google_translate(reply, language)
        if not ensure_target_language_text(reply, language) or contains_latin_letters(reply):
            reply = (
                "क्षमा करें, अभी हिंदी में उत्तर तैयार करने में समस्या हो रही है।"
                if language == "hi-IN" else
                "క్షమించండి, ప్రస్తుతం తెలుగులో సమాధానం ఇవ్వడంలో సమస్య ఉంది."
            )
            st.warning("Native-script enforcement failed.")

    st.markdown(f"### 🤖 Answer\n{reply}")

    with st.spinner("🔊 Synthesizing speech…"):
        audio_file = synthesize_speech(reply, language)

    st.audio(audio_file, format="audio/mp3")
    os.remove(audio_file)

    st.divider()
    if st.button("🎙️ Ask another question"):
        for k in ["step","user_text","candidates_map","ai_picks","user_selections","final_answer","active_sources"]:
            st.session_state[k] = {} if k in ("candidates_map","ai_picks","user_selections") \
                                   else [] if k == "active_sources" else ""
        st.session_state.step = "record"
        st.rerun()
