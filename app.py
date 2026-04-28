# pip install streamlit openai gTTS edge-tts deep-translator python-dotenv wikipedia-api requests
# export EURI_API_KEY="your-key"
# Optional: export SERPAPI_KEY="your-key"  (for richer LinkedIn/Instagram/Facebook search)

import streamlit as st
import tempfile
import os
import asyncio
import re
import json
import requests
from urllib.parse import quote_plus

from openai import OpenAI
from gtts import gTTS
import edge_tts
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
load_dotenv()
EURI_API_KEY    = os.getenv("EURI_API_KEY") or os.getenv("OPENAI_API_KEY")
SERPAPI_KEY     = os.getenv("SERPAPI_KEY", "")   # optional – richer social search
STT_MODEL       = os.getenv("STT_MODEL",       "gpt-4o-mini-transcribe")
CHAT_MODEL      = os.getenv("CHAT_MODEL",      "gpt-4.1-mini")
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "512"))
TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", "gpt-4o-mini")

# ═══════════════════════════════════════════════════════════
#  SEARCH LAYER
# ═══════════════════════════════════════════════════════════

# ── Wikipedia ──────────────────────────────────────────────
def search_wikipedia(query: str, sentences: int = 5) -> dict:
    """
    Returns {"source": "Wikipedia", "title": ..., "summary": ..., "url": ...}
    or {"source": "Wikipedia", "error": ...}
    """
    try:
        # Step 1: find the best matching page title
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
        }
        r = requests.get(search_url, params=search_params, timeout=8)
        r.raise_for_status()
        results = r.json().get("query", {}).get("search", [])
        if not results:
            return {"source": "Wikipedia", "error": "No results found."}

        page_title = results[0]["title"]

        # Step 2: fetch the extract (plain-text summary)
        extract_params = {
            "action": "query",
            "titles": page_title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "exsentences": sentences,
            "format": "json",
        }
        r2 = requests.get(search_url, params=extract_params, timeout=8)
        r2.raise_for_status()
        pages = r2.json()["query"]["pages"]
        page = next(iter(pages.values()))
        summary = page.get("extract", "").strip()
        url = f"https://en.wikipedia.org/wiki/{quote_plus(page_title.replace(' ', '_'))}"

        return {"source": "Wikipedia", "title": page_title, "summary": summary, "url": url}
    except Exception as e:
        return {"source": "Wikipedia", "error": str(e)}


# ── DuckDuckGo Instant-Answer API ─────────────────────────
def search_duckduckgo(query: str) -> dict:
    """
    Uses the free DuckDuckGo Instant Answer API.
    Covers general web + public social-media pages.
    Returns {"source": "DuckDuckGo", "abstract": ..., "url": ...}
    """
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1, "skip_disambig": 1}
        r = requests.get(url, params=params, timeout=8, headers={"User-Agent": "VoiceAIBot/1.0"})
        r.raise_for_status()
        data = r.json()
        abstract   = data.get("AbstractText", "").strip()
        result_url = data.get("AbstractURL", "") or data.get("Redirect", "")

        # Fall back to first related topic if no abstract
        if not abstract:
            topics = data.get("RelatedTopics", [])
            for t in topics:
                if isinstance(t, dict) and t.get("Text"):
                    abstract   = t["Text"]
                    result_url = t.get("FirstURL", "")
                    break

        if abstract:
            return {"source": "DuckDuckGo", "abstract": abstract, "url": result_url}
        return {"source": "DuckDuckGo", "error": "No instant answer found."}
    except Exception as e:
        return {"source": "DuckDuckGo", "error": str(e)}


# ── Social profile search via DuckDuckGo ──────────────────
def search_social_platform(query: str, platform: str) -> dict:
    """
    Searches public profiles/pages on LinkedIn, Instagram, or Facebook
    by prepending 'site:<domain>' to the query and hitting DuckDuckGo.

    NOTE: LinkedIn/Facebook/Instagram do NOT have open public APIs.
    This approach only surfaces publicly indexed pages and therefore
    cannot access private profiles, authenticated content, or feeds.
    """
    site_map = {
        "linkedin":  "site:linkedin.com",
        "instagram": "site:instagram.com",
        "facebook":  "site:facebook.com",
    }
    site_filter = site_map.get(platform.lower(), "")
    full_query  = f"{site_filter} {query}".strip()
    result      = search_duckduckgo(full_query)
    result["platform"] = platform.capitalize()
    return result


# ── SerpAPI (optional, richer results) ────────────────────
def search_serpapi(query: str, platform: str = "") -> dict:
    """
    Uses SerpAPI (Google backend) for richer search results.
    Requires SERPAPI_KEY env variable.
    """
    if not SERPAPI_KEY:
        return {"source": "SerpAPI", "error": "SERPAPI_KEY not configured."}
    try:
        site_map = {
            "linkedin":  "site:linkedin.com",
            "instagram": "site:instagram.com",
            "facebook":  "site:facebook.com",
        }
        site_filter = site_map.get(platform.lower(), "") if platform else ""
        full_query  = f"{site_filter} {query}".strip()
        url = "https://serpapi.com/search"
        params = {"q": full_query, "api_key": SERPAPI_KEY, "num": 3, "output": "json"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        organic = data.get("organic_results", [])
        if not organic:
            return {"source": "SerpAPI", "error": "No results."}
        top = organic[0]
        return {
            "source": "SerpAPI",
            "title":   top.get("title", ""),
            "snippet": top.get("snippet", ""),
            "url":     top.get("link", ""),
        }
    except Exception as e:
        return {"source": "SerpAPI", "error": str(e)}


# ── Master search orchestrator ─────────────────────────────
def gather_search_context(query: str, sources: list[str]) -> str:
    """
    Runs the requested searches in sequence and returns a single
    formatted context block to inject into the LLM system prompt.
    """
    snippets = []

    for source in sources:
        s = source.lower()

        if s == "wikipedia":
            res = search_wikipedia(query)
            if "summary" in res:
                snippets.append(
                    f"[Wikipedia – {res['title']}]\n{res['summary']}\nURL: {res['url']}"
                )
            else:
                snippets.append(f"[Wikipedia] {res.get('error', 'No data')}")

        elif s == "web":
            res = search_duckduckgo(query)
            if "abstract" in res:
                snippets.append(f"[Web – DuckDuckGo]\n{res['abstract']}\nURL: {res.get('url','')}")
            else:
                snippets.append(f"[Web – DuckDuckGo] {res.get('error', 'No data')}")

        elif s in {"linkedin", "instagram", "facebook"}:
            # Try SerpAPI first (richer); fall back to DuckDuckGo
            if SERPAPI_KEY:
                res = search_serpapi(query, platform=s)
                if "snippet" in res:
                    snippets.append(
                        f"[{s.capitalize()} via SerpAPI – {res.get('title','')}]\n"
                        f"{res['snippet']}\nURL: {res.get('url','')}"
                    )
                else:
                    snippets.append(f"[{s.capitalize()} via SerpAPI] {res.get('error','No data')}")
            else:
                res = search_social_platform(query, s)
                if "abstract" in res:
                    snippets.append(
                        f"[{s.capitalize()} – public page]\n{res['abstract']}\nURL: {res.get('url','')}"
                    )
                else:
                    snippets.append(
                        f"[{s.capitalize()}] {res.get('error', 'No publicly indexed data found.')}"
                    )

    if not snippets:
        return ""
    return (
        "=== VERIFIED SEARCH CONTEXT ===\n"
        + "\n\n".join(snippets)
        + "\n=== END OF CONTEXT ===\n\n"
        "Use the above context to answer accurately. "
        "If the context doesn't cover the question, use your general knowledge but say so."
    )


# ═══════════════════════════════════════════════════════════
#  LANGUAGE UTILITIES  (unchanged from original)
# ═══════════════════════════════════════════════════════════

def ensure_target_language_text(text: str, language_code: str) -> bool:
    script_checks = {
        "hi-IN": r"[\u0900-\u097F]",
        "te-IN": r"[\u0C00-\u0C7F]",
    }
    pattern = script_checks.get(language_code)
    if not pattern:
        return True
    return bool(re.search(pattern, text))


def contains_latin_letters(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def translate_to_target_language(
    client: OpenAI, text: str, language_code: str,
    target_language_instruction: str, model: str, max_tokens: int
) -> str:
    script_hint = ""
    if language_code == "hi-IN":
        script_hint = "Use only Devanagari script. Do not use any English letters."
    elif language_code == "te-IN":
        script_hint = "Use only Telugu script. Do not use any English letters."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Translate the text into {target_language_instruction}. "
                    "Return only translated text. Do not include English. "
                    f"{script_hint}"
                ),
            },
            {"role": "user", "content": text},
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def answer_directly_in_target_language(
    client: OpenAI, user_text: str,
    target_language_instruction: str, model: str, max_tokens: int
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Answer the question only in {target_language_instruction}. "
                    "Do not use English. Keep response concise and natural."
                ),
            },
            {"role": "user", "content": user_text},
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def enforce_native_language_reply(
    client: OpenAI, user_text: str, draft_reply: str, language_code: str,
    target_language_instruction: str, max_tokens: int
) -> str:
    if language_code == "en-US":
        return draft_reply

    model_candidates = list(dict.fromkeys([TRANSLATE_MODEL, CHAT_MODEL, "gpt-4o-mini"]))
    reply = draft_reply

    for model_name in model_candidates:
        try:
            translated = translate_to_target_language(
                client, reply, language_code, target_language_instruction, model_name, max_tokens
            )
            if translated:
                reply = translated
        except Exception:
            pass

        if language_code in {"hi-IN", "te-IN"} and not ensure_target_language_text(reply, language_code):
            try:
                regenerated = answer_directly_in_target_language(
                    client, user_text, target_language_instruction, model_name, max_tokens
                )
                if regenerated:
                    reply = regenerated
            except Exception:
                pass

        if language_code in {"hi-IN", "te-IN"}:
            for _ in range(2):
                if ensure_target_language_text(reply, language_code) and not contains_latin_letters(reply):
                    break
                try:
                    strict_retry = translate_to_target_language(
                        client, reply, language_code, target_language_instruction, model_name, max_tokens
                    )
                    if strict_retry:
                        reply = strict_retry
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
    voice_map = {
        "en-US": "en-US-AriaNeural",
        "hi-IN": "hi-IN-SwaraNeural",
        "te-IN": "te-IN-ShrutiNeural",
        "es-ES": "es-ES-ElviraNeural",
    }
    voice = voice_map.get(language_code, "en-US-AriaNeural")
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
        tts_lang = lang_map.get(language_code, "en")
        gTTS(text=text, lang=tts_lang).save(output_file)

    return output_file


# ═══════════════════════════════════════════════════════════
#  OPENAI CLIENT
# ═══════════════════════════════════════════════════════════

client = OpenAI(
    api_key=EURI_API_KEY,
    base_url="https://api.euron.one/api/v1/euri",
)

# ═══════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════

st.set_page_config(page_title="Voice AI Assistant", layout="centered")
st.title("🎤 Voice AI Assistant (Multilingual + Search)")
st.caption("Grounded answers via Wikipedia, Web, LinkedIn, Instagram & Facebook (public data).")

# ── Language selector ──────────────────────────────────────
language_options = {
    "English": "en-US",
    "Hindi":   "hi-IN",
    "Telugu":  "te-IN",
    "Spanish": "es-ES",
}
selected_language_label = st.selectbox("Select Language", list(language_options.keys()))
language = language_options[selected_language_label]

# ── Search source selector ─────────────────────────────────
st.markdown("#### 🔍 Search Sources")
col1, col2 = st.columns(2)
with col1:
    use_wikipedia  = st.checkbox("Wikipedia",  value=True)
    use_web        = st.checkbox("Web (DuckDuckGo)", value=True)
with col2:
    use_linkedin   = st.checkbox("LinkedIn (public profiles)")
    use_instagram  = st.checkbox("Instagram (public profiles)")
    use_facebook   = st.checkbox("Facebook (public pages)")

if (use_linkedin or use_instagram or use_facebook) and not SERPAPI_KEY:
    st.info(
        "ℹ️ **No SERPAPI_KEY detected.** Social searches will use DuckDuckGo to find publicly "
        "indexed profile pages. For richer results, add `SERPAPI_KEY` to your environment variables. "
        "\n\n⚠️ LinkedIn, Instagram, and Facebook **do not provide public APIs**. "
        "Only publicly visible/indexed pages can be retrieved."
    )

st.divider()

if not EURI_API_KEY:
    st.error("EURI_API_KEY is missing. Add it in Render / your .env file.")
    st.stop()

# ── Audio input ────────────────────────────────────────────
audio_input = st.audio_input("Record your voice")

if audio_input is not None:
    # ── Save temp WAV ──────────────────────────────────────
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav.write(audio_input.getbuffer())
    temp_wav.close()

    # ── Speech-to-Text ─────────────────────────────────────
    stt_lang_map = {"en-US": "en", "hi-IN": "hi", "te-IN": "te", "es-ES": "es"}
    stt_lang = stt_lang_map.get(language, "en")

    transcript = None
    transcription_errors = []
    for model_name in list(dict.fromkeys([STT_MODEL, "gpt-4o-mini-transcribe", "whisper-1"])):
        try:
            with open(temp_wav.name, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model=model_name, file=audio_file, language=stt_lang
                )
            break
        except Exception as err:
            transcription_errors.append(f"{model_name}: {err}")

    if transcript is None:
        os.remove(temp_wav.name)
        st.error("Transcription failed for all configured models.")
        with st.expander("Transcription error details"):
            st.code("\n".join(transcription_errors))
        st.stop()

    user_text = transcript.text.strip()
    if not user_text:
        st.error("Could not understand audio.")
        st.stop()
    st.write(f"🗣 **You said:** {user_text}")

    # ── Search ─────────────────────────────────────────────
    active_sources = []
    if use_wikipedia:  active_sources.append("wikipedia")
    if use_web:        active_sources.append("web")
    if use_linkedin:   active_sources.append("linkedin")
    if use_instagram:  active_sources.append("instagram")
    if use_facebook:   active_sources.append("facebook")

    search_context = ""
    if active_sources:
        with st.spinner("🔍 Searching selected sources…"):
            search_context = gather_search_context(user_text, active_sources)

        with st.expander("📚 Search context used by AI", expanded=False):
            st.text(search_context if search_context else "No relevant context found.")

    # ── Language instruction ───────────────────────────────
    language_instruction_map = {
        "en-US": "English",
        "hi-IN": "Hindi written in Devanagari script only",
        "te-IN": "Telugu written in Telugu script only",
        "es-ES": "Spanish",
    }
    target_language_instruction = language_instruction_map.get(language, "English")

    system_prompt = (
        search_context
        + f"Reply only in {target_language_instruction}. "
        "Do not transliterate into English letters. "
        "Answer concisely and naturally. "
        "If search context is provided, base your answer on it and cite the source name."
    )

    # ── LLM call ───────────────────────────────────────────
    try:
        llm_response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_text},
            ],
            max_tokens=CHAT_MAX_TOKENS,
        )
    except Exception as err:
        os.remove(temp_wav.name)
        st.error(f"LLM request failed: {err}")
        st.stop()

    reply = llm_response.choices[0].message.content.strip()
    reply = enforce_native_language_reply(
        client, user_text, reply, language, target_language_instruction, CHAT_MAX_TOKENS
    )

    # ── Script enforcement for Hindi / Telugu ──────────────
    if language in {"hi-IN", "te-IN"} and not ensure_target_language_text(reply, language):
        reply = fallback_google_translate(reply, language)
        if not ensure_target_language_text(reply, language) or contains_latin_letters(reply):
            if language == "hi-IN":
                reply = "क्षमा करें, अभी हिंदी में उत्तर तैयार करने में समस्या हो रही है। कृपया फिर से प्रयास करें।"
            elif language == "te-IN":
                reply = "క్షమించండి, ప్రస్తుతం తెలుగులో సమాధానం ఇవ్వడంలో సమస్య ఉంది. దయచేసి మళ్లీ ప్రయత్నించండి."
            st.warning("Native-script enforcement failed.")

    st.write(f"🤖 **AI says:** {reply}")

    # ── TTS ────────────────────────────────────────────────
    audio_file = synthesize_speech(reply, language)
    st.audio(audio_file, format="audio/mp3")

    os.remove(temp_wav.name)
    os.remove(audio_file)
