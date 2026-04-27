#pip install streamlit openai gTTS
#export EURI_API_KEY="your-key"
import streamlit as st
import tempfile
import os  
import asyncio
import re

from openai import OpenAI
from gtts import gTTS
import edge_tts
from deep_translator import GoogleTranslator

# ---------------- CONFIG ----------------
EURI_API_KEY = os.getenv("EURI_API_KEY")
STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "512"))
TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", "gpt-4o-mini")


def ensure_target_language_text(text: str, language_code: str) -> str:
    # If the reply is not in the expected script, force a translation pass.
    script_checks = {
        "hi-IN": r"[\u0900-\u097F]",  # Devanagari
        "te-IN": r"[\u0C00-\u0C7F]",  # Telugu
    }
    pattern = script_checks.get(language_code)
    if not pattern:
        return text
    if re.search(pattern, text):
        return text
    return ""


def contains_latin_letters(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def translate_to_target_language(
    client: OpenAI,
    text: str,
    language_code: str,
    target_language_instruction: str,
    model: str,
    max_tokens: int
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
    client: OpenAI, user_text: str, target_language_instruction: str, model: str, max_tokens: int
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

    model_candidates = [TRANSLATE_MODEL, CHAT_MODEL, "gpt-4o-mini"]
    model_candidates = list(dict.fromkeys(model_candidates))
    reply = draft_reply

    for model_name in model_candidates:
        # Pass 1: translate current draft to target language.
        try:
            translated = translate_to_target_language(
                client, reply, language_code, target_language_instruction, model_name, max_tokens
            )
            if translated:
                reply = translated
        except Exception:
            pass

        # Pass 2: if script-based language still wrong, regenerate from source question.
        if language_code in {"hi-IN", "te-IN"} and not ensure_target_language_text(reply, language_code):
            try:
                regenerated = answer_directly_in_target_language(
                    client, user_text, target_language_instruction, model_name, max_tokens
                )
                if regenerated:
                    reply = regenerated
            except Exception:
                pass

        # Pass 3: strict script retry loop for Hindi/Telugu.
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
    target_map = {
        "hi-IN": "hi",
        "te-IN": "te",
        "es-ES": "es",
    }
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
        # Fallback to gTTS if edge-tts voice/API fails.
        lang_map = {
            "en-US": "en",
            "hi-IN": "hi",
            "te-IN": "te",
            "es-ES": "es",
        }
        tts_lang = lang_map.get(language_code, "en")
        gTTS(text=text, lang=tts_lang).save(output_file)

    return output_file

client = OpenAI(
    api_key=EURI_API_KEY,
    base_url="https://api.euron.one/api/v1/euri",
)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Voice AI Assistant", layout="centered")
st.title("🎤 Voice AI Assistant (Multilingual)")
st.caption("Works on Render using browser microphone input.")

language = st.selectbox(
    "Select Language",
    {
        "English": "en-US",
        "Hindi": "hi-IN",
        "Telugu": "te-IN",
        "Spanish": "es-ES"
    }
)

if not EURI_API_KEY:
    st.error("EURI_API_KEY is missing. Add it in Render Environment Variables.")
    st.stop()

audio_input = st.audio_input("Record your voice")

if audio_input is not None:
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav.write(audio_input.getbuffer())
    temp_wav.close()

    # ---------------- SPEECH TO TEXT (Whisper via EURI) ----------------
    stt_lang_map = {
        "en-US": "en",
        "hi-IN": "hi",
        "te-IN": "te",
        "es-ES": "es",
    }
    stt_lang = stt_lang_map.get(language, "en")

    transcript = None
    transcription_errors = []
    stt_candidates = [STT_MODEL, "gpt-4o-mini-transcribe", "whisper-1"]
    # Keep order while removing duplicates.
    stt_candidates = list(dict.fromkeys(stt_candidates))

    for model_name in stt_candidates:
        try:
            with open(temp_wav.name, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model=model_name,
                    file=audio_file,
                    language=stt_lang,
                )
            break
        except Exception as err:
            transcription_errors.append(f"{model_name}: {err}")

    if transcript is None:
        os.remove(temp_wav.name)
        st.error(
            "Transcription failed for all configured models. "
            "Set STT_MODEL in Render env vars to a model available in your EURI account."
        )
        with st.expander("Transcription error details"):
            st.code("\n".join(transcription_errors))
        st.stop()

    user_text = transcript.text.strip()
    if not user_text:
        st.error("Could not understand audio")
        st.stop()
    st.write(f"🗣 You said: **{user_text}**")

    # ---------------- LLM CALL ----------------
    language_instruction_map = {
        "en-US": "English",
        "hi-IN": "Hindi written in Devanagari script only",
        "te-IN": "Telugu written in Telugu script only",
        "es-ES": "Spanish",
    }
    target_language_instruction = language_instruction_map.get(language, "English")

    try:
        llm_response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Reply only in the target language requested by the user. "
                        f"Target language: {target_language_instruction}. "
                        "Do not transliterate into English letters."
                    ),
                },
                {"role": "user", "content": user_text},
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

    if language in {"hi-IN", "te-IN"} and not ensure_target_language_text(reply, language):
        reply = fallback_google_translate(reply, language)
        if language in {"hi-IN", "te-IN"} and (
            not ensure_target_language_text(reply, language) or contains_latin_letters(reply)
        ):
            if language == "hi-IN":
                reply = "क्षमा करें, अभी हिंदी में उत्तर तैयार करने में समस्या हो रही है। कृपया फिर से प्रयास करें।"
            elif language == "te-IN":
                reply = "క్షమించండి, ప్రస్తుతం తెలుగులో సమాధానం ఇవ్వడంలో సమస్య ఉంది. దయచేసి మళ్లీ ప్రయత్నించండి."
            st.warning("Native-script enforcement failed for the selected model/provider response.")

    st.write(f"🤖 AI says: **{reply}**")

    # ---------------- TEXT TO SPEECH (multilingual neural voices) ----------------
    audio_file = synthesize_speech(reply, language)

    # Play audio in Streamlit
    st.audio(audio_file, format="audio/mp3")

    os.remove(temp_wav.name)
    os.remove(audio_file)
