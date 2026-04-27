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

# ---------------- CONFIG ----------------
EURI_API_KEY = os.getenv("EURI_API_KEY")
STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "512"))


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
    if not ensure_target_language_text(reply, language):
        try:
            translate_response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Translate the text into {target_language_instruction}. "
                            "Return only the translated text, natural and fluent."
                        ),
                    },
                    {"role": "user", "content": reply},
                ],
                max_tokens=CHAT_MAX_TOKENS,
            )
            translated = translate_response.choices[0].message.content.strip()
            if ensure_target_language_text(translated, language):
                reply = translated
        except Exception:
            pass

    st.write(f"🤖 AI says: **{reply}**")

    # ---------------- TEXT TO SPEECH (multilingual neural voices) ----------------
    audio_file = synthesize_speech(reply, language)

    # Play audio in Streamlit
    st.audio(audio_file, format="audio/mp3")

    os.remove(temp_wav.name)
    os.remove(audio_file)
