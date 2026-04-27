#pip install streamlit openai gTTS
#export EURI_API_KEY="your-key"
import streamlit as st
import tempfile
import os  

from openai import OpenAI
from gtts import gTTS

# ---------------- CONFIG ----------------
EURI_API_KEY = os.getenv("EURI_API_KEY")

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

    try:
        with open(temp_wav.name, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=stt_lang,
            )
    except Exception as err:
        os.remove(temp_wav.name)
        st.error(f"Transcription failed: {err}")
        st.stop()

    user_text = transcript.text.strip()
    if not user_text:
        st.error("Could not understand audio")
        st.stop()
    st.write(f"🗣 You said: **{user_text}**")

    # ---------------- LLM CALL ----------------
    try:
        llm_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Reply in the same language as the user."},
                {"role": "user", "content": user_text},
            ],
        )
    except Exception as err:
        os.remove(temp_wav.name)
        st.error(f"LLM request failed: {err}")
        st.stop()

    reply = llm_response.choices[0].message.content
    st.write(f"🤖 AI says: **{reply}**")

    # ---------------- TEXT TO SPEECH (FREE: gTTS) ----------------
    lang_map = {
        "en-US": "en",
        "hi-IN": "hi",
        "te-IN": "te",
        "es-ES": "es",
    }
    tts_lang = lang_map.get(language, "en")
    tts_response = gTTS(text=reply, lang=tts_lang)

    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts_response.save(audio_file)

    # Play audio in Streamlit
    st.audio(audio_file, format="audio/mp3")

    os.remove(temp_wav.name)
    os.remove(audio_file)
