#pip install streamlit openai sounddevice scipy gTTS
#export EURI_API_KEY="your-key"
import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wav
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

language = st.selectbox(
    "Select Language",
    {
        "English": "en-US",
        "Hindi": "hi-IN",
        "Telugu": "te-IN",
        "Spanish": "es-ES"
    }
)

duration = st.slider("Recording duration (seconds)", 3, 10, 5)

if st.button("🎙 Record & Ask"):
    st.info("Recording... Speak now")

    fs = 16000
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_wav.name, fs, recording)

    st.success("Recording complete")

    # ---------------- SPEECH TO TEXT (Whisper via EURI) ----------------
    stt_lang_map = {
        "en-US": "en",
        "hi-IN": "hi",
        "te-IN": "te",
        "es-ES": "es",
    }
    stt_lang = stt_lang_map.get(language, "en")

    with open(temp_wav.name, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=stt_lang,
        )

    user_text = transcript.text.strip()
    if not user_text:
        st.error("Could not understand audio")
        st.stop()
    st.write(f"🗣 You said: **{user_text}**")

    # ---------------- LLM CALL ----------------
    llm_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Reply in the same language as the user."},
            {"role": "user", "content": user_text},
        ],
    )

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

    audio_file = "response.mp3"
    tts_response.save(audio_file)

    # Play audio in Streamlit
    st.audio(audio_file, format="audio/mp3")

    os.remove(temp_wav.name)
