# Voice AI Assistant (Multilingual)

A simple Streamlit app that:
- records voice from microphone
- converts speech to text
- sends text to an LLM via EURI-compatible OpenAI API
- converts AI reply to speech using free `gTTS`

## Requirements

- Python 3.10+
- Microphone access enabled
- Internet connection

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env`:

```env
EURI_API_KEY=your_euri_api_key
```

## Run the app

```bash
streamlit run app.py
```

Then open:
- http://localhost:8501

## Notes

- Speech-to-text uses Whisper via your EURI-compatible OpenAI endpoint.
- TTS uses `gTTS` (free).
- Only `EURI_API_KEY` is required in `.env`.

