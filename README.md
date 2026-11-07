# AI Video Summarizer

## Windows setup

1. Install Python 3.10 or 3.11 and select **Add Python to PATH**.
2. Install FFmpeg and make sure `ffmpeg` works in Command Prompt.
3. Extract this folder.
4. Open Command Prompt in this folder.
5. Run:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser, usually at `http://localhost:8501`.

## Notes

- Choose **tiny** for older or low-powered computers. It is the fastest Whisper model but may be less accurate.
- The first run downloads Whisper and PyTorch files and needs internet access.
- Video processing happens on your computer. Long videos can take a while.
- The summary is lightweight and extractive; it selects important sentences instead of downloading a second large language model.
- Urdu speech transcription is supported by Whisper. Topic extraction currently focuses on English words.
