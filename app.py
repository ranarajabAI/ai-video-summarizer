"""
AI Video Summarizer — lightweight single-file Streamlit app
Requires FFmpeg and Python packages in requirements.txt.
Run: streamlit run app.py
"""
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from functools import lru_cache

import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

APP_TITLE = "AI Video Summarizer"

@lru_cache(maxsize=2)
def load_whisper(model_size="tiny"):
    import whisper
    return whisper.load_model(model_size)

def extract_audio(video_path, audio_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("FFmpeg is not installed or is not on PATH.")
    cmd = [ffmpeg, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", audio_path]
    subprocess.run(cmd, check=True, capture_output=True)
    return audio_path

def transcribe(audio_path, model_size="tiny", language=None, translate=False):
    model = load_whisper(model_size)
    return model.transcribe(
        audio_path, language=language,
        task="translate" if translate else "transcribe",
        fp16=False, verbose=False
    )

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?؟۔])\s+", text) if len(s.strip()) > 15]

def extractive_summary(text, max_sentences=5):
    """Lightweight sentence ranking; does not download a second AI model."""
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return "\n".join(f"- {s}" for s in sentences) if sentences else text.strip()
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", text.lower())
    freq = Counter(w for w in words if w not in ENGLISH_STOP_WORDS)
    if not freq:
        chosen = sentences[:max_sentences]
    else:
        scored = []
        for i, sentence in enumerate(sentences):
            tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", sentence.lower())
            score = sum(freq[w] for w in tokens if w in freq) / max(len(tokens), 1)
            scored.append((score, i, sentence))
        chosen = [s for _, _, s in sorted(sorted(scored, reverse=True)[:max_sentences], key=lambda x: x[1])]
    return "\n".join(f"- {s}" for s in chosen)

def detailed_summary(text, max_words=300):
    sentences = split_sentences(text)
    summary = extractive_summary(text, max_sentences=min(12, max(5, len(sentences))))
    return " ".join(summary.replace("- ", "").split()[:max_words])

ACTION_RE = re.compile(r"\b(need to|needs to|should|must|will|let's|action item|next step|deadline|follow up|make sure|going to|todo|to-do)\b", re.I)
DECISION_RE = re.compile(r"\b(decided|agreed|decision|we('ll| will) go with|approved|finalized)\b", re.I)

def extract_action_items(text, limit=10):
    return [s for s in split_sentences(text) if ACTION_RE.search(s)][:limit]

def extract_decisions(text, limit=8):
    return [s for s in split_sentences(text) if DECISION_RE.search(s)][:limit]

def executive_summary(text):
    overview = detailed_summary(text, 120)
    decisions = extract_decisions(text)
    actions = extract_action_items(text)
    return (
        f"**Overview**\n{overview}\n\n"
        "**Key Decisions**\n" + ("\n".join(f"- {x}" for x in decisions) or "- None detected") +
        "\n\n**Action Items**\n" + ("\n".join(f"- {x}" for x in actions) or "- None detected")
    )

def get_summary(text, kind):
    if kind == "Short (5 bullets)":
        return extractive_summary(text, 5)
    if kind == "Detailed (~300 words)":
        return detailed_summary(text, 300)
    return executive_summary(text)

def extract_topics(text, k=8):
    words = re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", text.lower())
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    counts = Counter(words)
    bigrams = Counter(f"{a} {b}" for a, b in zip(words, words[1:]))
    ranked = [b for b, count in bigrams.most_common(k) if count > 1]
    ranked += [w for w, _ in counts.most_common(k * 2) if not any(w in r for r in ranked)]
    return [t.title() for t in ranked[:k]]

def fmt_time(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def generate_timestamps(segments, window_sec=300):
    if not segments:
        return []
    blocks, current, start = [], [], 0.0
    for seg in segments:
        if seg["start"] - start >= window_sec and current:
            blocks.append((start, " ".join(current)))
            current, start = [], seg["start"]
        current.append(seg["text"].strip())
    blocks.append((start, " ".join(current)))
    result = []
    for i, (stamp, block) in enumerate(blocks):
        label = "Introduction" if i == 0 and len(block.split()) < 40 else split_sentences(block)
        if isinstance(label, list):
            label = label[0] if label else "Video discussion"
        result.append(f"{fmt_time(stamp)} {label[:100]}")
    return result

def build_notes(summary, topics, timestamps, transcript):
    return (
        "# Video Notes\n\n## Summary\n" + summary +
        "\n\n## Key Topics\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(topics, 1)) +
        "\n\n## Timestamps\n" + "\n".join(timestamps) +
        "\n\n## Full Transcript\n" + transcript + "\n"
    )

def build_pdf(summary, topics, timestamps, transcript):
    from fpdf import FPDF
    def clean(value):
        return str(value).replace("**", "").encode("latin-1", "replace").decode("latin-1")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    sections = [
        ("AI Video Summary Report", None), ("Summary", summary),
        ("Key Topics", "\n".join(f"{i}. {t}" for i, t in enumerate(topics, 1))),
        ("Timestamps", "\n".join(timestamps)), ("Transcript", transcript)
    ]
    for title, body in sections:
        pdf.set_font("Helvetica", "B", 18 if body is None else 14)
        pdf.multi_cell(
    0, 10, clean(title),
    new_x="LMARGIN",
    new_y="NEXT"
)

pdf.multi_cell(
    0, 5, clean(body),
    new_x="LMARGIN",
    new_y="NEXT"
)
    return bytes(pdf.output())

st.set_page_config(page_title=APP_TITLE, page_icon="🎥", layout="wide")
st.title("🎥 AI Video Summarizer")
st.caption("Upload a video to get a transcript, summary, topics, timestamps and downloadable notes.")

with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox("Speech model (smaller = faster)", ["tiny", "base", "small"], index=0)
    summary_kind = st.selectbox(
        "Summary type",
        ["Short (5 bullets)", "Detailed (~300 words)", "Executive (decisions + actions)"]
    )
    language = st.selectbox(
        "Spoken language",
        ["Auto-detect", "en", "ur", "hi", "ar", "es", "fr", "de", "zh"]
    )
    translate = st.checkbox("Translate speech to English", value=False)
    st.caption("The lightweight summary works locally and does not download a separate summarization model.")

video = st.file_uploader("Upload video", type=["mp4", "mov", "mkv", "avi", "webm"])
if video:
    st.video(video)
    if st.button("Summarize video", type="primary"):
        suffix = os.path.splitext(video.name)[1] or ".mp4"
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "uploaded_video" + suffix)
            audio_path = os.path.join(tmp, "audio.wav")
            with open(video_path, "wb") as handle:
                handle.write(video.getbuffer())
            try:
                with st.status("Working on your video...", expanded=True) as status:
                    st.write("1. Extracting audio...")
                    extract_audio(video_path, audio_path)
                    st.write("2. Converting speech to text...")
                    result = transcribe(
                        audio_path, model_size,
                        None if language == "Auto-detect" else language,
                        translate
                    )
                    transcript = result.get("text", "").strip()
                    if not transcript:
                        status.update(label="No speech detected", state="error")
                        st.warning("No speech was detected. Try a clearer video or check its audio.")
                    else:
                        st.write("3. Creating summary and topics...")
                        summary = get_summary(transcript, summary_kind)
                        topics = extract_topics(transcript)
                        stamps = generate_timestamps(result.get("segments", []))
                        st.session_state["out"] = {
                            "transcript": transcript, "summary": summary,
                            "topics": topics, "stamps": stamps,
                            "words": len(transcript.split()),
                            "lang": result.get("language", "unknown")
                        }
                        status.update(label="Finished!", state="complete")
            except FileNotFoundError as exc:
                st.error(f"{exc} Install FFmpeg, then restart the app.")
            except subprocess.CalledProcessError:
                st.error("FFmpeg could not read this video. Check that the file is valid and try another video.")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")

out = st.session_state.get("out")
if out:
    left, right = st.columns(2)
    left.metric("Transcript words", out["words"])
    right.metric("Detected language", out["lang"])
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🏷️ Topics", "⏱️ Timestamps", "📄 Transcript"])
    with tab1:
        st.markdown(out["summary"])
    with tab2:
        if out["topics"]:
            for i, topic in enumerate(out["topics"], 1):
                st.write(f"{i}. {topic}")
        else:
            st.info("No English-language topics were detected.")
    with tab3:
        st.text("\n".join(out["stamps"]))
    with tab4:
        st.text_area("Full transcript", out["transcript"], height=300)
    st.subheader("📥 Download your results")
    notes = build_notes(out["summary"], out["topics"], out["stamps"], out["transcript"])
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("Summary TXT", out["summary"], "summary.txt")
    c2.download_button("Transcript TXT", out["transcript"], "transcript.txt")
    c3.download_button("Notes Markdown", notes, "video_notes.md")
    c4.download_button(
        "PDF report", build_pdf(out["summary"], out["topics"], out["stamps"], out["transcript"]),
        "video_summary_report.pdf", "application/pdf"
    )
