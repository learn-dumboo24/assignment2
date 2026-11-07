import os
import streamlit as st
import subprocess
import pytesseract
import cv2
from faster_whisper import WhisperModel
from transformers import pipeline
import tempfile
import textwrap

MAX_VIDEO_BYTES = 1024 ** 3
MAX_PREVIEW_BYTES = 50 * 1024 ** 2

@st.cache_resource
def load_whisper():
    return WhisperModel("small", compute_type="int8")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

st.markdown("<br><br><br>", unsafe_allow_html=True)
col1, spacer, col2 = st.columns([2, 2, 3])

def download_video(link, folder):
    command = [
        "yt-dlp",
        "--no-playlist",
        "--max-filesize", str(MAX_VIDEO_BYTES),
        "--js-runtimes", "node",
        "-f", "bv*[height<=480]+ba/b[height<=480]/b",
        "-o", os.path.join(folder, "video.%(ext)s"),
    ]
    # COOKIES is a root-level Streamlit secret, which Streamlit also exposes as an env var
    cookies = os.environ.get("COOKIES")
    if cookies:
        cookies_path = os.path.join(folder, "cookies.txt")
        with open(cookies_path, "w") as f:
            f.write(cookies)
        command += ["--cookies", cookies_path]
    result = subprocess.run(command + ["--", link], capture_output=True, text=True)
    videos = [f for f in os.listdir(folder) if f.startswith("video.") and not f.endswith(".part")]
    if result.returncode != 0 or not videos:
        lines = (result.stderr or result.stdout or "no output").strip().splitlines()
        st.error(f"Download failed: {lines[-1]}")
        return None
    return os.path.join(folder, videos[0])

def transcribe_audio(path):
    try:
        segments, info = load_whisper().transcribe(path, vad_filter=True)
        progress = st.progress(0.0, "Transcribing...")
        texts = []
        for segment in segments:
            texts.append(segment.text)
            progress.progress(min(segment.end / max(info.duration, 1), 1.0), "Transcribing...")
        progress.empty()
        return " ".join(texts)
    except Exception as e:
        st.warning(f"No transcript, using on-screen text only: {e}")
        return ""

def extract_text_from_frames(path, max_samples=60):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(fps), total // max_samples, 1)
    texts = []
    for frame_number in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break
        text = pytesseract.image_to_string(frame).strip()
        if text and text not in texts:
            texts.append(text)
    cap.release()
    return " ".join(texts)

def summarize_text(text, target_length=1500):
    try:
        summarizer = load_summarizer()
        while True:
            chunks = textwrap.wrap(text, 1000)
            results = summarizer(chunks, max_length=60, min_length=15, do_sample=False, truncation=True, batch_size=4)
            text = " ".join(result['summary_text'] for result in results)
            if len(text) <= target_length or len(chunks) == 1:
                return text
    except Exception as e:
        st.error(f"Summarization failed: {e}")
        return ""

def process_video(video_path):
    with st.spinner("Processing video..."):
        transcript = transcribe_audio(video_path)
        visual_text = extract_text_from_frames(video_path)
        full_text = (transcript + " " + visual_text).strip()
        if not full_text:
            st.warning("No meaningful text found.")
            return
        summary = summarize_text(full_text)
        st.success("Summary generated!")
        st.subheader("🧠 Summary")
        st.write(summary)

with col1:
    st.title("Paste your link here 🔗")
    video_link = st.text_input("Paste a video link (YouTube, Instagram, TikTok, X, Facebook, Reddit...)")
    if st.button("Process Link", key="process_link_button"):
        with tempfile.TemporaryDirectory() as folder:
            video_file = download_video(video_link, folder)
            if video_file:
                process_video(video_file)
                if os.path.getsize(video_file) <= MAX_PREVIEW_BYTES:
                    st.video(video_file)

with col2:
    st.title("Upload video here 📁")
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file is not None:
        st.video(uploaded_file)
        if st.button("Process Uploaded Video", key="process_uploaded_video_button"):
            with tempfile.TemporaryDirectory() as folder:
                video_path = os.path.join(folder, uploaded_file.name)
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                process_video(video_path)
