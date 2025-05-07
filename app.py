import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"
import streamlit as st
import subprocess
import ssl
import hashlib
import pytesseract
import cv2
from faster_whisper import WhisperModel
from transformers import pipeline
import tempfile
import moviepy as mp

ssl._create_default_https_context = ssl._create_unverified_context

def initialize_session_state():
    if 'reel_id' not in st.session_state:
        st.session_state.reel_id = None
initialize_session_state()

def clear_previous_files(video_filename=None, audio_filename=None):
    if video_filename and os.path.exists(video_filename):
        os.remove(video_filename)
    if audio_filename and os.path.exists(audio_filename):
        os.remove(audio_filename)

st.markdown("<br><br><br>", unsafe_allow_html=True)
col1, spacer, col2 = st.columns([2, 2, 3])

def download_video(link):
    try:
        video_hash = hashlib.md5(link.encode()).hexdigest()[:8]
        filename = f"input_video_{video_hash}.mp4"
        subprocess.run([
            "yt-dlp",
            "--cookies", "cookies.txt",
            "-o", filename,
            link
        ], check=True)
        return filename
    except Exception as e:
        st.error(f"Download failed: {e}")
        return None

def transcribe_audio(path):
    try:
        clip = mp.VideoFileClip(path)
        clip.audio.write_audiofile("audio.wav")
        model = WhisperModel("tiny", compute_type="int8")  
        segments, _ = model.transcribe("audio.wav")
        text = " ".join([segment.text for segment in segments])
        return text
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return ""

def extract_text_from_frames(path):
    cap = cv2.VideoCapture(path)
    texts = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > 300: 
            break
        if frame_count % 30 == 0:  
            text = pytesseract.image_to_string(frame)
            if text.strip():
                texts.append(text.strip())
        frame_count += 1

    cap.release()
    return " ".join(texts)

def summarize_text(text):
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        summary = ""
        for chunk in chunks:
            result = summarizer(chunk, max_length=60, min_length=15, do_sample=False)
            summary += result[0]['summary_text'] + " "
        return summary.strip()
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
    video_link = st.text_input("Paste a YouTube or Instagram video link")
    if st.button("Process Link", key="process_link_button"):
        video_file = download_video(video_link)
        if video_file:
            process_video(video_file)
            st.video(video_file)
            clear_previous_files(video_filename=video_file)

with col2:
    st.title("Upload video here 📁")
    uploaded_file = st.file_uploader("Upload a short video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file is not None:
        st.video(uploaded_file)
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        if st.button("Process Uploaded Video", key="process_uploaded_video_button"):
            process_video("uploaded_video.mp4")
            clear_previous_files(video_filename="uploaded_video.mp4")
