Step 1: Clone the repository

Step 2: python3 -m venv .venv (bash)

Step 3: source .venv/bin/activate(bash)

step 4: pip install -r requirements.txt(bash)

step 5: streamlit run app.py

note- ffmpeg and tesseract must be installed locally (packages.txt covers Streamlit Cloud)

Demo Link: https://drive.google.com/file/d/1O9yD7aDEMRdZalXr5yjoyfnMwz3rgelU/view?usp=sharing

Links: any site yt-dlp supports works, including YouTube, Instagram, TikTok, X, Facebook and Reddit.

Cookies: for login-only sites, export a Netscape cookies file and set it as a root-level COOKIES secret in Streamlit (App settings > Secrets), or as a COOKIES env var locally. A file can hold cookies for several sites. Never commit it.

COOKIES = '''
paste cookies file here
'''
