import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from ntscraper import Nitter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="🧠",
    layout="centered"
)

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#1e1b4b,#312e81,#020617);
}

.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    letter-spacing: 1px;
    color: #e5e7eb;
}

.sub-title {
    text-align: center;
    font-size: 16px;
    color: #c7d2fe;
    margin-bottom: 30px;
}

.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 20px 50px rgba(0,0,0,0.6);
    margin: 40px auto;
}

textarea, input {
    border-radius: 12px !important;
    font-size: 15px !important;
}

button {
    background: linear-gradient(90deg,#6366f1,#a855f7) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
}

.footer {
    text-align: center;
    margin-top: 40px;
    font-size: 14px;
    color: #c7d2fe;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_resources():
    nltk.download("stopwords")

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    stop_words = set(stopwords.words("english"))
    return stop_words, model, vectorizer

# ---------------- LOAD TWITTER SCRAPER ----------------
@st.cache_resource
def load_scraper():
    return Nitter(log_level=1, skip_instance_check=True)

stop_words, model, vectorizer = load_resources()

# ---------------- PREDICTION FUNCTION ----------------
def predict_sentiment(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = ' '.join([word for word in text if word not in stop_words])
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return "Positive" if prediction == 1 else "Negative"

# ---------------- UI ----------------
st.markdown("<div class='main-title'>🧠 Sentiment Analysis System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Machine Learning based Text & Twitter Sentiment Classification</div>",
    unsafe_allow_html=True
)

st.markdown("<div class='card'>", unsafe_allow_html=True)

mode = st.radio(
    "Select Input Method",
    ["Analyze Custom Text", "Analyze Twitter User"]
)

# ---------- MODE 1: CUSTOM TEXT ----------
if mode == "Analyze Custom Text":
    user_text = st.text_area(
        "Input Text",
        placeholder="Enter a sentence or tweet for sentiment prediction...",
        label_visibility="collapsed"
    )

    if st.button("Predict Emotion"):
        if user_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            result = predict_sentiment(user_text)
            if result == "Positive":
                st.success("Prediction Result: Positive 😊")
            else:
                st.error("Prediction Result: Negative 😞")

# ---------- MODE 2: TWITTER USER ----------
elif mode == "Analyze Twitter User":
    scraper = load_scraper()
    username = st.text_input(
        "Twitter Username",
        placeholder="Enter username without @",
        label_visibility="collapsed"
    )

    if st.button("Fetch & Analyze Tweets"):
        if username.strip() == "":
            st.warning("Please enter a Twitter username.")
        else:
            try:
                tweets = scraper.get_tweets(username, mode="user", number=5)

                if "tweets" not in tweets or len(tweets["tweets"]) == 0:
                    st.warning("No tweets found or account may be private.")
                else:
                    for tweet in tweets["tweets"]:
                        text = tweet["text"]
                        sentiment = predict_sentiment(text)

                        if sentiment == "Positive":
                            st.success(text)
                        else:
                            st.error(text)

            except Exception:
                st.error(
                    "Twitter data could not be fetched at the moment. "
                    "Please try again later or use custom text mode."
                )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='footer'>Academic Project • CSE (AI) • Institution of Engineering and Management</div>",
    unsafe_allow_html=True
)
