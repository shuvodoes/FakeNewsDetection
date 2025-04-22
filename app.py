import streamlit as st
import numpy as np
from joblib import load
import re

# Load model and vectorizer
model = load('logistic_model.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Custom CSS for a smooth, modern design
st.markdown("""
    <style>
    body {
        background: #f5f7fa;
        font-family: 'Helvetica', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #89f7fe, #66a6ff);
        padding: 20px;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        text-shadow: 1px 1px 5px rgba(0, 0, 0, 0.2);
        font-size: 2.5em;
    }
    h5 {
        color: #34495e;
        text-align: center;
        font-size: 1.2em;
        margin-top: 10px;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        color: #333;
        font-size: 16px;
        border-radius: 15px;
        padding: 15px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        width: 100%;
        transition: border-color 0.3s ease-in-out;
    }
    .stTextArea textarea:focus {
        border-color: #66a6ff;
        outline: none;
    }
    .stButton button {
        background-color: #66a6ff;
        color: white;
        font-size: 18px;
        padding: 12px 30px;
        border-radius: 30px;
        border: none;
        box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1);
        cursor: pointer;
        transition: background-color 0.3s ease-in-out;
    }
    .stButton button:hover {
        background-color: #4e86c1;
        color: white;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-left: 5px solid #28a745;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border-left: 5px solid #dc3545;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>📰 Fake News Detection App</h1>", unsafe_allow_html=True)
st.markdown("<h5>Instantly check if a news article is Real or Fake!</h5>", unsafe_allow_html=True)
st.markdown("---")

# Layout
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    user_input = st.text_area("📝 Enter news content (title + body):", height=250)

    if st.button("🔍 Detect"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]

            if prediction == 1:
                st.success("✅ This news is **Real**.")
                st.balloons()
            else:
                st.error("❌ This news is **Fake**.")