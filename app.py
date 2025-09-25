# app.py
import streamlit as st
import pickle
import json
import os
import string
import nltk
from nltk.corpus import stopwords

# -----------------------------
# Ensure NLTK data is available
# -----------------------------
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
try:
    stopwords.words('english')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)

# -----------------------------
# Paths to model files
# -----------------------------
MODEL_PATH = os.path.join("models", "emotion_model.pkl")
VECT_PATH = os.path.join("models", "vectorizer.pkl")
LABEL_PATH = os.path.join("models", "label_map.json")


# -----------------------------
# Load model, vectorizer, label map
# -----------------------------
@st.cache_resource
def load_artifacts():
    # quick existence checks
    for p in (MODEL_PATH, VECT_PATH, LABEL_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required artifact not found: {p}")

    # read first bytes to detect common corruption (text/BOM) that breaks pickle
    def _head(path, n=4):
        with open(path, "rb") as fh:
            return fh.read(n)

    m_head = _head(MODEL_PATH)
    v_head = _head(VECT_PATH)
    # UTF-8 BOM begins with EF BB BF
    if m_head.startswith(b"\xef\xbb\xbf") or v_head.startswith(b"\xef\xbb\xbf"):
        raise RuntimeError(
            "One of the .pkl files appears to be saved as text (starts with UTF-8 BOM).\n"
            f"Model head: {m_head.hex()}  Vectorizer head: {v_head.hex()}\n"
            "This commonly happens if the files were opened/saved in text mode or were corrupted by git line-ending conversions.\n"
            "Recreate and save the pickles using binary mode (open(..., 'wb')) and add a .gitattributes entry to keep the files binary."
        )

    # safe to load
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECT_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}
    return model, vectorizer, label_map


model, vectorizer, label_map = load_artifacts()

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = "".join([c for c in text if not c.isdigit()])
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="NLP Emotion Analyzer", layout="centered")
st.title("Emotion Analyzer")
st.write("Type a sentence and click Predict.")

user_input = st.text_area("Enter text:", height=150)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Preprocess input
        cleaned = preprocess(user_input)
        X = vectorizer.transform([cleaned])

        # Predict
        pred_num = model.predict(X)[0]  # numeric class
        pred_emotion = label_map.get(int(pred_num), str(pred_num))
        st.success(f"Predicted emotion: **{pred_emotion}**")

        # Show confidence if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            top_idx = probs.argmax()
            st.write(f"Confidence: {probs[top_idx]*100:.1f}%")

            # Show table of all probabilities
            prob_dict = {}
            for cls_idx, cls in enumerate(model.classes_):
                label = label_map.get(int(cls), str(cls))
                prob_dict[label] = f"{probs[cls_idx]*100:.1f}%"
            st.table(prob_dict)
