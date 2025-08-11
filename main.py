import streamlit as st
import json
import os
from sentence_transformers import SentenceTransformer, util
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import difflib

# --- Settings ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SPREADSHEET_ID = "1_fy_83CUtcfT7iXSSC2aPNS7sJOWONyTPs0h9aFUzmc"
THRESHOLD = 0.80  # Passing score

# --- Google Sheets setup ---
@st.cache_resource
def get_gsheet():
    if "credentials.json" in os.listdir():
        # Local mode: load creds from file
        creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
    elif st.secrets.get("gcp_service_account"):
        # Deployed mode: load creds from Streamlit secrets
        creds_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(json.loads(creds_info), scopes=SCOPES)
    else:
        st.warning("Google credentials not configured.")
        return None  # Graceful fallback

    client = gspread.authorize(creds)
    sheet = client.open_by_key(SPREADSHEET_ID).sheet1
    return sheet

sheet = get_gsheet()

# --- Load sentence transformer model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

model = load_model()

# --- Load translations ---
def load_translations(file_path="data/translations.json"):
    if not os.path.exists(file_path):
        st.error("Translation file not found.")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

translations = load_translations()
japanese_to_entry = {entry["japanese"]: entry for entry in translations}

# --- Helper functions ---
def compute_score_and_best(user_text, variants):
    embeddings = model.encode([user_text] + variants, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(embeddings[0], embeddings[1:])
    best_idx = scores.argmax().item()
    best_score = scores.max().item()
    return best_score, variants[best_idx]

def highlight_diff(user_text, best_variant):
    """Highlight differences using difflib."""
    diff = difflib.ndiff(user_text.split(), best_variant.split())
    highlighted = []
    for token in diff:
        if token.startswith("+ "):  # in best_variant but not in user_text
            highlighted.append(f"<span style='background-color:#b3ffb3'> {token[2:]} </span>")
        elif token.startswith("- "):  # in user_text but not in best_variant
            highlighted.append(f"<span style='background-color:#ffcccc'> {token[2:]} </span>")
        elif token.startswith("  "):
            highlighted.append(token[2:])
    return " ".join(highlighted)

# --- Streamlit UI ---
st.title("🧠 Japanese to English Translation Helper")

if not japanese_to_entry:
    st.warning("No translation data found.")
    st.stop()

selected_japanese = st.selectbox("Select a Japanese sentence:", list(japanese_to_entry.keys()))
entry = japanese_to_entry[selected_japanese]

show_translation = st.checkbox("Show correct English translation")

if show_translation:
    st.markdown("### 📘 Translation")
    st.write(entry["english"])
    if "alternatives" in entry:
        st.markdown("### 🔄 Alternatives")
        for alt in entry["alternatives"]:
            st.write(alt)

st.markdown("### ✍️ Enter your English translation")
user_input = st.text_input("Your answer:")

# Keep last score in session
if "last_score" not in st.session_state:
    st.session_state.last_score = None
if "last_variant" not in st.session_state:
    st.session_state.last_variant = None

# Try Translation
if st.button("🔍 Try Translation"):
    if user_input.strip() == "":
        st.warning("Please enter a translation before trying.")
    else:
        all_variants = [entry["english"]] + entry.get("alternatives", [])
        best_score, best_variant = compute_score_and_best(user_input, all_variants)
        st.session_state.last_score = best_score
        st.session_state.last_variant = best_variant

        if best_score >= THRESHOLD:
            st.success(f"✅ Good enough! Score: {best_score:.2f}")
        else:
            st.warning(f"⚠️ Not quite there yet. Score: {best_score:.2f} (Try to improve)")

        # Show differences
        diff_html = highlight_diff(user_input, best_variant)
        st.markdown("**Closest matching correct version:**")
        st.write(best_variant)
        st.markdown("**Differences (green = missing words, red = extra words):**", unsafe_allow_html=True)
        st.markdown(diff_html, unsafe_allow_html=True)

# Submit Translation
if st.button("✅ Submit this translation"):
    if user_input.strip() == "":
        st.warning("Please enter a translation before submitting.")
    else:
        if st.session_state.last_score is None:
            all_variants = [entry["english"]] + entry.get("alternatives", [])
            best_score, best_variant = compute_score_and_best(user_input, all_variants)
        else:
            best_score = st.session_state.last_score
            best_variant = st.session_state.last_variant

        if sheet is None:
            st.error("Google Sheet not available. Submission failed.")
        else:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sheet.append_row([
                    timestamp,
                    selected_japanese,
                    user_input,
                    f"{best_score:.4f}",
                    "Pass" if best_score >= THRESHOLD else "Fail"
                ])
                if best_score >= THRESHOLD:
                    st.success(f"✅ Submitted! Score: {best_score:.2f} (Pass)")
                else:
                    st.warning(f"✅ Submitted! Score: {best_score:.2f} (Below threshold)")
                st.session_state.last_score = None
                st.session_state.last_variant = None
            except Exception as e:
                st.error(f"Failed to save your submission: {e}")
