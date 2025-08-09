import streamlit as st
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# Load sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

model = load_model()

# Load translations
def load_translations(file_path="data/translations.json"):
    if not os.path.exists(file_path):
        st.error("Translation file not found.")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

translations = load_translations()

# Build a dict for fast lookup
japanese_to_entry = {entry["japanese"]: entry for entry in translations}

# Load or initialize submissions
SUBMISSION_FILE = "submissions.json"
if os.path.exists(SUBMISSION_FILE):
    with open(SUBMISSION_FILE, "r", encoding="utf-8") as f:
        try:
            submissions = json.load(f)
        except json.JSONDecodeError:
            submissions = []
else:
    submissions = []

# Save submissions helper
def save_submissions():
    with open(SUBMISSION_FILE, "w", encoding="utf-8") as f:
        json.dump(submissions, f, indent=2, ensure_ascii=False)

st.title("🧠 Japanese to English Translation Helper")

if not japanese_to_entry:
    st.warning("No translation data found.")
    st.stop()

selected_japanese = st.selectbox("Select a Japanese sentence:", list(japanese_to_entry.keys()))
entry = japanese_to_entry[selected_japanese]

st.markdown("### 📘 Japanese Sentence")
st.write(selected_japanese)

# Optional: show correct English translation or hide it if you want
if st.checkbox("Show correct English translation"):
    st.markdown("### ✔️ Correct Translation")
    st.write(entry["english"])
    if "alternatives" in entry:
        st.markdown("### 🔄 Alternatives")
        for alt in entry["alternatives"]:
            st.write(alt)

st.markdown("### ✍️ Try your English translation")

user_input = st.text_input("Type your translation here:")

if user_input:
    # Calculate similarity live (optional)
    all_variants = [entry["english"]] + entry.get("alternatives", [])
    embeddings = model.encode([user_input] + all_variants, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(embeddings[0], embeddings[1:])
    best_score = scores.max().item()
    st.write(f"Similarity score (live): {best_score:.2f}")

    if best_score > 0.8:
        st.success("✅ Pretty close!")
    elif best_score > 0.6:
        st.info("🧐 Somewhat similar.")
    else:
        st.warning("❌ Quite different. Keep trying!")

    # Submit button
    if st.button("Submit this translation"):
        submission = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "japanese": selected_japanese,
            "student_translation": user_input,
            "similarity_score": best_score,
        }
        submissions.append(submission)
        save_submissions()
        st.success("🎉 Your translation has been submitted!")

        # Clear the input box after submission
        st.rerun()

else:
    st.info("Type a translation above to see similarity scores and submit.")

