# app.py
import streamlit as st

# -----------------------------
# Safe imports & helpful error display
# -----------------------------
try:
    import PyPDF2
    import docx
    import string
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import nltk
    import traceback
except Exception as e:
    st.set_page_config(page_title="AI HR Resume Ranking", layout="wide")
    st.title("🤖 AI HR Resume Ranking (Import error)")
    st.error("An import error occurred while loading the app. See details below.")
    st.exception(e)
    st.stop()

# -----------------------------
# Ensure NLTK resources
# -----------------------------
def ensure_nltk_resources():
    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for pkg, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)

ensure_nltk_resources()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# -----------------------------
# Helper functions
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    try:
        uploaded_file.seek(0)
        reader = PyPDF2.PdfReader(uploaded_file)
        text_pages = [p.extract_text() for p in reader.pages if p.extract_text()]
        return "\n".join(text_pages)
    except Exception as e:
        st.warning(f"Could not extract PDF text from {getattr(uploaded_file, 'name', '')}: {e}")
        return ""

def extract_text_from_docx(uploaded_file):
    try:
        uploaded_file.seek(0)
        doc = docx.Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs if p.text])
    except Exception as e:
        st.warning(f"Could not extract DOCX text from {getattr(uploaded_file, 'name', '')}: {e}")
        return ""

def clean_text(text):
    try:
        text = (text or "").lower()
        text = "".join([ch for ch in text if ch not in string.punctuation])
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and t.strip()]
        return " ".join(tokens)
    except Exception:
        return (text or "").lower()

# -----------------------------
# UI Setup
# -----------------------------
st.set_page_config(page_title="AI HR Resume Ranking", layout="wide")
st.markdown(
    """
    <style>
      .app-header {
        background: linear-gradient(90deg,#0f172a,#1f4e79);
        padding:18px;border-radius:8px;color:white;margin-bottom:16px;
      }
      .top-card {
        border-radius:8px;padding:12px;background:#e6fff0;border-left:6px solid #2ca02c;margin-bottom:10px;
      }
    </style>
    <div class="app-header">
      <h1 style="margin:0">🤖 AI HR Resume Ranking</h1>
      <div style="opacity:0.92">Upload resumes and match them against a job description using NLP & ML</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Layout: two columns
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📋 Job Description")
    job_description = st.text_area("Paste the job description here:", height=180)

    st.subheader("📂 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload multiple resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True
    )

    rank_clicked = st.button("🚀 Rank Resumes")

# -----------------------------
# Processing function
# -----------------------------
def process_and_display(job_desc_text, uploaded_files_list):
    try:
        resume_texts, resume_names = [], []

        for f in uploaded_files_list:
            try:
                f.seek(0)
            except:
                pass

            if f.name.lower().endswith(".pdf"):
                raw = extract_text_from_pdf(f)
            elif f.name.lower().endswith(".docx"):
                raw = extract_text_from_docx(f)
            else:
                raw = ""

            if not raw.strip():
                st.warning(f"No text extracted from {f.name}; skipping.")
                continue

            resume_names.append(f.name)
            resume_texts.append(clean_text(raw))

        if not resume_texts:
            st.warning("No resume text found. Please upload valid PDF/DOCX resumes.")
            return

        job_clean = clean_text(job_desc_text)

        # Vectorize and compute similarity
        documents = [job_clean] + resume_texts
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(documents)
        sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

        results = pd.DataFrame({
            "Resume": resume_names,
            "Match Score (%)": (sims * 100).round(2)
        }).sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)

        # Top match card
        top = results.iloc[0]
        st.markdown(
            f'<div class="top-card"><strong>Top Match:</strong> {top["Resume"]} — '
            f'<strong>{top["Match Score (%)"]}%</strong></div>',
            unsafe_allow_html=True,
        )

        # Show results table
        st.subheader("📊 Ranked Candidates")
        st.dataframe(results, use_container_width=True)

        # Bar chart
        st.subheader("📈 Match Score Chart")
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#2ca02c" if name == top["Resume"] else "#1f77b4" for name in results["Resume"]]
        bars = ax.bar(results["Resume"], results["Match Score (%)"], color=colors, edgecolor="black", alpha=0.9)
        ax.set_ylabel("Match Score (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Resume Match Scores")
        plt.xticks(rotation=30, ha="right")

        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

        st.pyplot(fig)

        # Download CSV
        csv_data = results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", data=csv_data, file_name="resume_ranking.csv", mime="text/csv")

    except Exception as e:
        st.error("An error occurred during processing. See details below.")
        st.exception(e)
        st.text("Traceback (for debugging):")
        st.text(traceback.format_exc())

# -----------------------------
# Trigger processing
# -----------------------------
if rank_clicked:
    if not job_description:
        st.warning("Please enter a job description before ranking.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        process_and_display(job_description, uploaded_files)

# Footer
st.markdown("---")
st.markdown("Built with Python • Streamlit • NLTK • scikit-learn")
