import streamlit as st
import PyPDF2
import docx
import string
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# NLTK Setup
# ---------------------------
import nltk

# Download only if missing
nltk_packages = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
for pkg in nltk_packages:
    try:
        if pkg in ["punkt", "punkt_tab"]:
            nltk.data.find(f"tokenizers/{pkg}")
        else:
            nltk.data.find(f"corpora/{pkg}")
    except:
        nltk.download(pkg)


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------
# Helper Functions
# ---------------------------
lemmatizer = WordNetLemmatizer()

def extract_text_from_pdf(file):
    """Extract text from uploaded PDF file."""
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    except:
        st.warning(f"⚠️ Could not read {file.name}")
    return text

def extract_text_from_docx(file):
    """Extract text from uploaded DOCX file."""
    doc = docx.Document(file)
    text = [para.text for para in doc.paragraphs]
    return "\n".join(text)

def clean_text(text):
    """Lowercase, remove punctuation, stopwords, lemmatize, and tokenize."""
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="AI HR Resume Ranking", layout="wide")
st.title("🤖 AI HR Resume Ranking App")
st.write("Upload resumes and a job description to see which candidates are the best match!")

# Step 1: Job Description input
job_description = st.text_area("✍️ Paste Job Description here:")

# Step 2: Resume Upload
uploaded_files = st.file_uploader(
    "📂 Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True
)

# Step 3: Rank Resumes
if st.button("Rank Resumes"):
    if not job_description:
        st.warning("⚠️ Please enter a job description.")
    elif not uploaded_files:
        st.warning("⚠️ Please upload at least one resume.")
    else:
        resume_texts = []
        resume_names = []

        for file in uploaded_files:
            if file.name.endswith(".pdf"):
                text = extract_text_from_pdf(file)
            elif file.name.endswith(".docx"):
                text = extract_text_from_docx(file)
            else:
                continue

            if not text.strip():
                st.warning(f"⚠️ No text extracted from {file.name}")
                continue

            cleaned = clean_text(text)
            resume_texts.append(cleaned)
            resume_names.append(file.name)

        # Clean Job Description
        job_description_clean = clean_text(job_description)

        # TF-IDF Vectorization & Cosine Similarity
        documents = [job_description_clean] + resume_texts
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Results DataFrame
        results = pd.DataFrame({
            "Resume": resume_names,
            "Match Score (%)": (similarity_scores * 100).round(2)
        }).sort_values(by="Match Score (%)", ascending=False)

        # Display Table
        st.subheader("📊 Resume Ranking Results")
        st.dataframe(results)

        # Display Bar Chart
        st.subheader("📈 Match Score Chart")
        fig, ax = plt.subplots()
        ax.barh(results["Resume"], results["Match Score (%)"], color='skyblue')
        ax.set_xlabel("Match Score (%)")
        ax.set_ylabel("Resume")
        ax.invert_yaxis()  # Highest score at top
        st.pyplot(fig)

        # Download CSV
        st.download_button(
            label="📥 Download Results as CSV",
            data=results.to_csv(index=False),
            file_name="resume_ranking.csv",
            mime="text/csv"
        )
