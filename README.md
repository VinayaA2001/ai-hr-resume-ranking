AI HR Resume Ranking

A web-based application that helps HR professionals rank resumes against a job description using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The app extracts and cleans text from uploaded resumes (PDF/DOCX) and compares it with a job description to calculate a match score.

🔹 Features

Upload multiple resumes in PDF or DOCX format.

Input a job description to rank candidates.

Automatic text extraction, cleaning, lemmatization, and tokenization using NLTK.

Calculates TF-IDF similarity between the job description and resumes.

Displays a match score in a table and a bar chart visualization.

Download the results as a CSV file.

🔹 Technologies Used

Python 3.13

Streamlit – Web app framework

NLTK – Natural Language Toolkit for text preprocessing

PyPDF2 – PDF text extraction

python-docx – DOCX text extraction

scikit-learn – TF-IDF vectorization & cosine similarity

pandas – Data manipulation

matplotlib – Data visualization

🔹 How It Works

Upload Resumes – Upload one or more resumes in PDF or DOCX format.

Paste Job Description – Provide the job description text.

Rank Resumes – Click the “Rank Resumes” button.

View Results – The app calculates the match score, displays a table, and visualizes results with a bar chart.

Download CSV – Download the match results as a CSV for record-keeping.
