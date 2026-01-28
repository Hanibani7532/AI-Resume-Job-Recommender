from flask import Flask, render_template, request
import pdfplumber
import pandas as pd
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# uploads folder ensure
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- helper functions ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_cv_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

# ---------- load jobs ----------
jobs = pd.read_csv("model/jobs.csv")   # ✅ correct path
jobs = jobs[['Job Title', 'Job Description']]
jobs.columns = ['job_title', 'job_description']
jobs.dropna(inplace=True)

jobs['clean_description'] = jobs['job_description'].apply(clean_text)

# ---------- TF-IDF ----------
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
job_vectors = tfidf.fit_transform(jobs['clean_description'])

# ---------- routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        file = request.files["cv"]

        if file.filename != "":
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            cv_text = extract_cv_text(file_path)
            cv_clean = clean_text(cv_text)
            cv_vector = tfidf.transform([cv_clean])

            scores = cosine_similarity(cv_vector, job_vectors)[0]
            top_indices = scores.argsort()[-5:][::-1]

            results = []
            for idx in top_indices:
                results.append({
                    "title": jobs.iloc[idx]['job_title'],
                    "score": round(scores[idx], 2)
                })

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
