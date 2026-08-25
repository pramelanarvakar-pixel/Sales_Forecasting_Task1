import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Job Description
job_description = """
We are hiring a Data Scientist. Required skills: Python, Machine Learning,
Pandas, Scikit-learn, NLP, SQL, Data Visualization
"""

# Sample Resumes
resumes = {
    'Amit Sharma': 'Experienced Data Scientist with Python, Machine Learning, Pandas, SQL, Deep Learning',
    'Priya Singh': 'Software Developer skilled in Java, C++, HTML, CSS, JavaScript',
    'Rahul Verma': 'Data Analyst with Python, Pandas, SQL, Excel, Data Visualization, Power BI',
    'Neha Patel': 'ML Engineer with Python, Scikit-learn, NLP, Tensorflow, Machine Learning, SQL',
    'Vikram Rao': 'Fresher with knowledge of Python basics and MS Office'
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

required_skills = ['python', 'machine learning', 'pandas', 'scikit-learn', 'nlp', 'sql', 'data visualization']

def extract_skills(text):
    text = text.lower()
    found = []
    missing = []
    for skill in required_skills:
        if skill in text:
            found.append(skill)
        else:
            missing.append(skill)
    return found, missing

cleaned_job = clean_text(job_description)
cleaned_resumes = [clean_text(r) for r in resumes.values()]
all_texts = [cleaned_job] + cleaned_resumes

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(all_texts)

job_vector = tfidf_matrix[0]
resume_vectors = tfidf_matrix[1:]
scores = cosine_similarity(job_vector, resume_vectors)[0]

results = []
for i, (name, resume_text) in enumerate(resumes.items()):
    found, missing = extract_skills(resume_text)
    score = round(scores[i] * 100, 2)
    results.append({
        'Candidate': name,
        'Match Score %': score,
        'Found Skills': ', '.join(found),
        'Missing Skills': ', '.join(missing) if missing else 'None'
    })

df = pd.DataFrame(results)
df = df.sort_values(by='Match Score %', ascending=False)

print("=== Ranked Candidates ===")
print(df.to_string(index=False))
print(f"\nTop Candidate: {df.iloc[0]['Candidate']} with {df.iloc[0]['Match Score %']}% match")
