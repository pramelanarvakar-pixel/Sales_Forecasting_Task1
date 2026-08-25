import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample Support Tickets Data
data = {
    'ticket': [
        'My payment failed but money deducted',
        'Unable to login to my account',
        'App is crashing on startup',
        'I want refund for my subscription',
        'How to change my password?',
        'Server is down urgent help needed',
        'Feature request: dark mode please',
        'Invoice not generated for last month',
        'Critical bug payment gateway not working',
        'Need help with account verification',
        'My order is delayed',
        'Application is very slow',
        'Please add paypal payment option',
        'I was charged twice urgent',
        'Cannot reset password link not working'
    ],
    'category': [
        'Billing', 'Account', 'Technical', 'Billing', 'Account',
        'Technical', 'Feature Request', 'Billing', 'Technical', 'Account',
        'Technical', 'Technical', 'Feature Request', 'Billing', 'Account'
    ]
}

df = pd.DataFrame(data)

# Text Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['clean'] = df['ticket'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['clean'])

# Model Training
model = LogisticRegression()
model.fit(X, df['category'])

# Priority Logic
def get_priority(text):
    text = text.lower()
    if any(w in text for w in ['urgent', 'critical', 'failed', 'down', 'twice', 'crashing']):
        return 'High'
    elif any(w in text for w in ['request', 'please', 'how to', 'add']):
        return 'Low'
    else:
        return 'Medium'

# Final Prediction Function
def predict_ticket(new_ticket):
    clean = clean_text(new_ticket)
    vec = vectorizer.transform([clean])
    category = model.predict(vec)[0]
    priority = get_priority(new_ticket)
    return category, priority

# Testing
tests = [
    "Payment deducted twice urgent refund",
    "Please add dark mode feature",
    "Login not working cannot access account"
]

for t in tests:
    cat, pri = predict_ticket(t)
    print(f"Ticket: {t} -> Category: {cat}, Priority: {pri}")
