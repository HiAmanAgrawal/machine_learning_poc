from flask import Flask, request, jsonify
import pandas as pd
import re
import nltk
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score

# Flask App
app = Flask(__name__)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing utilities
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load pre-trained components
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Saved vectorizer
model = joblib.load('naive_bayes_model.pkl')  # Saved classifier


# Preprocess text function
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)


# Route to check API health
@app.route('/')
def home():
    return jsonify({'message': 'Ticket Classifier API is running!'})


# Single ticket prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'ticket_description' not in data:
        return jsonify({'error': 'Missing ticket_description field'}), 400

    ticket_description = data['ticket_description']
    cleaned_text = preprocess_text(ticket_description)
    vectorized_text = vectorizer.transform([cleaned_text]).toarray()
    prediction = model.predict(vectorized_text)
    probability = np.max(model.predict_proba(vectorized_text)) * 100

    return jsonify({
        'ticket_description': ticket_description,
        'predicted_product': prediction[0],
        'probability': f'{probability:.2f}%'
    })


# Batch ticket prediction
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    data = request.json
    if 'tickets' not in data:
        return jsonify({'error': 'Missing tickets field'}), 400

    descriptions = [ticket['description'] for ticket in data['tickets']]
    actual_labels = [ticket['product'] for ticket in data['tickets']]
    cleaned_texts = [preprocess_text(desc) for desc in descriptions]
    vectorized_texts = vectorizer.transform(cleaned_texts).toarray()
    predictions = model.predict(vectorized_texts)
    probabilities = np.max(model.predict_proba(vectorized_texts), axis=1) * 100

    accuracy = accuracy_score(actual_labels, predictions) * 100
    results = []

    for i in range(len(descriptions)):
        results.append({
            'description': descriptions[i],
            'actual_product': actual_labels[i],
            'predicted_product': predictions[i],
            'probability': f'{probabilities[i]:.2f}%'
        })

    return jsonify({
        'accuracy': f'{accuracy:.2f}%',
        'data': results
    })


if __name__ == '__main__':
    app.run(debug=True)
