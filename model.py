
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
dataset = pd.read_csv('ticket_data_2.csv', on_bad_lines='skip')  # Skip bad lines to avoid errors

# Handle missing values
dataset['Ticket Description'] = dataset['summary'].fillna('')
dataset.dropna(subset=['product'], inplace=True)

# Initialize preprocessing utilities
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower()  # Remove non-alphabetic characters and convert to lowercase
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(tokens)

# Apply text cleaning
dataset['cleaned_description'] = dataset['Ticket Description'].apply(preprocess_text)

# Features and labels
X = dataset['cleaned_description']
y = dataset['product']

# Check dataset consistency
print(f'Dataset size: {len(X)} samples')
print(f'Missing values in cleaned features: {X.isnull().sum()}')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1200, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Model training
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Predictions and evaluation
y_pred = classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the vectorizer and model
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(classifier, 'naive_bayes_model.pkl')
print("Model and vectorizer saved!")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Analyze misclassifications
wrong_predictions = pd.DataFrame({
    'Ticket Description': X_test.reset_index(drop=True),
    'Actual Label': y_test.reset_index(drop=True),
    'Predicted Label': y_pred
})

# Filter only rows where predictions are incorrect
wrong_predictions = wrong_predictions[wrong_predictions['Actual Label'] != wrong_predictions['Predicted Label']]

print("\nMisclassified Samples:")
print(wrong_predictions.head())  # Display the first few misclassified samples

# Save wrong predictions for further analysis (optional)
wrong_predictions.to_csv('misclassified_samples.csv', index=False)

