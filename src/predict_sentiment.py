import re
import joblib


# ==========================================
# 1. Load saved model
# ==========================================

model = joblib.load(
    "models/sentiment_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# ==========================================
# 2. Text cleaning function
# ==========================================

def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove @username
    text = re.sub(r"@\w+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# 3. Prediction function
# ==========================================

def predict_sentiment(text):

    cleaned_text = clean_text(text)

    # Convert text into TF-IDF numbers
    text_vector = vectorizer.transform([cleaned_text])

    # Make prediction
    prediction = model.predict(text_vector)[0]

    # Get probability
    probabilities = model.predict_proba(text_vector)[0]

    confidence = max(probabilities)

    if prediction == 1:
        sentiment = "POSITIVE"
    else:
        sentiment = "NEGATIVE"

    return sentiment, confidence


# ==========================================
# 4. Test the model
# ==========================================

print("======================================")
print("AI SENTIMENT PREDICTOR")
print("======================================")

while True:

    text = input("\nEnter a sentence (or type 'exit'): ")

    if text.lower() == "exit":
        print("\nExiting...")
        break

    sentiment, confidence = predict_sentiment(text)

    print("\nPrediction:", sentiment)
    print("Confidence:", round(confidence * 100, 2), "%")