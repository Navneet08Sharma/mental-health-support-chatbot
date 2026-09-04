import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. Load cleaned dataset
# ==========================================

file_path = "data/processed/sentiment_cleaned.csv"

df = pd.read_csv(file_path)

print("Dataset loaded.")
print("Number of rows:", len(df))


# ==========================================
# 2. Remove empty text
# ==========================================

df = df.dropna(subset=["clean_text"])

print("Rows after removing empty text:", len(df))


# ==========================================
# 3. Separate input and target
# ==========================================

X = df["clean_text"]
y = df["sentiment"]


# ==========================================
# 4. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining examples:", len(X_train))
print("Testing examples:", len(X_test))


# ==========================================
# 5. TF-IDF
# ==========================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# ==========================================
# 6. Logistic Regression
# ==========================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

# ==========================================
# 6.5 Save model and vectorizer
# ==========================================

joblib.dump(
    model,
    "models/sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("\nModel saved successfully.")
print("models/sentiment_model.pkl")
print("models/tfidf_vectorizer.pkl")

# ==========================================
# 7. Predictions
# ==========================================

print("\nMaking predictions...")

y_pred = model.predict(X_test_tfidf)


# ==========================================
# 8. Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print("BASELINE MODEL RESULTS")
print("==========================================")

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))