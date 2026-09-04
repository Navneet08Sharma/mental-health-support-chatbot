import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# 1. Load dataset
# ==========================================

file_path = "data/processed/sentiment_cleaned.csv"

df = pd.read_csv(file_path)

df = df.dropna(subset=["clean_text"])

print("Dataset loaded.")
print("Rows:", len(df))


# ==========================================
# 2. Separate input and target
# ==========================================

X = df["clean_text"]
y = df["sentiment"]


# ==========================================
# 3. Create the same test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Load saved model
# ==========================================

model = joblib.load(
    "models/sentiment_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

print("Saved model loaded successfully.")


# ==========================================
# 5. Convert test text into TF-IDF
# ==========================================

X_test_tfidf = vectorizer.transform(X_test)


# ==========================================
# 6. Make predictions
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# 7. Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("BASELINE MODEL EVALUATION")
print("======================================")

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")


# ==========================================
# 8. Classification report
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Negative", "Positive"]
    )
)


# ==========================================
# 9. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 10. Display confusion matrix
# ==========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

display.plot()

plt.title("Baseline Sentiment Model - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "reports/confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved to:")
print("reports/confusion_matrix.png")