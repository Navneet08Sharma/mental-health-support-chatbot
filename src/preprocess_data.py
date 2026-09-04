import pandas as pd

# -----------------------------
# 1. File locations
# -----------------------------

input_file = "data/raw/training.1600000.processed.noemoticon.csv"
output_file = "data/processed/sentiment_sample.csv"


# -----------------------------
# 2. Column names
# -----------------------------

column_names = [
    "sentiment",
    "id",
    "date",
    "query",
    "user",
    "text"
]


# -----------------------------
# 3. Load dataset
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(
    input_file,
    encoding="latin-1",
    header=None,
    names=column_names
)


# -----------------------------
# 4. Keep only useful columns
# -----------------------------

df = df[["sentiment", "text"]]


# -----------------------------
# 5. Take 100,000 examples
# -----------------------------

df = df.sample(
    n=100000,
    random_state=42
)


# -----------------------------
# 6. Convert sentiment labels
# -----------------------------

df["sentiment"] = df["sentiment"].map({
    0: 0,
    4: 1
})


# -----------------------------
# 7. Save processed dataset
# -----------------------------

df.to_csv(
    output_file,
    index=False
)


# -----------------------------
# 8. Display information
# -----------------------------

print("\n================================")
print("PREPROCESSING COMPLETE")
print("================================")

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nSaved to:")
print(output_file)