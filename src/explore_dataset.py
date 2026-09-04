import pandas as pd

# Location of our original dataset
file_path = "data/raw/training.1600000.processed.noemoticon.csv"

# Column names from the Sentiment140 dataset
column_names = [
    "sentiment",
    "id",
    "date",
    "query",
    "user",
    "text"
]

# Load the dataset
df = pd.read_csv(
    file_path,
    encoding="latin-1",
    header=None,
    names=column_names
)

print("==========================================")
print("SENTIMENT140 DATASET")
print("==========================================")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nSample negative tweets:")
print(df[df["sentiment"] == 0]["text"].head(5).to_string(index=False))

print("\nSample positive tweets:")
print(df[df["sentiment"] == 4]["text"].head(5).to_string(index=False))