import re
import pandas as pd


# ---------------------------------------
# 1. Text cleaning function
# ---------------------------------------

def clean_text(text):
    """
    Clean tweet text while preserving important words.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove @username mentions
    text = re.sub(r"@\w+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------
# 2. Load processed dataset
# ---------------------------------------

input_file = "data/processed/sentiment_sample.csv"

df = pd.read_csv(input_file)

print("Dataset loaded.")
print("Number of rows:", len(df))


# ---------------------------------------
# 3. Clean the text
# ---------------------------------------

print("\nCleaning text...")

df["clean_text"] = df["text"].apply(clean_text)


# ---------------------------------------
# 4. Show examples
# ---------------------------------------

print("\n========================================")
print("CLEANING EXAMPLES")
print("========================================")

for i in range(5):

    print("\nOriginal:")
    print(df.iloc[i]["text"])

    print("\nCleaned:")
    print(df.iloc[i]["clean_text"])

    print("-" * 50)


# ---------------------------------------
# 5. Save cleaned dataset
# ---------------------------------------

output_file = "data/processed/sentiment_cleaned.csv"

df.to_csv(output_file, index=False)

print("\n========================================")
print("CLEANING COMPLETE")
print("========================================")

print("Rows:", len(df))
print("Saved to:", output_file)