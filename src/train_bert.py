import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/sentiment_cleaned.csv"
MODEL_NAME = "bert-base-uncased"

OUTPUT_DIR = "models/bert_sentiment_exp2"
REPORT_DIR = "reports"

MAX_LENGTH = 128

# Start with 2 epochs.
# We can increase this later if necessary.
NUM_EPOCHS = 2

# RTX 5050 should handle this, but we keep the batch size
# conservative to avoid CUDA out-of-memory errors.
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

SEED = 42


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 60)
print("BERT SENTIMENT TRAINING")
print("=" * 60)

print("\nChecking GPU...")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print("CUDA available: True")
    print("GPU:", device_name)
else:
    print("CUDA available: False")
    print("WARNING: Training will use CPU.")

print("PyTorch version:", torch.__version__)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Original rows:", len(df))

# Keep only the columns we need.
df = df[["clean_text", "sentiment"]]

# Remove missing text rows.
df = df.dropna(subset=["clean_text"])

# Convert text to string.
df["clean_text"] = df["clean_text"].astype(str)

# Convert labels to integers.
df["sentiment"] = df["sentiment"].astype(int)

print("Rows after removing missing text:", len(df))

print("\nClass distribution:")
print(df["sentiment"].value_counts())


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating train/validation split...")

train_df, val_df = train_test_split(
    df,
    test_size=0.10,
    random_state=SEED,
    stratify=df["sentiment"],
)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))


# ============================================================
# CONVERT TO HUGGING FACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df.reset_index(drop=True)
)

val_dataset = Dataset.from_pandas(
    val_df.reset_index(drop=True)
)


# ============================================================
# LOAD BERT TOKENIZER
# ============================================================

print("\nLoading BERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# TOKENIZATION
# ============================================================

print("Tokenizing training data...")


def tokenize_function(examples):
    return tokenizer(
        examples["clean_text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing training data",
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing validation data",
)


# Rename sentiment -> labels because this is what
# the BERT classification model expects.

train_dataset = train_dataset.rename_column(
    "sentiment",
    "labels",
)

val_dataset = val_dataset.rename_column(
    "sentiment",
    "labels",
)


# Keep only columns needed by the model.
columns_to_keep = [
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "labels",
]

# BERT normally has token_type_ids, but keeping the code
# flexible in case the tokenizer does not produce them.
train_columns = [
    c for c in columns_to_keep
    if c in train_dataset.column_names
]

val_columns = [
    c for c in columns_to_keep
    if c in val_dataset.column_names
]

train_dataset.set_format(
    type="torch",
    columns=train_columns,
)

val_dataset.set_format(
    type="torch",
    columns=val_columns,
)


# ============================================================
# LOAD BERT CLASSIFICATION MODEL
# ============================================================

print("\nLoading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_prediction):

    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids

    # Some versions return predictions as a tuple.
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predicted_labels = np.argmax(
        predictions,
        axis=-1,
    )

    accuracy = accuracy_score(
        labels,
        predicted_labels,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predicted_labels,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

use_fp16 = torch.cuda.is_available()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    gradient_accumulation_steps=2,

    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,


    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,

    logging_strategy="steps",
    logging_steps=100,

    save_total_limit=2,

    fp16=use_fp16,

    report_to="none",

    seed=SEED,

    dataloader_num_workers=0,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING BERT TRAINING")
print("=" * 60)

print("Model:", MODEL_NAME)
print("Epochs:", NUM_EPOCHS)
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Batch size:", TRAIN_BATCH_SIZE)
print("Gradient accumulation:", 2)
print("Maximum sequence length:", MAX_LENGTH)
print("FP16:", use_fp16)

print("\nTraining is starting...\n")

train_result = trainer.train()


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL BERT EVALUATION")
print("=" * 60)

evaluation = trainer.evaluate()

print("\nResults:")

for key, value in evaluation.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving BERT model...")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    REPORT_DIR,
    "bert_results_exp2.txt",
)

with open(results_path, "w", encoding="utf-8") as f:

    f.write("BERT SENTIMENT MODEL RESULTS\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Epochs: {NUM_EPOCHS}\n")
    f.write(f"Training samples: {len(train_dataset)}\n")
    f.write(f"Validation samples: {len(val_dataset)}\n")
    f.write(f"Maximum sequence length: {MAX_LENGTH}\n")
    f.write(f"Training batch size: {TRAIN_BATCH_SIZE}\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n")
    f.write(f"FP16: {use_fp16}\n\n")

    f.write("Evaluation results:\n")

    for key, value in evaluation.items():
        f.write(f"{key}: {value}\n")


print("\nModel saved to:")
print(OUTPUT_DIR)

print("\nResults saved to:")
print(results_path)

print("\n" + "=" * 60)
print("BERT TRAINING COMPLETE")
print("=" * 60)
