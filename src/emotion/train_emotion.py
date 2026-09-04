import os
import numpy as np
import torch

from datasets import (
    load_dataset,
    Dataset,
    concatenate_datasets,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

# Import our custom augmentation examples
from emotion_augmentation import augmentation_data


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

# IMPORTANT:
# This creates a NEW model and does not overwrite V1.
OUTPUT_DIR = "./models/emotion_model_v2"

REPORT_DIR = "./reports"

MAX_LENGTH = 128

EPOCHS = 3

BATCH_SIZE = 16

LEARNING_RATE = 2e-5


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("EMOTION DETECTION MODEL V2 TRAINING")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

print("\nLoading original emotion dataset...")

dataset = load_dataset(
    "dair-ai/emotion"
)

print(dataset)

print("\nOriginal dataset labels:")

print(
    dataset["train"].features["label"].names
)


# ============================================================
# LABEL INFORMATION
# ============================================================

label_names = (
    dataset["train"]
    .features["label"]
    .names
)

num_labels = len(label_names)

print(
    f"\nNumber of emotions: {num_labels}"
)

for i, label in enumerate(label_names):

    print(
        f"{i}: {label}"
    )


# ============================================================
# ORIGINAL TRAINING DATA DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL TRAINING DATA DISTRIBUTION")
print("=" * 70)

train_labels = dataset["train"]["label"]

for label_id, label_name in enumerate(label_names):

    count = train_labels.count(label_id)

    print(
        f"{label_name:<10}: {count}"
    )


# ============================================================
# LOAD CUSTOM AUGMENTATION DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING CUSTOM AUGMENTATION DATA")
print("=" * 70)

print(
    f"Additional examples: {len(augmentation_data)}"
)


# ============================================================
# CONVERT AUGMENTATION DATA TO DATASET
# ============================================================

custom_texts = []

custom_labels = []


for text, emotion in augmentation_data:

    custom_texts.append(text)

    # Convert emotion name to the SAME
    # numeric label used by the original dataset.
    label_id = label_names.index(emotion)

    custom_labels.append(label_id)


custom_dataset = Dataset.from_dict(
    {
        "text": custom_texts,
        "label": custom_labels,
    }
)


print(
    f"Custom dataset created: {len(custom_dataset)} examples"
)


# ============================================================
# FIX LABEL TYPE
# ============================================================

# IMPORTANT:
#
# The original dataset uses Hugging Face ClassLabel.
#
# The custom dataset initially uses Value("int64").
#
# concatenate_datasets() cannot combine those two
# different feature types.
#
# Therefore we cast the custom dataset to exactly
# the same feature structure as the original dataset.

custom_dataset = custom_dataset.cast(
    dataset["train"].features
)


# ============================================================
# CUSTOM DATA DISTRIBUTION
# ============================================================

print("\nCustom augmentation distribution:")

custom_labels_after_cast = custom_dataset["label"]

for label_id, label_name in enumerate(label_names):

    count = custom_labels_after_cast.count(
        label_id
    )

    if count > 0:

        print(
            f"{label_name:<10}: {count}"
        )


# ============================================================
# COMBINE ORIGINAL + CUSTOM TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("COMBINING ORIGINAL + CUSTOM TRAINING DATA")
print("=" * 70)


combined_train_dataset = concatenate_datasets(
    [
        dataset["train"],
        custom_dataset,
    ]
)


print(
    f"Original training examples: "
    f"{len(dataset['train'])}"
)

print(
    f"Additional examples: "
    f"{len(custom_dataset)}"
)

print(
    f"Combined training examples: "
    f"{len(combined_train_dataset)}"
)


# ============================================================
# SHUFFLE COMBINED DATA
# ============================================================

print("\nShuffling combined training dataset...")

combined_train_dataset = combined_train_dataset.shuffle(
    seed=42
)


# ============================================================
# SHOW FINAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TRAINING DATA DISTRIBUTION")
print("=" * 70)

final_labels = combined_train_dataset["label"]

for label_id, label_name in enumerate(label_names):

    count = final_labels.count(label_id)

    print(
        f"{label_name:<10}: {count}"
    )


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


print("Tokenizing training dataset...")

tokenized_train_dataset = (
    combined_train_dataset.map(
        tokenize_function,
        batched=True,
    )
)


print("Tokenizing validation dataset...")

tokenized_validation_dataset = (
    dataset["validation"].map(
        tokenize_function,
        batched=True,
    )
)


print("Tokenizing test dataset...")

tokenized_test_dataset = (
    dataset["test"].map(
        tokenize_function,
        batched=True,
    )
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,

    num_labels=num_labels,

    id2label={
        i: label
        for i, label in enumerate(label_names)
    },

    label2id={
        label: i
        for i, label in enumerate(label_names)
    },
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )
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

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    weight_decay=0.01,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    fp16=torch.cuda.is_available(),

    logging_steps=100,

    report_to="none",

    save_total_limit=2,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train_dataset,

    eval_dataset=tokenized_validation_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


# ============================================================
# START TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING EMOTION MODEL V2 TRAINING")
print("=" * 70)

print(
    f"Training examples: "
    f"{len(tokenized_train_dataset)}"
)

print(
    f"Validation examples: "
    f"{len(tokenized_validation_dataset)}"
)

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)


trainer.train()


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL EMOTION MODEL V2 EVALUATION")
print("=" * 70)


results = trainer.evaluate(
    tokenized_test_dataset
)


print("\nResults:")

for key, value in results.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving emotion model V2...")

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# ============================================================
# SAVE RESULTS
# ============================================================

report_path = os.path.join(
    REPORT_DIR,
    "emotion_results_v2.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EMOTION DETECTION MODEL V2 RESULTS\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Base model: {MODEL_NAME}\n"
    )

    f.write(
        f"Epochs: {EPOCHS}\n"
    )

    f.write(
        f"Batch size: {BATCH_SIZE}\n"
    )

    f.write(
        f"Learning rate: {LEARNING_RATE}\n"
    )

    f.write(
        f"Maximum sequence length: {MAX_LENGTH}\n"
    )

    f.write(
        f"FP16: {torch.cuda.is_available()}\n"
    )

    f.write(
        f"Original training examples: "
        f"{len(dataset['train'])}\n"
    )

    f.write(
        f"Custom augmentation examples: "
        f"{len(custom_dataset)}\n"
    )

    f.write(
        f"Combined training examples: "
        f"{len(combined_train_dataset)}\n\n"
    )

    f.write(
        "Emotion labels:\n"
    )

    for i, label in enumerate(label_names):

        f.write(
            f"{i}: {label}\n"
        )

    f.write(
        "\nEvaluation results:\n"
    )

    for key, value in results.items():

        f.write(
            f"{key}: {value}\n"
        )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("EMOTION MODEL V2 TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nModel saved to:"
)

print(
    f"  {OUTPUT_DIR}"
)

print(
    f"\nResults saved to:"
)

print(
    f"  {report_path}"
)

print("\nYour original V1 model was NOT modified.")

print(
    "V2 contains the original training data "
    "plus your custom emotion examples."
)

print("=" * 70)