from pathlib import Path

import torch
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "emotion_model_v2"

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_LENGTH = 128

print("=" * 70)
print("EMOTION MODEL V2 - FULL TEST SET EVALUATION")
print("=" * 70)

print(f"\nModel path:")
print(MODEL_PATH)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nV2 model not found at:\n{MODEL_PATH}"
    )


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading emotion dataset...")

dataset = load_dataset(
    "dair-ai/emotion"
)

test_dataset = dataset["test"]

print(
    f"Test examples: {len(test_dataset)}"
)


# ============================================================
# LABEL INFORMATION
# ============================================================

label_names = (
    dataset["train"]
    .features["label"]
    .names
)

print("\nEmotion labels:")

for index, label in enumerate(label_names):
    print(f"{index}: {label}")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading V2 emotion model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)

model.eval()


# ============================================================
# VERIFY MODEL LABELS
# ============================================================

print("\nModel labels:")

for index, label in model.config.id2label.items():
    print(f"{index}: {label}")


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning predictions...")

all_predictions = []
all_labels = []
all_confidences = []
all_texts = []


for index, example in enumerate(
    test_dataset
):

    text = example["text"]
    actual_label = example["label"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

    predicted_label = torch.argmax(
        probabilities
    ).item()

    confidence = probabilities[
        predicted_label
    ].item()

    all_predictions.append(
        predicted_label
    )

    all_labels.append(
        actual_label
    )

    all_confidences.append(
        confidence
    )

    all_texts.append(
        text
    )

    if (index + 1) % 200 == 0:

        print(
            f"Processed "
            f"{index + 1}/{len(test_dataset)}"
        )


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )
)


# ============================================================
# PER-CLASS METRICS
# ============================================================

class_precision, class_recall, class_f1, support = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=list(range(len(label_names))),
        zero_division=0,
    )
)


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Test examples : {len(test_dataset)}"
)

print(
    f"Accuracy      : {accuracy * 100:.2f}%"
)

print(
    f"Precision     : {precision * 100:.2f}%"
)

print(
    f"Recall        : {recall * 100:.2f}%"
)

print(
    f"F1-score      : {f1 * 100:.2f}%"
)


# ============================================================
# PRINT PER-CLASS RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PER-EMOTION RESULTS")
print("=" * 70)

print(
    f"{'Emotion':<12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'Support':>12}"
)

print("-" * 60)

for index, emotion in enumerate(label_names):

    print(
        f"{emotion:<12}"
        f"{class_precision[index] * 100:>11.2f}%"
        f"{class_recall[index] * 100:>11.2f}%"
        f"{class_f1[index] * 100:>11.2f}%"
        f"{support[index]:>12}"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(label_names))),
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

confusion_matrix_path = (
    REPORT_DIR
    / "emotion_confusion_matrix_v2.png"
)


plt.figure(
    figsize=(9, 7)
)

plt.imshow(
    cm,
    interpolation="nearest",
)

plt.title(
    "Emotion Model V2 - Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(label_names)
)

plt.xticks(
    tick_marks,
    label_names,
    rotation=45,
    ha="right",
)

plt.yticks(
    tick_marks,
    label_names,
)

plt.xlabel(
    "Predicted Emotion"
)

plt.ylabel(
    "Actual Emotion"
)

# Write values inside the matrix

for row in range(cm.shape[0]):

    for column in range(cm.shape[1]):

        plt.text(
            column,
            row,
            str(cm[row, column]),
            ha="center",
            va="center",
        )


plt.tight_layout()

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# MISCLASSIFIED EXAMPLES
# ============================================================

misclassified = []

for index in range(
    len(test_dataset)
):

    actual = all_labels[index]

    predicted = all_predictions[index]

    if actual != predicted:

        misclassified.append(
            {
                "text": all_texts[index],
                "actual": label_names[actual],
                "predicted": label_names[predicted],
                "confidence": all_confidences[index],
            }
        )


print("\n" + "=" * 70)
print("MISCLASSIFIED EXAMPLES")
print("=" * 70)

print(
    f"Total misclassified: "
    f"{len(misclassified)}"
)

# Show first 30 mistakes

for number, item in enumerate(
    misclassified[:30],
    start=1,
):

    print(
        f"\n{number}. Text:"
    )

    print(
        f"   {item['text']}"
    )

    print(
        f"   Actual    : "
        f"{item['actual']}"
    )

    print(
        f"   Predicted : "
        f"{item['predicted']}"
    )

    print(
        f"   Confidence: "
        f"{item['confidence'] * 100:.2f}%"
    )


# ============================================================
# SAVE FULL RESULTS
# ============================================================

results_path = (
    REPORT_DIR
    / "emotion_full_evaluation_v2.txt"
)


with open(
    results_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "EMOTION MODEL V2 - FULL TEST SET EVALUATION\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Model path: {MODEL_PATH}\n"
    )

    file.write(
        f"Test examples: {len(test_dataset)}\n"
    )

    file.write(
        f"Device: {device}\n\n"
    )

    file.write(
        "Emotion labels:\n"
    )

    for index, emotion in enumerate(
        label_names
    ):

        file.write(
            f"{index}: {emotion}\n"
        )

    file.write(
        "\nOverall results:\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.4f}%\n"
    )

    file.write(
        f"Precision: {precision * 100:.4f}%\n"
    )

    file.write(
        f"Recall: {recall * 100:.4f}%\n"
    )

    file.write(
        f"F1-score: {f1 * 100:.4f}%\n"
    )

    file.write(
        "\nPer-emotion results:\n\n"
    )

    for index, emotion in enumerate(
        label_names
    ):

        file.write(
            f"{emotion}\n"
        )

        file.write(
            f"  Precision: "
            f"{class_precision[index] * 100:.4f}%\n"
        )

        file.write(
            f"  Recall: "
            f"{class_recall[index] * 100:.4f}%\n"
        )

        file.write(
            f"  F1-score: "
            f"{class_f1[index] * 100:.4f}%\n"
        )

        file.write(
            f"  Support: "
            f"{support[index]}\n\n"
        )

    file.write(
        "Confusion matrix:\n"
    )

    file.write(
        str(cm)
    )

    file.write(
        "\n\nMisclassified examples:\n"
    )

    for number, item in enumerate(
        misclassified,
        start=1,
    ):

        file.write(
            f"\n{number}. {item['text']}\n"
        )

        file.write(
            f"   Actual: {item['actual']}\n"
        )

        file.write(
            f"   Predicted: {item['predicted']}\n"
        )

        file.write(
            f"   Confidence: "
            f"{item['confidence'] * 100:.2f}%\n"
        )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FULL V2 EVALUATION COMPLETE")
print("=" * 70)

print(
    f"\nResults saved to:"
)

print(
    f"  {results_path}"
)

print(
    f"\nConfusion matrix saved to:"
)

print(
    f"  {confusion_matrix_path}"
)

print("\nNo model training was performed.")
print("The existing V2 model was only evaluated.")

print("=" * 70)