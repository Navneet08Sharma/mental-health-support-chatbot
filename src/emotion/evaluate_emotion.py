from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# MODEL PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# IMPORTANT:
# We are testing V2, NOT V1.
MODEL_PATH = PROJECT_ROOT / "models" / "emotion_model_v2"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("EMOTION MODEL V2 - TARGETED EVALUATION")
print("=" * 70)

print(f"\nModel path:")
print(MODEL_PATH)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nV2 model not found at:\n{MODEL_PATH}"
    )


print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


print("Loading emotion model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

model.eval()


print(f"Device: {device}")

print(
    f"Emotions: {list(model.config.id2label.values())}"
)


# ============================================================
# EMOTION ANALYZER
# ============================================================

def predict_emotion(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
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


    predicted_id = torch.argmax(
        probabilities
    ).item()


    predicted_emotion = (
        model.config.id2label[predicted_id]
    )


    confidence = (
        probabilities[predicted_id]
        .item()
    )


    # Create sorted probability dictionary

    emotion_probabilities = {}

    for emotion_id, probability in enumerate(
        probabilities
    ):

        emotion_name = (
            model.config.id2label[emotion_id]
        )

        emotion_probabilities[
            emotion_name
        ] = probability.item()


    emotion_probabilities = dict(
        sorted(
            emotion_probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )


    return (
        predicted_emotion,
        confidence,
        emotion_probabilities,
    )


# ============================================================
# TARGETED TEST DATA
# ============================================================

test_data = [

    # ========================================================
    # LOVE
    # ========================================================

    (
        "I love spending time with my family.",
        "love",
    ),

    (
        "I really enjoy being with my parents.",
        "love",
    ),

    (
        "My family means everything to me.",
        "love",
    ),

    (
        "I care deeply about my friends.",
        "love",
    ),

    (
        "I love my best friend so much.",
        "love",
    ),

    (
        "Spending time with people I love makes me happy.",
        "love",
    ),

    (
        "I feel close to my family.",
        "love",
    ),

    (
        "I really care about the people around me.",
        "love",
    ),


    # ========================================================
    # JOY
    # ========================================================

    (
        "I am so happy today!",
        "joy",
    ),

    (
        "I finally achieved my goal!",
        "joy",
    ),

    (
        "Today was an amazing day.",
        "joy",
    ),

    (
        "I am really excited about my success.",
        "joy",
    ),

    (
        "I feel wonderful today.",
        "joy",
    ),

    (
        "I am so proud of myself.",
        "joy",
    ),


    # ========================================================
    # SADNESS
    # ========================================================

    (
        "I feel completely alone.",
        "sadness",
    ),

    (
        "I am really sad today.",
        "sadness",
    ),

    (
        "Nobody understands me.",
        "sadness",
    ),

    (
        "I feel hopeless.",
        "sadness",
    ),

    (
        "I failed my exam and I feel terrible.",
        "sadness",
    ),

    (
        "I have been feeling down lately.",
        "sadness",
    ),


    # ========================================================
    # ANGER
    # ========================================================

    (
        "I am extremely angry.",
        "anger",
    ),

    (
        "I am furious about what happened.",
        "anger",
    ),

    (
        "That really made me angry.",
        "anger",
    ),

    (
        "I cannot believe they treated me like that.",
        "anger",
    ),

    (
        "I am so frustrated right now.",
        "anger",
    ),

    (
        "This situation makes me furious.",
        "anger",
    ),


    # ========================================================
    # FEAR
    # ========================================================

    (
        "I am scared about tomorrow.",
        "fear",
    ),

    (
        "I am afraid something bad will happen.",
        "fear",
    ),

    (
        "I am nervous about my interview.",
        "fear",
    ),

    (
        "I am worried about what might happen.",
        "fear",
    ),

    (
        "I feel terrified.",
        "fear",
    ),

    (
        "I am anxious about the future.",
        "fear",
    ),


    # ========================================================
    # SURPRISE
    # ========================================================

    (
        "Wow, I did not expect that!",
        "surprise",
    ),

    (
        "I cannot believe that happened!",
        "surprise",
    ),

    (
        "That completely surprised me.",
        "surprise",
    ),

    (
        "I was shocked by what happened.",
        "surprise",
    ),

    (
        "I never saw that coming.",
        "surprise",
    ),

    (
        "What an unexpected surprise!",
        "surprise",
    ),
]


# ============================================================
# RUN EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TARGETED EMOTION EVALUATION")
print("=" * 70)


correct = 0
total = len(test_data)


current_emotion = None


for number, (message, expected) in enumerate(
    test_data,
    start=1,
):

    predicted, confidence, probabilities = (
        predict_emotion(message)
    )


    # Print emotion heading when it changes

    if expected != current_emotion:

        current_emotion = expected

        print("\n")
        print("=" * 70)
        print(
            f"EXPECTED EMOTION: "
            f"{expected.upper()}"
        )
        print("=" * 70)


    print("\nMessage:")
    print(message)

    print(
        f"Expected : "
        f"{expected.upper()}"
    )

    print(
        f"Predicted: "
        f"{predicted.upper()}"
    )

    print(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )


    if predicted == expected:

        print("✓ CORRECT")

        correct += 1

    else:

        print("✗ WRONG")


    print("Top predictions:")

    for emotion, probability in list(
        probabilities.items()
    )[:3]:

        print(
            f"  {emotion:<10} "
            f"{probability * 100:.2f}%"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

accuracy = (
    correct / total
) * 100


print("\n" + "=" * 70)
print("V2 TARGETED EVALUATION RESULT")
print("=" * 70)

print(
    f"Correct predictions : "
    f"{correct}/{total}"
)

print(
    f"Accuracy            : "
    f"{accuracy:.2f}%"
)

print("=" * 70)