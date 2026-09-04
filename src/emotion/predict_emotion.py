"""
Reusable Emotion Analysis Module

This module loads the V2 trained emotion classification model
and provides a reusable EmotionAnalyzer class.

The chatbot backend will later use this module like:

    analyzer = EmotionAnalyzer()
    result = analyzer.analyze("I feel really lonely today.")
"""

from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Project root:
# C:\Projects\MAJOR PROJECT(JONAS)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# IMPORTANT:
# Use V2 model.
#
# V1 remains untouched at:
# models/emotion_model
#
# V2 is:
# models/emotion_model_v2

MODEL_PATH = PROJECT_ROOT / "models" / "emotion_model_v2"


# ============================================================
# EMOTION ANALYZER
# ============================================================

class EmotionAnalyzer:

    def __init__(self, model_path=MODEL_PATH):
        """
        Load the trained emotion model and tokenizer.
        """

        self.model_path = Path(model_path)

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Emotion model not found at: "
                f"{self.model_path}"
            )


        print("Loading emotion tokenizer...")

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_path
            )
        )


        print("Loading emotion model...")

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.model_path
            )
        )


        # ====================================================
        # DEVICE
        # ====================================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


        self.model.to(self.device)

        self.model.eval()


        # ====================================================
        # LABEL INFORMATION
        # ====================================================

        self.id2label = (
            self.model.config.id2label
        )

        self.label2id = (
            self.model.config.label2id
        )


        print(
            "Emotion model loaded successfully."
        )

        print(
            f"Model: {self.model_path}"
        )

        print(
            f"Device: {self.device}"
        )

        print(
            f"Emotions: "
            f"{list(self.id2label.values())}"
        )


    # ========================================================
    # ANALYZE TEXT
    # ========================================================

    def analyze(self, text):
        """
        Analyze the emotional content of a
        user's message.

        Returns:

            {
                "emotion": predicted emotion,
                "confidence": confidence score,
                "probabilities": all emotion probabilities
            }
        """

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if not isinstance(text, str):

            raise TypeError(
                "text must be a string"
            )


        text = text.strip()


        if not text:

            raise ValueError(
                "text cannot be empty"
            )


        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )


        # ----------------------------------------------------
        # Move tensors to device
        # ----------------------------------------------------

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }


        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )


            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )[0]


        # ----------------------------------------------------
        # Highest probability emotion
        # ----------------------------------------------------

        predicted_id = (
            torch.argmax(
                probabilities
            ).item()
        )


        predicted_emotion = (
            self.id2label[predicted_id]
        )


        confidence = (
            probabilities[
                predicted_id
            ].item()
        )


        # ----------------------------------------------------
        # Create probability dictionary
        # ----------------------------------------------------

        emotion_probabilities = {}


        for emotion_id, probability in enumerate(
            probabilities
        ):

            emotion_name = (
                self.id2label[emotion_id]
            )


            emotion_probabilities[
                emotion_name
            ] = float(
                probability.item()
            )


        # ----------------------------------------------------
        # Sort highest → lowest
        # ----------------------------------------------------

        emotion_probabilities = dict(
            sorted(
                emotion_probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {

            "emotion": predicted_emotion,

            "confidence": confidence,

            "probabilities":
                emotion_probabilities,
        }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print(
        "EMOTION ANALYSIS MODULE TEST - V2"
    )

    print("=" * 60)


    # Load V2 analyzer

    analyzer = EmotionAnalyzer()


    # Test messages

    test_messages = [

        "I feel really happy today!",

        "I feel lonely and nobody understands me.",

        "I am extremely angry about what happened.",

        "I am scared about tomorrow.",

        "I love spending time with my family.",

        "Wow, I did not expect that at all!",
    ]


    # --------------------------------------------------------
    # Run tests
    # --------------------------------------------------------

    for number, message in enumerate(
        test_messages,
        start=1,
    ):

        result = analyzer.analyze(
            message
        )


        print("\n" + "-" * 60)

        print(
            f"Test {number}"
        )

        print(
            f"Input: {message}"
        )

        print(
            f"Predicted emotion: "
            f"{result['emotion'].upper()}"
        )

        print(
            f"Confidence: "
            f"{result['confidence'] * 100:.2f}%"
        )


        print(
            "\nEmotion probabilities:"
        )


        for emotion, probability in (
            result["probabilities"].items()
        ):

            print(
                f"  {emotion:<10}: "
                f"{probability * 100:.2f}%"
            )


    print("\n" + "=" * 60)

    print(
        "V2 EMOTION ANALYZER TEST COMPLETE"
    )

    print("=" * 60)