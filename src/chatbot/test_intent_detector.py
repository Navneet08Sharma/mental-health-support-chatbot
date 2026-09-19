from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.intent_detector import IntentDetector


TEST_CASES = [
    ("I feel really sad today", "emotional_support"),
    ("I don't know why I feel this way", "emotional_support"),
    ("Can you help me calm down?", "emotional_support"),
    ("I'm having trouble breathing because I'm stressed", "emotional_support"),
    ("Show me a simple breathing exercise", "breathing"),
    ("How can I calm myself with breathing?", "breathing"),
    ("Can you recommend a yoga routine?", "yoga"),
    ("What yoga can I do to relax?", "yoga"),
    ("Help me practice mindfulness", "meditation"),
    ("Give me a short meditation", "meditation"),
    ("I can't focus on studying", "study_help"),
    ("I have an exam next week and I'm stressed", "study_help"),
    ("Help me write about my feelings", "journaling"),
    ("I want to start keeping a diary", "journaling"),
    ("Can we just talk for a while?", "casual_conversation"),
    ("What can you do?", "general_conversation"),
    ("Thanks, I will talk to you later", "goodbye"),
]


def main():

    detector = IntentDetector()

    correct = 0

    print("\n" + "=" * 70)
    print("INTENT DETECTOR - REALISTIC TEST")
    print("=" * 70)

    for number, (message, expected) in enumerate(TEST_CASES, start=1):

        predicted = detector.detect(message)

        if predicted == expected:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"\n{number}. {message}")
        print(f"   Expected : {expected}")
        print(f"   Predicted: {predicted}")
        print(f"   Result   : {status}")

    total = len(TEST_CASES)
    accuracy = correct / total * 100

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Correct : {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
