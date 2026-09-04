from predict_emotion import EmotionAnalyzer


# Load the emotion model once
analyzer = EmotionAnalyzer()


# Realistic user messages
test_messages = [
    "I am so happy that I finally achieved my goal!",
    "I feel completely alone and nobody understands me.",
    "I am extremely angry about what happened today.",
    "I am scared about what is going to happen tomorrow.",
    "I really love spending time with my family.",
    "Wow, I didn't expect that to happen at all!",
    "I failed my exam again and I feel like giving up.",
    "My best friend stopped talking to me and I don't know why.",
    "I am nervous about my job interview tomorrow.",
    "Today was just a normal day. Nothing special happened."
]


print("\n" + "=" * 70)
print("EMOTION MODEL - REALISTIC MESSAGE TEST")
print("=" * 70)


for number, message in enumerate(test_messages, start=1):

    result = analyzer.analyze(message)

    print(f"\nTest {number}")
    print("-" * 70)

    print(f"Message: {message}")

    print(
        f"Predicted emotion: {result['emotion'].upper()}"
    )

    print(
        f"Confidence: {result['confidence'] * 100:.2f}%"
    )

    print("\nEmotion probabilities:")

    for emotion, probability in result["probabilities"].items():

        print(
            f"  {emotion:<10}: {probability * 100:.2f}%"
        )


print("\n" + "=" * 70)
print("TESTING COMPLETE")
print("=" * 70)