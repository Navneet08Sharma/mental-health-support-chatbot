from response_generator import ResponseGenerator


print("=" * 60)
print("RESPONSE GENERATOR TEST")
print("=" * 60)


generator = ResponseGenerator()


tests = [
    (
        "I finally achieved my goal!",
        "joy"
    ),
    (
        "I feel completely alone and nobody understands me.",
        "sadness"
    ),
    (
        "I am extremely angry about what happened.",
        "anger"
    ),
    (
        "I am scared about tomorrow.",
        "fear"
    ),
    (
        "I love spending time with my family.",
        "love"
    ),
    (
        "Wow, I did not expect that at all!",
        "surprise"
    ),
]


for index, (message, emotion) in enumerate(tests, start=1):

    print("\n" + "-" * 60)
    print(f"Test {index}")
    print(f"User: {message}")
    print(f"Emotion: {emotion.upper()}")

    response = generator.generate_response(
        message,
        emotion
    )

    print(f"Gemini: {response}")


print("\n" + "=" * 60)
print("RESPONSE GENERATOR TEST COMPLETE")
print("=" * 60)