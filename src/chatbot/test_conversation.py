"""
Conversation Layer Test

Tests the complete pipeline:

User message
    ↓
Emotion Analyzer
    ↓
Emotion
    ↓
Response Generator
    ↓
Chatbot response
"""

from conversation import ConversationManager


# ---------------------------------------------------------
# CREATE CHATBOT
# ---------------------------------------------------------

chatbot = ConversationManager()


# ---------------------------------------------------------
# TEST MESSAGES
# ---------------------------------------------------------

test_messages = [

    "I am so happy that I finally achieved my goal!",

    "I feel completely alone and nobody understands me.",

    "I am extremely angry about what happened today.",

    "I am scared about what is going to happen tomorrow.",

    "I really love spending time with my family.",

    "Wow, I didn't expect that to happen at all.",

    "I failed my exam again and I feel like giving up.",

    "My best friend stopped talking to me and I don't know why.",

    "I am nervous about my job interview tomorrow.",

    "Today was just a normal day. Nothing special happened."
]


# ---------------------------------------------------------
# RUN TESTS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CONVERSATION LAYER TEST")
print("=" * 70)


for number, message in enumerate(test_messages, start=1):

    print(f"\nTest {number}")
    print("-" * 70)

    result = chatbot.process_message(message)

    print(f"User: {result['message']}")

    print(
        f"Emotion: "
        f"{result['emotion'].upper()}"
    )

    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"Bot: "
        f"{result['response']}"
    )


print("\n" + "=" * 70)
print("CONVERSATION TEST COMPLETE")
print("=" * 70)