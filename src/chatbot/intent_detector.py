import re


class IntentDetector:

    INTENTS = {

        "goodbye": [
            "goodbye",
            "bye",
            "see you",
            "see you later",
            "talk to you later",
        ],

        "greeting": [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        ],

        "breathing": [
            "breathing exercise",
            "breath exercise",
            "deep breath",
            "breathing technique",
            "breathing",
        ],

        "yoga": [
            "yoga",
            "yoga pose",
            "yoga exercise",
            "yoga routine",
            "stretching",
            "stretch",
        ],

        "meditation": [
            "meditation",
            "meditate",
            "mindfulness",
            "mindful",
        ],

        "study_help": [
            "study",
            "studying",
            "exam",
            "exams",
            "homework",
            "assignment",
            "concentrate",
            "concentration",
            "focus",
            "pomodoro",
        ],

        "journaling": [
            "journal",
            "journaling",
            "write my thoughts",
            "write about my feelings",
            "write down my feelings",
            "diary",
        ],

        "emotional_support": [
            "sad",
            "sadness",
            "lonely",
            "upset",
            "stressed",
            "stress",
            "overwhelmed",
            "worried",
            "worry",
            "anxious",
            "anxiety",
            "scared",
            "afraid",
            "angry",
            "anger",
            "frustrated",
            "frustration",
            "hurt",
            "crying",
            "cry",
            "depressed",
            "hopeless",
            "calm down",
            "feel this way",
            "feeling this way",
            "feel terrible",
            "feel awful",
        ],

        "casual_conversation": [
            "how are you",
            "what are you doing",
            "tell me something",
            "let's talk",
            "lets talk",
            "talk for a while",
            "just talk",
        ],
    }


    # Intent priority.
    # More specific requests should be checked before
    # general emotional-support keywords.
    PRIORITY = [
        "goodbye",
        "greeting",
        "breathing",
        "yoga",
        "meditation",
        "study_help",
        "journaling",
        "casual_conversation",
        "emotional_support",
    ]


    def detect(self, text):

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = text.lower().strip()

        if not text:
            raise ValueError("text cannot be empty")

        for intent in self.PRIORITY:

            patterns = self.INTENTS[intent]

            for pattern in patterns:

                # Use word boundaries for short keywords
                # such as "hi", "bye", "study", etc.
                if re.search(
                    r"\b" + re.escape(pattern) + r"\b",
                    text
                ):
                    return intent

        return "general_conversation"


if __name__ == "__main__":

    detector = IntentDetector()

    test_messages = [
        "Hello",
        "I feel really lonely today",
        "Can you give me a breathing exercise?",
        "Can you suggest some yoga?",
        "I want to meditate",
        "I have an exam tomorrow",
        "I want to write in my journal",
        "How are you?",
        "Goodbye",
        "Tell me something interesting",
        "I don't know what to do today",
    ]

    print("\n" + "=" * 60)
    print("INTENT DETECTOR TEST")
    print("=" * 60)

    for message in test_messages:

        intent = detector.detect(message)

        print(f"\nMessage: {message}")
        print(f"Intent : {intent}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
