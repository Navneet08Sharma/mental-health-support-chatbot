"""
Conversation Layer

Connects:

User Message
    ↓
V2 Emotion Analyzer
    ↓
Detected Emotion
    ↓
Gemini Response Generator
    ↓
Context-Aware Response
    ↓
Conversation History

If Gemini is unavailable:
    ↓
Graceful fallback response
    ↓
Local emotion detection continues working
"""

from pathlib import Path
import sys


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# ============================================================
# IMPORTS
# ============================================================

from emotion.predict_emotion import EmotionAnalyzer
from chatbot.response_generator import ResponseGenerator


# ============================================================
# CONVERSATION MANAGER
# ============================================================

class ConversationManager:

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING CONVERSATION MANAGER")
        print("=" * 60)

        # ----------------------------------------------------
        # Emotion Analysis
        # ----------------------------------------------------

        print("\nLoading emotion analysis system...")

        self.emotion_analyzer = EmotionAnalyzer()


        # ----------------------------------------------------
        # Gemini Response Generator
        # ----------------------------------------------------

        print("\nLoading Gemini response generator...")

        self.response_generator = ResponseGenerator()


        # ----------------------------------------------------
        # Conversation History
        # ----------------------------------------------------

        self.history = []


        print("\nConversation manager ready.")


    # ========================================================
    # FALLBACK RESPONSE
    # ========================================================

    def _fallback_response(
        self,
        message,
        emotion,
        confidence
    ):
        """
        Generate a context-aware local response when Gemini
        is unavailable.

        The fallback does NOT blindly trust the emotion model.

        It first checks:
            1. Greetings
            2. Common conversational questions
            3. Low-confidence emotion predictions
            4. High-confidence emotions

        This allows the chatbot to remain usable even when
        Gemini API quota is exhausted.
        """

        # ----------------------------------------------------
        # Clean message
        # ----------------------------------------------------

        message_lower = message.lower().strip()


        # ====================================================
        # GREETINGS
        # ====================================================

        greeting_words = [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        ]


        if any(
            message_lower == greeting
            or message_lower.startswith(greeting + " ")
            for greeting in greeting_words
        ):

            return (
                "Hi! I'm doing well. How are you? "
                "What would you like to talk about?"
            )


        # ====================================================
        # HOW ARE YOU QUESTIONS
        # ====================================================

        if (
            "how are you" in message_lower
            or "how are u" in message_lower
        ):

            return (
                "I'm doing well, thanks for asking! "
                "How are you feeling today?"
            )


        # ====================================================
        # THANK YOU
        # ====================================================

        if (
            "thank you" in message_lower
            or "thanks" in message_lower
        ):

            return (
                "You're very welcome! "
                "I'm happy I could help."
            )


        # ====================================================
        # GOODBYE
        # ====================================================

        if any(
            phrase in message_lower
            for phrase in [
                "goodbye",
                "bye",
                "see you",
                "talk to you later",
            ]
        ):

            return (
                "Take care! "
                "I'll be here whenever you'd like to talk again."
            )


        # ====================================================
        # LOW CONFIDENCE
        # ====================================================

        # If the emotion classifier is uncertain, do not let
        # the emotion determine the response.

        if confidence < 0.60:

            return (
                "I'm here with you. "
                "Tell me a little more about what's on your mind."
            )


        # ====================================================
        # HIGH-CONFIDENCE EMOTION RESPONSES
        # ====================================================

        emotion = emotion.lower()


        fallback_responses = {

            "joy":
                (
                    "That sounds wonderful! "
                    "I'm glad to hear that."
                ),


            "love":
                (
                    "That sounds really meaningful. "
                    "It seems like this is something "
                    "you care deeply about."
                ),


            "sadness":
                (
                    "I'm sorry you're feeling this way. "
                    "I'm here to listen if you'd like "
                    "to talk about it."
                ),


            "anger":
                (
                    "It sounds like something really "
                    "upset you. I'm here if you'd like "
                    "to talk about what happened."
                ),


            "fear":
                (
                    "It sounds like you're feeling worried "
                    "or scared. I'm here to listen if you'd "
                    "like to talk about it."
                ),


            "surprise":
                (
                    "That sounds unexpected! "
                    "I'd be interested to hear more "
                    "about what happened."
                ),
        }


        return fallback_responses.get(
            emotion,
            (
                "I'm here to listen. "
                "Tell me more about what's on your mind."
            )
        )


    # ========================================================
    # CHECK GEMINI ERROR
    # ========================================================

    def _is_quota_error(self, error):
        """
        Check whether an exception appears to be related
        to Gemini API quota/rate limits.
        """

        error_text = str(error).lower()


        quota_keywords = [

            "429",

            "resource_exhausted",

            "quota",

            "rate limit",

            "rate_limit",

            "too many requests",

        ]


        return any(
            keyword in error_text
            for keyword in quota_keywords
        )


    # ========================================================
    # PROCESS MESSAGE
    # ========================================================

    def process_message(self, user_message):

        # ----------------------------------------------------
        # Validate message
        # ----------------------------------------------------

        if not isinstance(user_message, str):

            raise TypeError(
                "user_message must be a string"
            )


        user_message = user_message.strip()


        if not user_message:

            raise ValueError(
                "user_message cannot be empty"
            )


        # ====================================================
        # STEP 1 — LOCAL EMOTION ANALYSIS
        # ====================================================

        # IMPORTANT:
        #
        # Emotion analysis is completely local.
        #
        # Therefore it continues working even when
        # Gemini is unavailable.

        emotion_result = (
            self.emotion_analyzer.analyze(
                user_message
            )
        )


        emotion = emotion_result["emotion"]

        confidence = emotion_result["confidence"]


        # ====================================================
        # STEP 2 — GEMINI RESPONSE
        # ====================================================

        assistant_response = None

        gemini_available = True


        try:

            assistant_response = (
                self.response_generator.generate_response(

                    message=user_message,

                    emotion=emotion,

                    history=self.history
                )
            )


            # ------------------------------------------------
            # Safety check
            # ------------------------------------------------

            if not assistant_response:

                raise ValueError(
                    "Gemini returned an empty response."
                )


        except Exception as error:

            gemini_available = False


            # -----------------------------------------------
            # Handle quota/rate-limit errors
            # -----------------------------------------------

            if self._is_quota_error(error):

                print(
                    "\n[Gemini] API quota/rate limit reached."
                )

                print(
                    "[Gemini] Using local fallback response."
                )


            else:

                print(
                    "\n[Gemini] Response generation failed."
                )

                print(
                    f"[Gemini] {error}"
                )

                print(
                    "[Gemini] Using local fallback response."
                )


            # -----------------------------------------------
            # Local fallback
            # -----------------------------------------------

            assistant_response = (
                self._fallback_response(

                    message=user_message,

                    emotion=emotion,

                    confidence=confidence
                )
            )


        # ====================================================
        # STEP 3 — STORE CONVERSATION
        # ====================================================

        history_entry = {

            "user":
                user_message,

            "emotion":
                emotion,

            "confidence":
                confidence,

            "assistant":
                assistant_response,

            "gemini_available":
                gemini_available,
        }


        self.history.append(
            history_entry
        )


        # ====================================================
        # STEP 4 — RETURN RESULT
        # ====================================================

        return {

            "message":
                user_message,

            "emotion":
                emotion,

            "confidence":
                confidence,

            "probabilities":
                emotion_result["probabilities"],

            "response":
                assistant_response,

            "gemini_available":
                gemini_available,
        }


    # ========================================================
    # CLEAR HISTORY
    # ========================================================

    def clear_history(self):

        self.history = []


    # ========================================================
    # GET HISTORY
    # ========================================================

    def get_history(self):

        return self.history.copy()


# ============================================================
# INTERACTIVE CHAT TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("INTERACTIVE EMOTION-AWARE CHATBOT")
    print("=" * 60)

    print("\nType 'exit' to end the conversation.")
    print("Type 'clear' to clear conversation history.\n")


    # --------------------------------------------------------
    # Create Conversation Manager
    # --------------------------------------------------------

    conversation = ConversationManager()


    # --------------------------------------------------------
    # Interactive Loop
    # --------------------------------------------------------

    while True:

        user_message = input("\nYou: ").strip()


        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if user_message.lower() == "exit":

            print("\nEnding conversation...")

            break


        # ----------------------------------------------------
        # Clear History
        # ----------------------------------------------------

        if user_message.lower() == "clear":

            conversation.clear_history()

            print(
                "\nConversation history cleared."
            )

            continue


        # ----------------------------------------------------
        # Empty Input
        # ----------------------------------------------------

        if not user_message:

            print(
                "Please enter a message."
            )

            continue


        # ----------------------------------------------------
        # Process Message
        # ----------------------------------------------------

        try:

            result = (
                conversation.process_message(
                    user_message
                )
            )


            # ------------------------------------------------
            # Display Emotion
            # ------------------------------------------------

            print(
                f"\nEmotion: "
                f"{result['emotion'].upper()}"
            )


            print(
                f"Confidence: "
                f"{result['confidence'] * 100:.2f}%"
            )


            # ------------------------------------------------
            # Display Response
            # ------------------------------------------------

            print(
                f"\nBot: "
                f"{result['response']}"
            )


            # ------------------------------------------------
            # Show fallback status
            # ------------------------------------------------

            if not result["gemini_available"]:

                print(
                    "\n[Note: Gemini was unavailable, "
                    "so a local fallback response was used.]"
                )


        except Exception as error:

            # ------------------------------------------------
            # This should only catch unexpected local errors.
            #
            # Gemini errors should already have been handled
            # inside process_message().
            # ------------------------------------------------

            print(
                f"\nError: {error}"
            )


    # ========================================================
    # CONVERSATION FINISHED
    # ========================================================

    print(
        "\n" + "=" * 60
    )


    print(
        "CONVERSATION ENDED"
    )


    print(
        f"Total conversation turns: "
        f"{len(conversation.history)}"
    )


    print(
        "=" * 60
    )