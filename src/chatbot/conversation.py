from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from emotion.predict_emotion import EmotionAnalyzer
from chatbot.intent_detector import IntentDetector
from chatbot.response_generator import ResponseGenerator
from src.wellness.breathing import BreathingExercise
from src.wellness.yoga import YogaWellness
from src.wellness.meditation import MeditationWellness
from src.safety.safety_detector import SafetyDetector
from src.safety.safety_response import SafetyResponse


class ConversationManager:

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING CONVERSATION MANAGER")
        print("=" * 60)

        print("\nLoading emotion analysis system...")
        self.emotion_analyzer = EmotionAnalyzer()

        print("\nLoading intent detection system...")
        self.intent_detector = IntentDetector()
        self.breathing_exercise = BreathingExercise()
        self.yoga_wellness = YogaWellness()
        self.meditation_wellness = MeditationWellness()
        self.safety_detector = SafetyDetector()
        self.safety_response = SafetyResponse()

        print("\nLoading Gemini response generator...")
        self.response_generator = ResponseGenerator()

        self.history = []

        print("\nConversation manager ready.")


    # ========================================================
    # FALLBACK RESPONSE
    # ========================================================

    def _fallback_response(
        self,
        message,
        intent,
        emotion,
        confidence
    ):

        message_lower = message.lower().strip()


        # ----------------------------------------------------
        # SPECIFIC FEATURE INTENTS
        # ----------------------------------------------------

        if intent == "breathing":
            return self.breathing_exercise.format_exercise()

            

            


        if intent == "yoga":
            return self.yoga_wellness.format_session()

            


        if intent == "meditation":
            return self.meditation_wellness.format_session()

            

            


        if intent == "study_help":

            return (
                "Let's make studying manageable. "
                "Choose one small topic and focus on it for "
                "25 minutes, then take a short break. "
                "You can repeat the cycle if it works for you."
            )


        if intent == "journaling":

            return (
                "You could start with a simple prompt: "
                "'What am I feeling right now, and what might "
                "be contributing to it?' Write whatever comes "
                "to mind without worrying about making it perfect."
            )


        if intent == "casual_conversation":

            return (
                "Of course. I'm here to chat. "
                "What's on your mind?"
            )


        # ----------------------------------------------------
        # GREETING
        # ----------------------------------------------------

        if intent == "greeting":

            return (
                "Hi! How are you feeling today? "
                "What would you like to talk about?"
            )


        # ----------------------------------------------------
        # GOODBYE
        # ----------------------------------------------------

        if intent == "goodbye":

            return (
                "Take care! I'll be here whenever "
                "you'd like to talk again."
            )


        # ----------------------------------------------------
        # HOW ARE YOU
        # ----------------------------------------------------

        if (
            "how are you" in message_lower
            or "how are u" in message_lower
        ):

            return (
                "I'm doing well, thanks for asking! "
                "How are you feeling today?"
            )


        # ----------------------------------------------------
        # THANK YOU
        # ----------------------------------------------------

        if (
            "thank you" in message_lower
            or "thanks" in message_lower
        ):

            return (
                "You're very welcome! "
                "I'm happy I could help."
            )


        # ----------------------------------------------------
        # LOW CONFIDENCE
        # ----------------------------------------------------

        if confidence < 0.60:

            return (
                "I'm here with you. "
                "Tell me a little more about what's on your mind."
            )


        # ----------------------------------------------------
        # EMOTION RESPONSES
        # ----------------------------------------------------

        emotion = emotion.lower()

        fallback_responses = {

            "joy": (
                "That sounds wonderful! "
                "I'm glad to hear that."
            ),

            "love": (
                "That sounds really meaningful. "
                "It seems like this is something "
                "you care deeply about."
            ),

            "sadness": (
                "I'm sorry you're feeling this way. "
                "I'm here to listen if you'd like "
                "to talk about it."
            ),

            "anger": (
                "It sounds like something really "
                "upset you. I'm here if you'd like "
                "to talk about what happened."
            ),

            "fear": (
                "It sounds like you're feeling worried "
                "or scared. I'm here to listen if you'd "
                "like to talk about it."
            ),

            "surprise": (
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
    # GEMINI ERROR CHECK
    # ========================================================

    def _is_quota_error(self, error):

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

        if not isinstance(user_message, str):
            raise TypeError(
                "user_message must be a string"
            )

        user_message = user_message.strip()

        if not user_message:
            raise ValueError(
                "user_message cannot be empty"
            )


        # ----------------------------------------------------
        # STEP 1: EMOTION
        # ----------------------------------------------------

        emotion_result = (
            self.emotion_analyzer.analyze(
                user_message
            )
        )

        emotion = emotion_result["emotion"]
        confidence = emotion_result["confidence"]

        if confidence < 0.60:
            emotion_for_response = "uncertain"
        else:
            emotion_for_response = emotion


        # ----------------------------------------------------
        # STEP 2: INTENT
        # ----------------------------------------------------

        intent = self.intent_detector.detect(
            user_message
        )

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        safety_result = self.safety_detector.detect(
            user_message
        )


        # ----------------------------------------------------
        # STEP 3: SAFETY / GEMINI
        # ----------------------------------------------------

        assistant_response = None
        response_source = "LOCAL FALLBACK"
        gemini_available = True

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        safety_triggered = safety_result["matched"]

        if safety_triggered:

            # High-risk messages bypass Gemini.
            assistant_response = (
                self.safety_response.get_response(
                    safety_result["category"]
                )
            )

            response_source = "SAFETY RESPONSE"
            gemini_available = False

        else:

            # ------------------------------------------------
            # GEMINI RESPONSE
            # ------------------------------------------------

            try:

                assistant_response = (
                    self.response_generator.generate_response(
                        message=user_message,
                        emotion=emotion,
                        history=self.history
                    )
                )

                # ResponseGenerator handles Gemini errors
                # internally. Check which source was actually used.
                if (
                    self.response_generator.last_response_source
                    == "LOCAL FALLBACK"
                ):

                    gemini_available = False
                    response_source = "LOCAL FALLBACK"

                    # Use the intent-specific fallback instead
                    # of the generic emotion fallback.
                    assistant_response = (
                        self._fallback_response(
                            message=user_message,
                            intent=intent,
                            emotion=emotion,
                            confidence=confidence
                        )
                    )

                else:

                    response_source = "GEMINI"

                if not assistant_response:

                    raise ValueError(
                        "Gemini returned an empty response."
                    )

            except Exception as error:

                gemini_available = False
                response_source = "LOCAL FALLBACK"

                if self._is_quota_error(error):

                    print(
                        "\n[Gemini] API quota/rate limit reached."
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

                assistant_response = (
                    self._fallback_response(
                        message=user_message,
                        intent=intent,
                        emotion=emotion,
                        confidence=confidence
                    )
                )

        # ----------------------------------------------------
        # STEP 4: HISTORY
        # ----------------------------------------------------

        history_entry = {

            "user":
                user_message,

            "intent":
                intent,

            "emotion":
                emotion,

            "confidence":
                confidence,

            "assistant":
                assistant_response,

            "gemini_available":
                gemini_available,

            "response_source":
                response_source,

            "safety":
                safety_result,
        }

        self.history.append(
            history_entry
        )


        # ----------------------------------------------------
        # STEP 5: RESULT
        # ----------------------------------------------------

        return {

            "message":
                user_message,

            "intent":
                intent,

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

            "response_source":
                response_source,

            "safety":
                safety_result,
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
# INTERACTIVE TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("INTERACTIVE EMOTION-AWARE CHATBOT")
    print("=" * 60)

    print("\nType 'exit' to end the conversation.")
    print("Type 'clear' to clear conversation history.\n")


    conversation = ConversationManager()


    while True:

        user_message = input("\nYou: ").strip()


        if user_message.lower() == "exit":

            print("\nEnding conversation...")
            break


        if user_message.lower() == "clear":

            conversation.clear_history()

            print(
                "\nConversation history cleared."
            )

            continue


        if not user_message:

            print(
                "Please enter a message."
            )

            continue


        try:

            result = (
                conversation.process_message(
                    user_message
                )
            )


            safety = result.get("safety", {})
            safety_triggered = safety.get("matched", False)

            print(
                f"\nIntent: "
                f"{result['intent'].upper()}"
            )

            if safety_triggered:

                print(
                    "\nSafety: HIGH RISK"
                )

                print(
                    f"Category: "
                    f"{safety.get('category', 'SAFETY CONCERN').upper()}"
                )

                print(
                    "Emotion: OVERRIDDEN BY SAFETY"
                )

                print(
                    "Confidence: N/A"
                )

            else:

                print(
                    f"Emotion: "
                    f"{result['emotion'].upper()}"
                )

                print(
                    f"Confidence: "
                    f"{result['confidence'] * 100:.2f}%"
                )

            print(
                f"\nBot: "
                f"{result['response']}"
            )

            print(
                f"\nResponse source: "
                f"{result['response_source']}"
            )

            if result["response_source"] == "SAFETY RESPONSE":

                print(
                    "\n[Note: Safety response activated; "
                    "Gemini was bypassed.]"
                )

            elif not result["gemini_available"]:

                print(
                    "\n[Note: Gemini was unavailable, "
                    "so a local fallback response was used.]"
                )
        except Exception as error:

            print(
                f"\nError: {error}"
            )


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
