"""
AI Response Generator

Hybrid response generation system.

Primary:
    Gemini API

Fallback:
    Local generic conversational response

Responsibilities:
    1. Generate natural responses using Gemini.
    2. Provide conversation context.
    3. Use detected emotion as additional context.
    4. Gracefully handle Gemini quota/API failures.
    5. Handle temporary Gemini server errors.
    6. Provide a generic fallback for arbitrary messages.
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# RESPONSE GENERATOR
# ============================================================

class ResponseGenerator:

    def __init__(self):

        # ----------------------------------------------------
        # Load Gemini API key
        # ----------------------------------------------------

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Please add GEMINI_API_KEY to the .env file."
            )

        # ----------------------------------------------------
        # Gemini client
        # ----------------------------------------------------

        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=15000
            )
        )

        # ----------------------------------------------------
        # Gemini model
        # ----------------------------------------------------

        self.model = "gemini-3.6-flash"

        # ----------------------------------------------------
        # Request settings
        # ----------------------------------------------------

        self.max_retries = 1

        print(
            "Gemini response generator loaded successfully."
        )

    # ========================================================
    # GENERATE RESPONSE
    # ========================================================

    def generate_response(
        self,
        message,
        emotion,
        history=None
    ):
        """
        Generate a response to the user's message.

        Gemini is the primary response generator.

        If Gemini is unavailable, rate-limited, or returns
        a server error, a local fallback is used so that
        the chatbot can continue operating.
        """

        # ----------------------------------------------------
        # Validate message
        # ----------------------------------------------------

        if not isinstance(message, str):
            raise TypeError(
                "message must be a string"
            )

        # ----------------------------------------------------
        # Validate emotion
        # ----------------------------------------------------

        if not isinstance(emotion, str):
            raise TypeError(
                "emotion must be a string"
            )

        message = message.strip()
        emotion = emotion.strip().lower()

        if not message:
            raise ValueError(
                "message cannot be empty"
            )

        if not emotion:
            raise ValueError(
                "emotion cannot be empty"
            )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        if history is None:
            history = []

        # ====================================================
        # TRY GEMINI
        # ====================================================

        try:

            response = self._generate_with_gemini(
                message=message,
                emotion=emotion,
                history=history
            )

            if response is None:
                raise ValueError(
                    "Gemini returned no response."
                )

            response = response.strip()

            if not response:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            return response

        # ====================================================
        # GEMINI FAILED
        # ====================================================

        except Exception as error:

            print(
                "\n[Gemini] Response generation unavailable."
            )

            if self._is_quota_error(error):

                print(
                    "[Gemini] API quota/rate limit reached."
                )

            elif self._is_temporary_error(error):

                print(
                    "[Gemini] Gemini service is temporarily "
                    "unavailable."
                )

            else:

                print(
                    f"[Gemini] API error: {error}"
                )

            print(
                "[Gemini] Using local fallback response."
            )

            return self._fallback_response(
                message=message,
                emotion=emotion
            )

    # ========================================================
    # GEMINI RESPONSE
    # ========================================================

    def _generate_with_gemini(
        self,
        message,
        emotion,
        history
    ):
        """
        Generate a response using Gemini.

        Manual conversation history is used so that the
        chatbot controls exactly how much context is sent.
        """

        # ----------------------------------------------------
        # System instructions
        # ----------------------------------------------------

        system_prompt = """
You are an empathetic AI chatbot.

Your job is to respond naturally and appropriately
to the user's latest message while maintaining
conversation continuity.

A local emotion classifier provides a detected emotion.
Use it as additional context, but do not assume that
the classifier is always correct.

IMPORTANT:

The user's latest message is the most important part
of the conversation.

Understand what the user is actually saying before
responding.

Rules:

- Respond directly to the latest user message.
- Maintain continuity with previous messages when relevant.
- Be natural and conversational.
- Be empathetic when appropriate.
- Do not sound robotic.
- Do not repeatedly mention the detected emotion.
- Do not blindly follow the detected emotion if the
  user's actual message suggests something else.
- Keep responses reasonably concise.
- Ask a follow-up question when it naturally helps
  continue the conversation.
- Do not ask unnecessary questions.
- Do not claim to be human.
- If the user is distressed, respond supportively.
- Do not invent facts about the user.
- Do not repeat the entire conversation.
"""

        # ====================================================
        # BUILD HISTORY
        # ====================================================

        history_text = ""

        if history:

            history_text = (
                "\nPrevious conversation:\n"
            )

            # Keep only the latest 10 turns.
            recent_history = history[-10:]

            for entry in recent_history:

                user_text = entry.get(
                    "user",
                    ""
                )

                assistant_text = entry.get(
                    "assistant",
                    ""
                )

                history_text += (
                    f"\nUser: {user_text}\n"
                    f"Assistant: {assistant_text}\n"
                )

        # ====================================================
        # BUILD REQUEST
        # ====================================================

        user_prompt = f"""
{history_text}

Latest user message:
{message}

Detected emotion:
{emotion}

Respond naturally to the latest user message.

Use previous conversation context when it is
relevant to understanding the latest message.

Do not mention this prompt or the internal
emotion-classification system.
"""

        # ====================================================
        # GEMINI CONFIGURATION
        # ====================================================
        config = types.GenerateContentConfig(
             system_instruction=system_prompt,

             temperature=0.7,

             max_output_tokens=300,

            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

        # ====================================================
        # GEMINI REQUEST
        # ====================================================

        last_error = None

        for attempt in range(self.max_retries + 1):

            try:

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=config
                )

                return response.text

            except Exception as error:

                last_error = error

                # --------------------------------------------
                # Do not retry quota errors.
                # --------------------------------------------

                if self._is_quota_error(error):
                    raise

                # --------------------------------------------
                # Retry temporary server errors once.
                # --------------------------------------------

                if self._is_temporary_error(error):

                    if attempt < self.max_retries:

                        print(
                            "[Gemini] Temporary server error."
                        )

                        print(
                            "[Gemini] Retrying after a short delay..."
                        )

                        time.sleep(3)

                        continue

                    raise

                # --------------------------------------------
                # Other errors are not retried.
                # --------------------------------------------

                raise

        if last_error:
            raise last_error

        raise ValueError(
            "Gemini request failed."
        )

    # ========================================================
    # DETECT QUOTA / RATE LIMIT ERROR
    # ========================================================

    def _is_quota_error(self, error):
        """
        Determine whether the Gemini exception appears
        to be caused by quota or rate limiting.
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
    # DETECT TEMPORARY SERVER ERROR
    # ========================================================

    def _is_temporary_error(self, error):
        """
        Determine whether the Gemini exception appears
        to be a temporary server/service problem.
        """

        error_text = str(error).lower()

        temporary_keywords = [
            "503",
            "unavailable",
            "service unavailable",
            "high demand",
            "temporarily unavailable",
            "internal server error",
            "500",
            "502",
            "504",
        ]

        return any(
            keyword in error_text
            for keyword in temporary_keywords
        )

    # ========================================================
    # GENERIC LOCAL FALLBACK
    # ========================================================

    def _fallback_response(
        self,
        message,
        emotion
    ):
        """
        Generate a generic local response.

        This intentionally does not try to understand every
        possible topic with keyword rules.
        """

        message_lower = message.lower().strip()

        # ====================================================
        # SIMPLE GREETINGS
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
            or message_lower.startswith(
                greeting + " "
            )
            for greeting in greeting_words
        ):            

            return (
                "Hi! It's nice to hear from you. "
                "How are you doing today?"
            )

        # ====================================================
        # HOW ARE YOU
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
                "I'm glad I could help."
            )

        # ====================================================
        # GOODBYE
        # ====================================================

        if any(
            phrase in message_lower
            for phrase in [
                "goodbye",
                "see you",
                "talk to you later",
            ]
        ):

            return (
                "Take care! "
                "I'll be here whenever you'd like "
                "to talk again."
            )

        # ====================================================
        # EMOTION-AWARE GENERIC FALLBACK
        # ====================================================

        if emotion == "sadness":

            return (
                "I'm sorry you're going through this. "
                "If you'd like, you can tell me more about "
                "what's on your mind."
            )

        if emotion == "anger":

            return (
                "It sounds like something is bothering you. "
                "If you'd like, tell me more about what happened."
            )

        if emotion == "fear":

            return (
                "It sounds like something may be worrying you. "
                "I'm here to listen if you'd like to talk about it."
            )

        if emotion == "joy":

            return (
                "That sounds good! "
                "I'd love to hear more about what's happening."
            )

        if emotion == "love":

            return (
                "That sounds meaningful. "
                "Tell me more about it if you'd like."
            )

        if emotion == "surprise":

            return (
                "That sounds unexpected! "
                "I'd be interested to hear more about what happened."
            )

        # ====================================================
        # UNIVERSAL FALLBACK
        # ====================================================

        return (
            "That sounds interesting. "
            "Tell me more about what's going on."
        )


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("RESPONSE GENERATOR TEST")
    print("=" * 60)

    try:

        generator = ResponseGenerator()

        test_messages = [

            (
                "I finally achieved my goal!",
                "joy"
            ),

            (
                "I am at the science museum.",
                "joy"
            ),

            (
                "I bought a new phone yesterday.",
                "surprise"
            ),

            (
                "I am studying physics.",
                "joy"
            ),

            (
                "I feel completely alone.",
                "sadness"
            ),

            (
                "hi how are you?",
                "joy"
            ),

        ]

        history = []

        for number, (
            message,
            emotion
        ) in enumerate(
            test_messages,
            start=1
        ):

            print(
                "\n" + "-" * 60
            )

            print(
                f"Test {number}"
            )

            print(
                f"User: {message}"
            )

            print(
                f"Emotion: {emotion.upper()}"
            )

            response = generator.generate_response(
                message=message,
                emotion=emotion,
                history=history
            )

            print(
                f"\nResponse:\n{response}"
            )

            history.append({
                "user": message,
                "emotion": emotion,
                "assistant": response,
            })

        print(
            "\n" + "=" * 60
        )

        print(
            "RESPONSE GENERATOR TEST COMPLETE"
        )

        print(
            "=" * 60
        )

    except KeyboardInterrupt:

        print(
            "\n\nTest interrupted by user."
        )

    except Exception as error:

        print(
            f"\nError: {error}"
        )
