import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


class ResponseGenerator:
    """
    Generates supportive chatbot responses using Gemini.

    The class receives:
        - user's message
        - detected emotion
        - recent conversation history

    It sends the combined context to Gemini and returns
    a natural conversational response.

    If Gemini is temporarily unavailable, the class can
    retry the request and eventually fall back to a
    local response.
    """

    def __init__(self):
        # ----------------------------------------------------
        # Load environment variables
        # ----------------------------------------------------

        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables."
            )

        # ----------------------------------------------------
        # Gemini client
        # ----------------------------------------------------

        self.client = genai.Client(
            api_key=api_key
        )

        # ----------------------------------------------------
        # Gemini model
        # ----------------------------------------------------

        self.model = "gemini-3.8-flash"

        # ----------------------------------------------------
        # Retry configuration
        # ----------------------------------------------------

        self.max_retries = 1
        self.last_response_source = "LOCAL FALLBACK"

        print(
            "Gemini response generator loaded successfully."
        )

    # ========================================================
    # MAIN RESPONSE METHOD
    # ========================================================

    def generate_response(
        self,
        message,
        emotion,
        history=None
    ):
        """
        Generate a chatbot response.

        Parameters
        ----------
        message : str
            Latest user message.

        emotion : str
            Emotion detected by the local emotion model.

        history : list
            Previous conversation turns.

        Returns
        -------
        str
            Generated or fallback response.
        """

        if history is None:
            history = []

        try:
            return self._generate_with_gemini(
                message,
                emotion,
                history
            )

        except Exception as error:

            # ------------------------------------------------
            # Quota errors
            # ------------------------------------------------

            if self._is_quota_error(error):

                print(
                    "[Gemini] API quota exceeded."
                )

                return self._fallback_response(
                    message,
                    emotion
                )

            # ------------------------------------------------
            # Temporary server errors
            # ------------------------------------------------

            if self._is_temporary_error(error):

                print(
                    "[Gemini] Temporary Gemini service error."
                )

                return self._fallback_response(
                    message,
                    emotion
                )

            # ------------------------------------------------
            # Other Gemini errors
            # ------------------------------------------------

            print(
                f"[Gemini] Error: {error}"
            )

            return self._fallback_response(
                message,
                emotion
            )

    # ========================================================
    # GEMINI GENERATION
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

        The request intentionally does not use
        GenerateContentConfig because direct requests
        have been more reliable in the current API setup.
        """

        # ----------------------------------------------------
        # System instructions
        # ----------------------------------------------------

        system_prompt = """
You are an empathetic AI mental-health support chatbot.

Respond naturally and conversationally to the user's latest message.

Use previous conversation context when relevant.
Use the detected emotion as additional context, but do not assume it is always correct.

Be supportive when the user is distressed.
Do not sound robotic.
Do not claim to be human.
Do not invent facts about the user.
Keep responses reasonably concise.
Ask a follow-up question only when it naturally helps.
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
{system_prompt}

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
        # GEMINI REQUEST
        # ====================================================

        last_error = None

        for attempt in range(
            self.max_retries + 1
        ):

            try:

                response = (
                    self.client.models.generate_content(
                        model=self.model,
                        contents=user_prompt
                    )
                )

                # ------------------------------------------------
                # Make sure Gemini actually returned text
                # ------------------------------------------------

                if response.text:

                    return response.text.strip()

                raise ValueError(
                    "Gemini returned an empty response."
                )

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

                    # Let generate_response() handle the
                    # final temporary failure gracefully.
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
    # ERROR DETECTION
    # ========================================================

    def _is_quota_error(
        self,
        error
    ):
        """
        Detect Gemini quota/rate-limit errors.
        """

        error_text = str(error).lower()

        quota_indicators = [
            "429",
            "resource_exhausted",
            "quota exceeded",
            "quota_exceeded",
            "rate limit",
            "too_many_requests"
        ]

        return any(
            indicator in error_text
            for indicator in quota_indicators
        )

    # ========================================================

    def _is_temporary_error(
        self,
        error
    ):
        """
        Detect temporary Gemini/server errors.
        """

        error_text = str(error).lower()

        temporary_indicators = [
            "503",
            "unavailable",
            "service unavailable",
            "high demand",
            "temporarily unavailable",
            "500",
            "internal server error",
            "502",
            "bad gateway",
            "504",
            "gateway timeout"
        ]

        return any(
            indicator in error_text
            for indicator in temporary_indicators
        )

    # ========================================================
    # FALLBACK RESPONSE
    # ========================================================

    def _fallback_response(
        self,
        message,
        emotion
    ):
        """
        Generate a local fallback response when Gemini
        cannot be reached.
        """

        message_lower = message.lower().strip()

        # ----------------------------------------------------
        # Greetings
        # ----------------------------------------------------

        greetings = [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening"
        ]

        if message_lower in greetings:

            return (
                "Hi! It's nice to hear from you. "
                "How are you feeling today?"
            )

        # ----------------------------------------------------
        # Goodbye
        # ----------------------------------------------------

        goodbye_messages = [
            "goodbye",
            "bye",
            "bye bye",
            "see you",
            "talk to you later"
        ]

        if message_lower in goodbye_messages:

            return (
                "Take care. I'm glad you stopped by. "
                "You can always come back if you'd like to talk."
            )

        # ----------------------------------------------------
        # Sadness
        # ----------------------------------------------------

        if emotion == "sadness":

            return (
                "I'm sorry you're having a difficult time. "
                "You don't have to figure everything out at once. "
                "If you'd like, you can tell me what's been "
                "on your mind."
            )

        # ----------------------------------------------------
        # Anger
        # ----------------------------------------------------

        if emotion == "anger":

            return (
                "It sounds like something has really frustrated "
                "you. If you want, you can tell me what happened "
                "and we can talk through it."
            )

        # ----------------------------------------------------
        # Fear
        # ----------------------------------------------------

        if emotion == "fear":

            return (
                "It sounds like something is worrying you. "
                "Take a moment to breathe, and if you feel "
                "comfortable, tell me what's making you feel this way."
            )

        # ----------------------------------------------------
        # Surprise
        # ----------------------------------------------------

        if emotion == "surprise":

            return (
                "That sounds like it caught you off guard. "
                "What happened?"
            )

        # ----------------------------------------------------
        # Love
        # ----------------------------------------------------

        if emotion == "love":

            return (
                "That sounds meaningful to you. "
                "I'd be happy to hear more about it."
            )

        # ----------------------------------------------------
        # Joy
        # ----------------------------------------------------

        if emotion == "joy":

            return (
                "That sounds wonderful! "
                "I'm glad something is making you feel good. "
                "Tell me more about it."
            )

        # ----------------------------------------------------
        # Neutral/general fallback
        # ----------------------------------------------------

        return (
            "I'm here to listen. "
            "Tell me a little more about what's on your mind."
        )


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    generator = ResponseGenerator()

    response = generator.generate_response(
        message="I am feeling a little tired today.",
        emotion="sadness",
        history=[]
    )

    print()
    print("Chatbot:")
    print(response)