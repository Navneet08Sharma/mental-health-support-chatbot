import re


class SafetyDetector:
    """
    Detects explicit high-risk language in user messages.

    This is a simple rule-based safety layer.
    It is not a clinical risk assessment or diagnosis.
    """

    def __init__(self):
        self.self_harm_patterns = [
            r"\bi want to die\b",
            r"\bi wanna die\b",
            r"\bi want to kill myself\b",
            r"\bi wanna kill myself\b",
            r"\bkill myself\b",
            r"\bend my life\b",
            r"\bend it all\b",
            r"\btake my own life\b",
            r"\bcommit suicide\b",
            r"\bcommitting suicide\b",
            r"\bsuicidal\b",
            r"\bsuicide\b",
            r"\bhurt myself\b",
            r"\bhurt me\b",
            r"\bself[- ]?harm\b",
            r"\bself[- ]?harming\b",
            r"\boverdose\b",
        ]

        self.harm_others_patterns = [
            r"\bi want to kill someone\b",
            r"\bi wanna kill someone\b",
            r"\bkill someone\b",
            r"\bkill somebody\b",
            r"\bhurt someone\b",
            r"\bhurt somebody\b",
            r"\bi want to hurt someone\b",
            r"\bi wanna hurt someone\b",
        ]

        self.immediate_danger_patterns = [
            r"\bi am in immediate danger\b",
            r"\bi'm in immediate danger\b",
            r"\bim in immediate danger\b",
            r"\bin immediate danger\b",
            r"\bi am not safe\b",
            r"\bi'm not safe\b",
            r"\bim not safe\b",
            r"\bnot safe right now\b",
        ]

    def detect(self, message):
        """
        Return a structured safety result.
        """

        if not isinstance(message, str):
            raise TypeError("message must be a string")

        text = message.lower().strip()

        if not text:
            return {
                "risk_level": "none",
                "category": "none",
                "matched": False,
            }

        if self._matches(text, self.immediate_danger_patterns):
            return {
                "risk_level": "high",
                "category": "immediate_danger",
                "matched": True,
            }

        if self._matches(text, self.self_harm_patterns):
            return {
                "risk_level": "high",
                "category": "self_harm",
                "matched": True,
            }

        if self._matches(text, self.harm_others_patterns):
            return {
                "risk_level": "high",
                "category": "harm_to_others",
                "matched": True,
            }

        return {
            "risk_level": "none",
            "category": "none",
            "matched": False,
        }

    @staticmethod
    def _matches(text, patterns):
        return any(
            re.search(pattern, text)
            for pattern in patterns
        )
