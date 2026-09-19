class MeditationWellness:
    """
    Provides simple guided meditation exercises
    for relaxation and everyday mindfulness.
    """

    def __init__(self):
        self.sessions = {
            "short_meditation": {
                "name": "Short Mindfulness Meditation",
                "description": "A simple meditation focused on breathing and present-moment awareness.",
                "duration": "5 minutes",
                "steps": [
                    "Sit comfortably and allow your shoulders to relax.",
                    "Close your eyes if that feels comfortable, or soften your gaze.",
                    "Take a slow, natural breath and notice how it feels.",
                    "Bring your attention gently to your breathing.",
                    "If your thoughts wander, notice them without judging yourself and return your attention to your breath.",
                    "Continue for a few minutes at a comfortable pace.",
                    "Before finishing, take one more slow breath and notice how you feel."
                ]
            },
            "body_scan": {
                "name": "Simple Body Scan",
                "description": "A gentle exercise for noticing sensations in the body.",
                "duration": "5-10 minutes",
                "steps": [
                    "Sit or lie down in a comfortable position.",
                    "Take a few natural breaths.",
                    "Notice how your feet and legs feel.",
                    "Slowly bring your attention to your stomach, chest, and shoulders.",
                    "Notice your arms, hands, neck, and face.",
                    "You do not need to change anything; simply notice the sensations.",
                    "Take a final slow breath and gently bring your attention back to your surroundings."
                ]
            }
        }

    def get_session(self, session_name="short_meditation"):
        """Return a meditation session by name."""
        return self.sessions.get(session_name)

    def get_default_session(self):
        """Return the default meditation session."""
        return self.sessions["short_meditation"]

    def format_session(self, session_name="short_meditation"):
        """Return a user-friendly meditation session."""
        session = self.get_session(session_name)

        if session is None:
            session = self.get_default_session()

        response = [
            session["name"],
            "",
            session["description"],
            "",
            f"Suggested duration: {session['duration']}",
            "",
            "Follow these steps:",
        ]

        for index, step in enumerate(session["steps"], start=1):
            response.append(f"{index}. {step}")

        response.extend([
            "",
            "There is no need to do this perfectly.",
            "If meditation feels uncomfortable, stop and return to your normal routine."
        ])

        return "\n".join(response)
