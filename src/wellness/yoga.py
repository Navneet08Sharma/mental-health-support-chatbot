class YogaWellness:
    """
    Provides simple, general yoga and stretching suggestions
    for relaxation and everyday wellness.
    """

    def __init__(self):
        self.sessions = {
            "gentle_stretch": {
                "name": "Gentle Yoga and Stretching",
                "description": "A short, gentle sequence for relaxation and body awareness.",
                "steps": [
                    "Sit or stand comfortably and take a few natural breaths.",
                    "Slowly roll your shoulders backward 5 times.",
                    "Gently tilt your head from side to side without forcing the movement.",
                    "Reach both arms overhead and stretch gently.",
                    "Move into a comfortable seated forward stretch and hold gently.",
                    "Finish by sitting comfortably and taking a few slow breaths.",
                ],
                "duration": "5-10 minutes",
            },
            "morning_yoga": {
                "name": "Simple Morning Yoga",
                "description": "A gentle sequence to help you start the day with movement.",
                "steps": [
                    "Stand comfortably and take a few natural breaths.",
                    "Reach your arms overhead and stretch gently.",
                    "Perform a few slow shoulder rolls.",
                    "Move through a gentle standing side stretch on each side.",
                    "Slowly bend forward only as far as comfortable.",
                    "Return to standing and take a few relaxed breaths.",
                ],
                "duration": "5-10 minutes",
            },
        }

    def get_session(self, session_name="gentle_stretch"):
        """Return a yoga session by name."""
        return self.sessions.get(session_name)

    def get_default_session(self):
        """Return the default yoga session."""
        return self.sessions["gentle_stretch"]

    def format_session(self, session_name="gentle_stretch"):
        """Return a user-friendly yoga session."""
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
            "Try these gentle movements:",
        ]

        for index, step in enumerate(session["steps"], start=1):
            response.append(f"{index}. {step}")

        response.extend([
            "",
            "Move slowly and stay within a comfortable range.",
            "Stop if a movement causes pain or feels unsafe."
        ])

        return "\n".join(response)
