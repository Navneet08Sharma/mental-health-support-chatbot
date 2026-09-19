class BreathingExercise:
    """
    Provides simple guided breathing exercises
    for relaxation and stress management.
    """

    def __init__(self):
        self.exercises = {
            "box_breathing": {
                "name": "Box Breathing",
                "description": "A simple paced breathing exercise.",
                "steps": [
                    "Inhale slowly for 4 seconds.",
                    "Hold your breath for 4 seconds.",
                    "Exhale slowly for 4 seconds.",
                    "Hold for 4 seconds.",
                ],
                "cycles": 4,
            },
            "calm_breathing": {
                "name": "Calm Breathing",
                "description": "A gentle breathing exercise with a longer exhale.",
                "steps": [
                    "Inhale gently for 4 seconds.",
                    "Exhale slowly for 6 seconds.",
                ],
                "cycles": 5,
            },
        }

    def get_exercise(self, exercise_name="calm_breathing"):
        return self.exercises.get(exercise_name)

    def get_default_exercise(self):
        return self.exercises["calm_breathing"]

    def format_exercise(self, exercise_name="calm_breathing"):
        exercise = self.get_exercise(exercise_name)

        if exercise is None:
            exercise = self.get_default_exercise()

        response = [
            f"{exercise['name']}",
            "",
            exercise["description"],
            "",
            "Follow these steps:",
        ]

        for step in exercise["steps"]:
            response.append(f"- {step}")

        response.extend([
            "",
            f"Repeat the sequence for {exercise['cycles']} cycles.",
            "",
            "Take it gently and don't force your breathing."
        ])

        return "\n".join(response)
