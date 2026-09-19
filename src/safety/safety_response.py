class SafetyResponse:
    """
    Provides a cautious response when explicit high-risk
    language is detected.

    This is not a clinical risk assessment.
    """

    def get_response(self, category):
        if category == "self_harm":
            return (
                "I'm really sorry you're dealing with something this painful. "
                "Your immediate safety matters right now. "
                "Please move away from anything you could use to hurt yourself "
                "and stay with someone you trust if possible. "
                "Consider contacting local emergency services or a qualified "
                "mental-health professional for immediate support."
            )

        if category == "immediate_danger":
            return (
                "I'm concerned about your immediate safety. "
                "Please move to a safer place and contact local emergency "
                "services or a trusted person who can stay with you right now."
            )

        if category == "harm_to_others":
            return (
                "It sounds like you may be dealing with very intense feelings. "
                "Please create some distance from the person involved and from "
                "anything that could be used to cause harm. "
                "If you think someone may be in immediate danger, contact local "
                "emergency services or a qualified professional for urgent help."
            )

        return (
            "I'm here to listen, but if you or someone else is in immediate "
            "danger, please contact local emergency services or a trusted "
            "person who can help right now."
        )
