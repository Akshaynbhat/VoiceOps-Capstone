"""Conversation state and agent."""
class ConversationState:
    def __init__(self):
        self.history = []
        self.current_order_id = None
        self.verification_attempts = 0
        self.awaiting_verification = False
        self.last_intent = None

def handle_message(message, state, *args, **kwargs):
    raise NotImplementedError("Implement and test the agent.")
