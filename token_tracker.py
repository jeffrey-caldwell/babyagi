# token_tracker.py

class TokenTracker:
    def __init__(self):
        self.total_tokens = 0

    def add_tokens(self, tokens):
        self.total_tokens += tokens

    def get_total_tokens(self):
        return self.total_tokens

    def calculate_cost(self):
        cost_per_token = 0.002 / 1000
        return self.total_tokens * cost_per_token


token_tracker = TokenTracker()
