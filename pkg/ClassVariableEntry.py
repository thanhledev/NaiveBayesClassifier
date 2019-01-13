class ClassVariableEntry:

    # Initializer / Instance attributes:
    def __init__(self, value: str, prob: float):
        self.value = value
        self.prob = prob

    # instance methods
    def description(self):
        return "{} has probability {}".format(self.value, self.prob)
