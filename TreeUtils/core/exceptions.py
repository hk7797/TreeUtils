
class NotFittedError(Exception):
    def __init__(self, *args):
        if len(args) == 0:
            message = "This instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
        else:
            message = args[0]
        self.message = message

    def __str__(self):
        return self.message