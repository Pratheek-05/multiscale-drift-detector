import numpy as np

class PageHinkley:
    def __init__(self, delta=0.005, lambda_=50, alpha=0.999):
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.mean = 0
        self.cumulative = 0
        self.minimum = 0
        self.t = 1

    def update(self, value):
        self.mean = self.mean + (value - self.mean) / self.t
        self.cumulative = self.alpha * self.cumulative + (value - self.mean - self.delta)
        self.minimum = min(self.minimum, self.cumulative)
        self.t += 1

        if (self.cumulative - self.minimum) > self.lambda_:
            self.reset()
            return True

        return False