from bisect import bisect_left
from datetime import datetime

class TimeDict:
    def __init__(self, d: dict[datetime, float | int]):
        if not d:
            raise ValueError("Empty dictionary")

        self.keys = sorted(d.keys())
        self.values = [d[k] for k in self.keys]

    def closest(self, target: datetime):
        """Return the key-value pair closest to the target datetime."""
        i = bisect_left(self.keys, target)
        if i == 0:
            return self.keys[0], self.values[0]
        if i == len(self.keys):
            return self.keys[-1], self.values[-1]

        before_key = self.keys[i - 1]
        after_key = self.keys[i]
        if target - before_key <= after_key - target:
            return before_key, self.values[i - 1]
        else:
            return after_key, self.values[i]
