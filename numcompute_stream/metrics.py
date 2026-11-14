from collections import deque

import numpy as np


class StreamingAccuracy:
    """Track accuracy across 1D chunks of true and predicted labels."""

    def __init__(self, window_size=None):
        """Set an optional rolling window measured in samples."""
        if window_size is not None and (
            type(window_size) is not int or window_size <= 0
        ):
            raise ValueError("window_size must be a positive integer")
        self.window_size = window_size
        self.reset()

    def update(self, y_true_chunk, y_pred_chunk):
        """Add equally sized 1D label chunks and return self."""
        y_true = np.asarray(y_true_chunk)
        y_pred = np.asarray(y_pred_chunk)

        if y_true.ndim != 1 or y_pred.ndim != 1:
            raise ValueError("labels must be 1D")
        if len(y_true) == 0 or len(y_true) != len(y_pred):
            raise ValueError("label chunks must have the same nonzero length")

        for labels in (y_true, y_pred):
            if np.issubdtype(labels.dtype, np.number):
                if not np.all(np.isfinite(labels)):
                    raise ValueError("labels must not contain NaN or infinity")

        correct = y_true == y_pred
        self.correct += int(np.count_nonzero(correct))
        self.total += len(y_true)

        if self.window_size is not None:
            for value in correct:
                if len(self._recent) == self.window_size:
                    self._window_correct -= self._recent.popleft()
                self._recent.append(int(value))
                self._window_correct += int(value)

        return self

    def result(self):
        """Return accuracy over every sample added since reset."""
        if self.total == 0:
            raise ValueError("no samples have been added")
        return self.correct / self.total

    def window_result(self):
        """Return accuracy over the most recent window_size samples."""
        if self.window_size is None:
            raise ValueError("window_size was not set")
        if not self._recent:
            raise ValueError("no samples have been added")
        return self._window_correct / len(self._recent)

    def reset(self):
        """Clear cumulative and rolling counts."""
        self.correct = 0
        self.total = 0
        self._recent = deque()
        self._window_correct = 0