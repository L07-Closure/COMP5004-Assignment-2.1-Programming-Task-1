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

class StreamingConfusionMatrix:
    """Count true and predicted labels across 1D chunks."""

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
            if labels.dtype.kind not in "biufUS":
                raise TypeError("labels must be numbers or strings")
            if labels.dtype.kind in "biuf":
                if not np.all(np.isfinite(labels)):
                    raise ValueError("labels must not contain NaN or infinity")

        true_is_numeric = y_true.dtype.kind in "biuf"
        pred_is_numeric = y_pred.dtype.kind in "biuf"
        if true_is_numeric != pred_is_numeric:
            raise TypeError("true and predicted labels must have the same kind")

        for true, pred in zip(y_true, y_pred):
            pair = (true, pred)
            self._counts[pair] = self._counts.get(pair, 0) + 1
            self._labels.update(pair)
            self.total += 1

            if self.window_size is not None:
                if len(self._recent) == self.window_size:
                    old = self._recent.popleft()
                    self._window_counts[old] -= 1
                    if self._window_counts[old] == 0:
                        del self._window_counts[old]

                self._recent.append(pair)
                self._window_counts[pair] = (
                    self._window_counts.get(pair, 0) + 1
                )

        return self

    def _make_result(self, counts, labels):
        """Build a matrix with true labels as rows and predictions as columns."""
        ordered = sorted(labels)
        positions = {label: i for i, label in enumerate(ordered)}
        matrix = np.zeros((len(ordered), len(ordered)), dtype=int)

        for (true, pred), count in counts.items():
            matrix[positions[true], positions[pred]] = count

        return matrix, np.asarray(ordered)

    def result(self):
        """Return the cumulative (matrix, labels) pair."""
        if self.total == 0:
            raise ValueError("no samples have been added")
        return self._make_result(self._counts, self._labels)

    def window_result(self):
        """Return the (matrix, labels) pair for recent samples."""
        if self.window_size is None:
            raise ValueError("window_size was not set")
        if not self._recent:
            raise ValueError("no samples have been added")

        labels = set()
        for true, pred in self._recent:
            labels.update((true, pred))
        return self._make_result(self._window_counts, labels)

    def reset(self):
        """Clear cumulative and rolling counts."""
        self.total = 0
        self._counts = {}
        self._labels = set()
        self._recent = deque()
        self._window_counts = {}