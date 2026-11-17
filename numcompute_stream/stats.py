import numpy as np

from numcompute.stats import StreamingStats as _ColumnStats
from numcompute.stats import histogram as _histogram
from numcompute.stats import quantile as _quantile


class StreamingStats:
    """Track statistics for numeric 2D chunks, one column per feature."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all accumulated statistics and return self."""
        self.n_features_in_ = None
        self._columns = []
        self._chunks = []
        return self

    def update_stats(self, X_chunk):
        """Update statistics from X_chunk of shape (samples, features)."""
        array = np.asarray(X_chunk)
        if array.ndim != 2 or 0 in array.shape:
            raise ValueError("X_chunk must have samples and features")
        if array.dtype.kind not in "iuf":
            raise TypeError("X_chunk must contain numeric values")

        array = array.astype(float, copy=True)
        if np.isinf(array).any():
            raise ValueError("X_chunk must not contain infinity")

        if self.n_features_in_ is None: # the first chunk sets the expected number of features
            self.n_features_in_ = array.shape[1]
            self._columns = [
                _ColumnStats(nan_policy="omit")
                for _ in range(self.n_features_in_)
            ]
        elif array.shape[1] != self.n_features_in_:
            raise ValueError("X_chunk has the wrong number of features")

        for column, stats in zip(array.T, self._columns): # keep a splitted running summary for per colum
            stats.update(column)

        self._chunks.append(array) # Save chunks for later quantile and histogram calculations
        return self

    def update(self, X_chunk):
        """Update from one chunk; return self."""
        return self.update_stats(X_chunk)

    def _check_fitted(self):
        if self.n_features_in_ is None:
            raise RuntimeError("update_stats must be called first")

    def mean(self):
        """Return one cumulative mean per feature."""
        self._check_fitted()
        return np.array([stats.mean for stats in self._columns])

    def variance(self, ddof=0):
        """Return one cumulative variance per feature."""
        self._check_fitted()
        return np.array([
            stats.variance(ddof=ddof) for stats in self._columns
        ])

    def quantile(self, q):
        """Return quantiles per feature, ignoring NaNs."""
        self._check_fitted()
        data = np.vstack(self._chunks)
        return _quantile(data, q, axis=0, nan_policy="omit")

    def histogram(self, bins=10, range=None):
        """Return a (counts, edges) pair for each feature."""
        self._check_fitted()
        data = np.vstack(self._chunks)
        return [
            _histogram(column, bins=bins, range=range, nan_policy="omit")
            for column in data.T
        ]