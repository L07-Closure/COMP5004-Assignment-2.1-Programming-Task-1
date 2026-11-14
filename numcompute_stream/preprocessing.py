import numpy as np


class StandardScaler:
    """Standardize numeric columns using statistics updated by chunks."""

    def __init__(self):
        self.count_ = None
        self.mean_ = None
        self.M2_ = None

    def _check_X(self, X):
        array = np.asarray(X)
        if array.ndim != 2 or array.shape[1] == 0:
            raise ValueError("X must have shape (n_samples, n_features)")
        if array.dtype.kind not in "biuf":
            raise TypeError("X must contain numeric values")

        array = array.astype(float, copy=True)
        if np.isinf(array).any():
            raise ValueError("X must not contain infinity")
        return array

    def fit(self, X, y=None):
        """Reset statistics, fit on 2D X and return self."""
        self.count_ = None
        self.mean_ = None
        self.M2_ = None
        return self.partial_fit(X, y)

    def partial_fit(self, X_chunk, y=None):
        """Update column statistics from a 2D chunk; return self.

        NaNs are ignored. Raise ValueError for an empty chunk or a
        different feature count. y is accepted for pipeline use.
        """
        array = self._check_X(X_chunk)
        if len(array) == 0:
            raise ValueError("X_chunk must contain at least one sample")

        n_features = array.shape[1]
        if self.count_ is None:
            self.n_features_in_ = n_features
            self.count_ = np.zeros(n_features, dtype=int)
            self.mean_ = np.zeros(n_features)
            self.M2_ = np.zeros(n_features)
        elif n_features != self.n_features_in_:
            raise ValueError("X_chunk has the wrong number of features")

        valid = ~np.isnan(array)
        chunk_count = valid.sum(axis=0)
        chunk_sum = np.nansum(array, axis=0)
        chunk_mean = np.divide(
            chunk_sum, chunk_count,
            out=np.zeros(n_features), where=chunk_count > 0,
        )
        centered = np.where(valid, array - chunk_mean, 0.0)
        chunk_M2 = np.sum(centered ** 2, axis=0)

        total = self.count_ + chunk_count
        delta = chunk_mean - self.mean_
        weight = np.divide(
            chunk_count, total,
            out=np.zeros(n_features), where=total > 0,
        )
        cross = np.divide(
            self.count_ * chunk_count, total,
            out=np.zeros(n_features), where=total > 0,
        )

        self.M2_ += chunk_M2 + delta ** 2 * cross
        self.mean_ += delta * weight
        self.count_ = total
        return self

    def transform(self, X):
        """Return standardized 2D X with the same shape.

        Preserve NaNs. Columns with zero variance use scale 1.
        Raise RuntimeError if no chunk has been fitted.
        """
        if self.count_ is None:
            raise RuntimeError("StandardScaler must be fitted first")

        array = self._check_X(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of features")

        variance = np.divide(
            self.M2_, self.count_,
            out=np.zeros_like(self.M2_), where=self.count_ > 0,
        )
        scale = np.sqrt(np.maximum(variance, 0.0))
        scale = np.where(scale == 0, 1.0, scale)
        return (array - self.mean_) / scale

    def fit_transform(self, X, y=None):
        """Fit on X and return standardized X with the same shape."""
        return self.fit(X, y).transform(X)

class SimpleImputer:
    """Fill numeric NaNs using a running column mean or a fixed value."""

    def __init__(self, strategy="mean", fill_value=0.0):
        """Choose 'mean' or 'constant'; fill_value covers empty columns."""
        if strategy not in ("mean", "constant"):
            raise ValueError("strategy must be 'mean' or 'constant'")
        try:
            fill_value = float(fill_value)
        except (TypeError, ValueError) as exc:
            raise TypeError("fill_value must be numeric") from exc
        if not np.isfinite(fill_value):
            raise ValueError("fill_value must be finite")

        self.strategy = strategy
        self.fill_value = fill_value
        self.count_ = None
        self.mean_ = None

    def _check_X(self, X):
        array = np.asarray(X)
        if array.ndim != 2 or array.shape[1] == 0:
            raise ValueError("X must have shape (n_samples, n_features)")
        if array.dtype.kind not in "biuf":
            raise TypeError("X must contain numeric values")

        array = array.astype(float, copy=True)
        if np.isinf(array).any():
            raise ValueError("X must not contain infinity")
        return array

    def fit(self, X, y=None):
        """Reset estimates, fit on 2D X and return self."""
        self.count_ = None
        self.mean_ = None
        return self.partial_fit(X, y)

    def partial_fit(self, X_chunk, y=None):
        """Update column means from a 2D chunk; return self.

        Ignore NaNs and leave all-NaN columns without an estimate.
        Raise ValueError for empty chunks or a changed feature count.
        """
        array = self._check_X(X_chunk)
        if len(array) == 0:
            raise ValueError("X_chunk must contain at least one sample")

        n_features = array.shape[1]
        if self.count_ is None:
            self.n_features_in_ = n_features
            self.count_ = np.zeros(n_features, dtype=int)
            self.mean_ = np.zeros(n_features)
        elif n_features != self.n_features_in_:
            raise ValueError("X_chunk has the wrong number of features")

        valid = ~np.isnan(array)
        chunk_count = valid.sum(axis=0)
        chunk_sum = np.nansum(array, axis=0)
        chunk_mean = np.divide(
            chunk_sum, chunk_count,
            out=np.zeros(n_features), where=chunk_count > 0,
        )

        total = self.count_ + chunk_count
        weight = np.divide(
            chunk_count, total,
            out=np.zeros(n_features), where=total > 0,
        )
        self.mean_ += (chunk_mean - self.mean_) * weight
        self.count_ = total
        return self

    def transform(self, X):
        """Return 2D X with NaNs filled; preserve its shape.

        Use fill_value until a column has an observed value.
        Raise RuntimeError if no chunk has been fitted.
        """
        if self.count_ is None:
            raise RuntimeError("SimpleImputer must be fitted first")

        array = self._check_X(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of features")

        if self.strategy == "mean":
            values = np.where(self.count_ > 0, self.mean_, self.fill_value)
        else:
            values = np.full(self.n_features_in_, self.fill_value)
        return np.where(np.isnan(array), values, array)

    def fit_transform(self, X, y=None):
        """Fit on X and return a filled array with the same shape."""
        return self.fit(X, y).transform(X)