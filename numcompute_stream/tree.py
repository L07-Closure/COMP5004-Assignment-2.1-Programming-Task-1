import numpy as np

class _Node:
    def __init__(self, value):
        self.value = value
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    """Classify numeric rows with a decision tree."""

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        max_features=None,
        buffer_size=500,
        criterion="gini",
    ):
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer")
        if type(min_samples_split) is not int or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer of at least 2")
        if max_features is not None and (
            type(max_features) is not int or max_features < 1
        ):
            raise ValueError("max_features must be a positive integer or None")
        if type(buffer_size) is not int or buffer_size < 1:
            raise ValueError("buffer_size must be a positive integer")
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.buffer_size = buffer_size
        self.criterion = criterion

        self._root = None
        self._buffer_X = []
        self._buffer_y = []
        self._rng = np.random.default_rng(0)

    def _impurity(self, y):
        """Return the Gini impurity or entropy of the labels."""
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()

        if self.criterion == "gini":
            return float(1.0 - np.sum(p ** 2))
        return float(-np.sum(p * np.log2(p)))

    def _impurity_from_counts(self, counts):
        """Calculate impurity for several sets of class counts."""
        p = counts / counts.sum(axis=1, keepdims=True)

        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2, axis=1)

        log_p = np.zeros_like(p)
        np.log2(p, out=log_p, where=p > 0)
        return -np.sum(p * log_p, axis=1)

    def _best_split(self, X, y):
        """Find the feature and threshold with the lowest split impurity."""
        n_samples, n_features = X.shape
        classes, y_codes = np.unique(y, return_inverse=True)
        total_counts = np.bincount(y_codes, minlength=len(classes))

        best_feature = None
        best_threshold = None
        best_score = self._impurity(y)

        features = np.arange(n_features)
        if self.max_features is not None:
            features = self._rng.choice(
                features,
                size=min(self.max_features, n_features),
                replace=False,
            )

        for feature in features:
            order = np.argsort(X[:, feature], kind="stable")
            values = X[order, feature]
            cuts = np.flatnonzero(values[:-1] < values[1:]) + 1 # the same value can not be splitted
            if len(cuts) == 0:
                continue

            sorted_codes = y_codes[order]
            class_rows = sorted_codes[:, None] == np.arange(len(classes)) # count each class on the lesft of every possible vut
            left_counts = np.cumsum(class_rows, axis=0)[cuts - 1]
            right_counts = total_counts - left_counts

            left_score = self._impurity_from_counts(left_counts)
            right_score = self._impurity_from_counts(right_counts)
            scores = ( # weight each side by its number of samples
                cuts * left_score + (n_samples - cuts) * right_score
            ) / n_samples

            index = int(np.argmin(scores))
            if scores[index] < best_score - 1e-12: # ignore small impurity changes by rounding
                cut = int(cuts[index])
                threshold = values[cut - 1] / 2 + values[cut] / 2 # find midpoint
                if threshold >= values[cut]:
                    threshold = values[cut - 1]

                best_feature = int(feature)
                best_threshold = float(threshold)
                best_score = float(scores[index])

        return best_feature, best_threshold

    def _build(self, X, y, depth):
        """Build a node and its children recursively."""
        labels, counts = np.unique(y, return_counts=True)
        node = _Node(labels[np.argmax(counts)])

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(labels) == 1
        ):
            return node

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return node

        left_mask = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def fit(self, X, y):
        """Train from 2D numeric X and 1D labels y; return self.

        Replace NaNs in X with each column's median (0 for an all-NaN
        column). Raise TypeError for unsupported types and ValueError
        for invalid shapes, empty data, NaN labels or infinite X.
        """
        X_array = np.asarray(X)
        y_array = np.asarray(y)

        if X_array.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features)")
        if X_array.dtype.kind not in "biuf":
            raise TypeError("X must contain numeric values")
        if X_array.shape[0] == 0 or X_array.shape[1] == 0:
            raise ValueError("X must contain samples and features")
        if y_array.ndim != 1 or len(y_array) != len(X_array):
            raise ValueError("y must have shape (n_samples,)")
        if y_array.dtype.kind not in "biufUS":
            raise TypeError("y must contain numeric or string labels")
        if y_array.dtype.kind == "f" and np.isnan(y_array).any():
            raise ValueError("y must not contain NaN")

        X_array = X_array.astype(float, copy=True)
        if np.isinf(X_array).any():
            raise ValueError("X must not contain infinity")

        fill_values = np.zeros(X_array.shape[1])
        for feature in range(X_array.shape[1]):
            known = X_array[~np.isnan(X_array[:, feature]), feature]
            if len(known):
                fill_values[feature] = np.median(known)

        X_clean = np.where(np.isnan(X_array), fill_values, X_array)
        self._rng = np.random.default_rng(0)
        new_root = self._build(X_clean, y_array, depth=0)

        self._root = new_root
        self._fill_values = fill_values
        self.n_features_in_ = X_array.shape[1]
        self.classes_ = np.unique(y_array)
        self._buffer_X = []
        self._buffer_y = []
        return self
    def _predict_node(self, node, X, rows, predictions):
        """Send the selected rows through this node."""
        if len(rows) == 0:
            return
        if node.feature is None:
            predictions[rows] = node.value
            return

        left_mask = X[rows, node.feature] <= node.threshold
        self._predict_node(node.left, X, rows[left_mask], predictions)
        self._predict_node(node.right, X, rows[~left_mask], predictions)

    def predict(self, X):
        """Predict labels for 2D numeric X; return a 1D label array.

        Use the medians learned during fit for NaNs. Raise RuntimeError
        before fitting, TypeError for nonnumeric X, and ValueError for
        wrong shapes or infinite values.
        """
        if self._root is None:
            raise RuntimeError("fit or partial_fit must train the tree first")

        X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features)")
        if X_array.dtype.kind not in "biuf":
            raise TypeError("X must contain numeric values")
        if X_array.shape[1] != self.n_features_in_: # the current tree uses its old feature count before the retraining
            if (
                not self._buffer_X
                or X_array.shape[1] != self._buffer_X[0].shape[1]
            ):
                raise ValueError(f"X must have {self.n_features_in_} features")
            X_array = X_array[:, :self.n_features_in_]

        X_array = X_array.astype(float, copy=True)
        if np.isinf(X_array).any():
            raise ValueError("X must not contain infinity")

        X_clean = np.where(np.isnan(X_array), self._fill_values, X_array)
        predictions = np.empty(len(X_clean), dtype=self.classes_.dtype)
        rows = np.arange(len(X_clean))
        self._predict_node(self._root, X_clean, rows, predictions)
        return predictions
    def partial_fit(self, X_chunk, y_chunk):
        """Add one chunk and retrain when the buffer is full.

        X_chunk has shape (n_samples, n_features); y_chunk has shape
        (n_samples,). Return self. Raise TypeError for unsupported types
        and ValueError for invalid shapes, empty chunks or infinity.
        """
        X_array = np.asarray(X_chunk)
        y_array = np.asarray(y_chunk)

        if X_array.ndim != 2:
            raise ValueError("X_chunk must be a 2D array")
        if X_array.dtype.kind not in "biuf":
            raise TypeError("X_chunk must contain numeric values")
        if X_array.shape[0] == 0 or X_array.shape[1] == 0:
            raise ValueError("X_chunk must contain samples and features")
        if y_array.ndim != 1 or len(y_array) != len(X_array):
            raise ValueError("y_chunk must have shape (n_samples,)")
        if y_array.dtype.kind not in "biufUS":
            raise TypeError("y_chunk must contain numeric or string labels")
        if y_array.dtype.kind == "f" and np.isnan(y_array).any():
            raise ValueError("y_chunk must not contain NaN")

        X_array = X_array.astype(float, copy=True)
        if np.isinf(X_array).any():
            raise ValueError("X_chunk must not contain infinity")
        if self._root is not None and X_array.shape[1] < self.n_features_in_:
            raise ValueError("X_chunk has the wrong number of features")
        if self._buffer_X:
            old_width = self._buffer_X[0].shape[1]
            if X_array.shape[1] < old_width:
                raise ValueError("X_chunk has the wrong number of features")
            if X_array.shape[1] > old_width: # pad old buffer chunks when new category is added to the colunm
                extra = X_array.shape[1] - old_width
                for i in range(len(self._buffer_X)):
                    old_chunk = self._buffer_X[i]
                    zeros = np.zeros((len(old_chunk), extra))
                    self._buffer_X[i] = np.column_stack((old_chunk, zeros))

        self._buffer_X.append(X_array)
        self._buffer_y.append(y_array.copy())

        buffered_samples = sum(len(chunk) for chunk in self._buffer_y)
        if buffered_samples >= self.buffer_size:
            X_train = np.vstack(self._buffer_X)
            y_train = np.concatenate(self._buffer_y)
            self.fit(X_train, y_train)

        return self