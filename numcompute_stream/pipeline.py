class Pipeline:
    """Chain incremental transformers and a final model."""

    def __init__(self, steps):
        """Store nonempty, uniquely named steps in their execution order."""
        if not isinstance(steps, (list, tuple)) or not steps:
            raise ValueError("steps must be a nonempty list")

        names = []
        for item in steps:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("each step must be a (name, component) pair")
            name, _ = item
            if not isinstance(name, str) or not name:
                raise ValueError("step names must be nonempty strings")
            if name in names:
                raise ValueError("step names must be unique")
            names.append(name)

        self.steps = list(steps)

    def fit(self, X, y):
        """Fit transformers and model on 2D X and 1D y; return self."""
        data = X
        for _, step in self.steps[:-1]:
            step.fit(data, y)
            data = step.transform(data)
        self.steps[-1][1].fit(data, y)
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """Update each step from one chunk and return self.

        Transformers update before transforming the current chunk.
        The final model receives the transformed chunk.
        """
        data = X_chunk
        for _, step in self.steps[:-1]:
            if not hasattr(step, "partial_fit") or not hasattr(step, "transform"):
                raise TypeError("transformers need partial_fit and transform")
            step.partial_fit(data, y_chunk)
            data = step.transform(data)

        model = self.steps[-1][1]
        if not hasattr(model, "partial_fit"):
            raise TypeError("the final model needs partial_fit")
        model.partial_fit(data, y_chunk)
        return self

    def transform(self, X):
        """Apply preprocessing steps to 2D X and return the features."""
        data = X
        for _, step in self.steps[:-1]:
            data = step.transform(data)
        return data

    def predict(self, X):
        """Transform 2D X and return the model's 1D predictions."""
        data = self.transform(X)
        model = self.steps[-1][1]
        if not hasattr(model, "predict"):
            raise TypeError("the final model needs predict")
        return model.predict(data)