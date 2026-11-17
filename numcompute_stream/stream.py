import sys

import numpy as np

from numcompute_stream.metrics import StreamingAccuracy
from numcompute_stream.pipeline import Pipeline


class StreamTrainer:
    """Train a pipeline by chunks and log its prediction accuracy."""

    def __init__(self, pipeline):
        """Store a Pipeline whose final step is a predictive model."""
        if not isinstance(pipeline, Pipeline):
            raise TypeError("pipeline must be a Pipeline")
        self.pipeline = pipeline
        self.accuracy = StreamingAccuracy()
        self.logs = []

    def fit_chunk(self, X_chunk, y_chunk):
        """Update the pipeline with 2D X_chunk and 1D y_chunk."""
        self.pipeline.partial_fit(X_chunk, y_chunk)
        return self

    def _model_memory_bytes(self):
        """Estimate memory used by pipeline steps, arrays and tree nodes."""
        total = 0 # this is an estimate, not full python memory count

        for _, step in self.pipeline.steps:
            total += sys.getsizeof(step)

            for value in vars(step).values(): 
                if isinstance(value, np.ndarray):
                    total += value.nbytes
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, np.ndarray):
                            total += item.nbytes

            nodes = [getattr(step, "_root", None)] # the fitted tree include its nodes
            while nodes:
                node = nodes.pop()
                if node is not None:
                    total += sys.getsizeof(node)
                    nodes.extend((node.left, node.right))

        return total

    def score_chunk(self, X_chunk, y_chunk):
        """Score a chunk and log accuracy, cumulative accuracy and memory."""
        predictions = self.pipeline.predict(X_chunk)
        true_labels = np.asarray(y_chunk)
        self.accuracy.update(true_labels, predictions)

        chunk_accuracy = float(np.mean(true_labels == predictions))
        self.logs.append({
            "chunk": len(self.logs) + 1,
            "n_samples": len(true_labels),
            "accuracy": chunk_accuracy,
            "cumulative_accuracy": self.accuracy.result(),
            "model_memory_bytes": self._model_memory_bytes(),
        })
        return chunk_accuracy