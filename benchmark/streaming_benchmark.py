"""Compare two ways of counting accuracy across streamed chunks."""

from pathlib import Path
from time import perf_counter

import numpy as np

from numcompute.io import iter_csv_chunks
from numcompute_stream.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).resolve().parents[1] / "demo" / "stream_data.csv"


def load_chunks():
    """Read the demo CSV and prepare true and predicted labels."""
    chunks = []
    for rows in iter_csv_chunks(DATA_PATH, chunk_size=20, skip_header=1):
        y_true = rows[:, 2].astype(int)
        y_pred = (rows[:, 0] + rows[:, 1] > 1.0).astype(int)
        chunks.append((y_true, y_pred))
    return chunks


def loop_accuracy(chunks):
    """Count correct predictions one sample at a time."""
    correct = 0
    total = 0
    for y_true, y_pred in chunks:
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                correct += 1
            total += 1
    return correct / total


def vector_accuracy(chunks):
    """Count correct predictions with NumPy in each chunk."""
    correct = 0
    total = 0
    for y_true, y_pred in chunks:
        correct += np.count_nonzero(y_true == y_pred)
        total += len(y_true)
    return correct / total


def measure(function, chunks):
    """Return the median time for one complete stream pass, in ms."""
    for _ in range(5):
        function(chunks)

    times = []
    for _ in range(3):
        start = perf_counter()
        for _ in range(1000):
            function(chunks)
        times.append((perf_counter() - start) / 1000)

    return float(np.median(times)) * 1000

def compare_buffers():
    """Compare accuracy and retraining time for three buffer sizes."""
    for buffer_size in (20, 40, 80):
        total_retrain_ms = 0.0
        retrain_count = 0

        for _ in range(5):
            model = DecisionTreeClassifier(max_depth=5, buffer_size=buffer_size)
            correct = 0
            total = 0

            for number, rows in enumerate(
                iter_csv_chunks(DATA_PATH, chunk_size=20, skip_header=1),
                start=1,
            ):
                X_chunk = rows[:, :2]
                y_chunk = rows[:, 2].astype(int)

                if number >= 5:
                    y_pred = model.predict(X_chunk)
                    correct += int(np.count_nonzero(y_pred == y_chunk))
                    total += len(y_chunk)

                old_root = model._root
                start = perf_counter()
                model.partial_fit(X_chunk, y_chunk)
                elapsed_ms = (perf_counter() - start) * 1000

                if model._root is not old_root:
                    total_retrain_ms += elapsed_ms
                    retrain_count += 1

            accuracy = correct / total

        print(
            "buffer_size:", buffer_size,
            "accuracy:", round(accuracy, 3),
            "retrainings:", retrain_count // 5,
            "mean retraining call ms:",
            round(total_retrain_ms / retrain_count, 3),
            "total retraining call ms:",
            round(total_retrain_ms / 5, 3),
        )

if __name__ == "__main__":
    chunks = load_chunks()
    loop_result = loop_accuracy(chunks)
    vector_result = vector_accuracy(chunks)

    if loop_result != vector_result:
        raise AssertionError("accuracy results do not match")

    print("Chunks:", len(chunks))
    print("Accuracy:", loop_result)
    print("Loop ms:", round(measure(loop_accuracy, chunks), 4))
    print("Vector ms:", round(measure(vector_accuracy, chunks), 4))
    print("\nBuffer size comparison:")
    compare_buffers()