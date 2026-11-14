"""Compare two ways of counting accuracy across streamed chunks."""

from pathlib import Path
from time import perf_counter

import numpy as np

from numcompute.io import iter_csv_chunks


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