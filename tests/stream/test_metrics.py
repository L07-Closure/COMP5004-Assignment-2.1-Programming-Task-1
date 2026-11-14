import unittest

from numcompute_stream.metrics import (
    StreamingAccuracy,
    StreamingClassificationMetrics,
    StreamingConfusionMatrix,
)


class StreamingAccuracyTests(unittest.TestCase):
    def test_cumulative_and_rolling_accuracy_across_chunks(self):
        metric = StreamingAccuracy(window_size=3)
        metric.update([1, 0], [1, 1])
        metric.update([1, 1], [1, 0])

        self.assertEqual(metric.result(), 0.5)
        self.assertAlmostEqual(metric.window_result(), 1 / 3)

    def test_window_keeps_only_recent_samples(self):
        metric = StreamingAccuracy(window_size=2)
        metric.update([0, 1, 1, 0], [1, 1, 1, 0])

        self.assertEqual(metric.result(), 0.75)
        self.assertEqual(metric.window_result(), 1.0)

    def test_reset_starts_new_counts(self):
        metric = StreamingAccuracy(window_size=2)
        metric.update([0, 1], [1, 1])
        metric.reset()
        metric.update([0], [0])

        self.assertEqual(metric.result(), 1.0)
        self.assertEqual(metric.window_result(), 1.0)

    def test_string_labels(self):
        metric = StreamingAccuracy()
        metric.update(["cat", "dog"], ["cat", "cat"])
        self.assertEqual(metric.result(), 0.5)

    def test_rejects_invalid_chunk_shapes_and_lengths(self):
        metric = StreamingAccuracy()
        for y_true, y_pred in (
            ([[0, 1]], [[0, 1]]),
            ([0, 1], [0]),
            ([], []),
        ):
            with self.subTest(y_true=y_true):
                with self.assertRaises(ValueError):
                    metric.update(y_true, y_pred)

    def test_rejects_nonfinite_numeric_labels(self):
        metric = StreamingAccuracy()
        with self.assertRaises(ValueError):
            metric.update([0, float("nan")], [0, 1])
        with self.assertRaises(ValueError):
            metric.update([0, 1], [0, float("inf")])

    def test_window_and_empty_state_errors(self):
        with self.assertRaises(ValueError):
            StreamingAccuracy(window_size=0)

        metric = StreamingAccuracy()
        with self.assertRaises(ValueError):
            metric.result()
        with self.assertRaises(ValueError):
            metric.window_result()

        windowed = StreamingAccuracy(window_size=2)
        with self.assertRaises(ValueError):
            windowed.window_result()

class StreamingConfusionMatrixTests(unittest.TestCase):
    def test_cumulative_counts_and_new_labels(self):
        metric = StreamingConfusionMatrix()
        metric.update([2], [2])
        metric.update([0, 2], [2, 0])

        matrix, labels = metric.result()
        self.assertEqual(labels.tolist(), [0, 2])
        self.assertEqual(matrix.tolist(), [[0, 1], [1, 1]])

    def test_rolling_counts_across_chunks(self):
        metric = StreamingConfusionMatrix(window_size=2)
        metric.update([0, 1, 1], [0, 0, 1])
        metric.update([0], [1])

        matrix, labels = metric.window_result()
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual(matrix.tolist(), [[0, 1], [0, 1]])

    def test_string_labels(self):
        metric = StreamingConfusionMatrix()
        metric.update(["cat", "dog"], ["dog", "dog"])

        matrix, labels = metric.result()
        self.assertEqual(labels.tolist(), ["cat", "dog"])
        self.assertEqual(matrix.tolist(), [[0, 1], [0, 1]])

    def test_reset_clears_old_counts(self):
        metric = StreamingConfusionMatrix()
        metric.update([0], [0])
        metric.reset()
        metric.update([1], [0])

        matrix, labels = metric.result()
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual(matrix.tolist(), [[0, 0], [1, 0]])

    def test_rejects_invalid_chunks(self):
        metric = StreamingConfusionMatrix()
        with self.assertRaises(ValueError):
            metric.update([[0]], [[0]])
        with self.assertRaises(ValueError):
            metric.update([0, 1], [0])
        with self.assertRaises(ValueError):
            metric.update([0, float("nan")], [0, 1])
        with self.assertRaises(TypeError):
            metric.update([0], ["0"])

    def test_empty_state_and_window_errors(self):
        with self.assertRaises(ValueError):
            StreamingConfusionMatrix(window_size=0)

        metric = StreamingConfusionMatrix()
        with self.assertRaises(ValueError):
            metric.result()
        with self.assertRaises(ValueError):
            metric.window_result()

        windowed = StreamingConfusionMatrix(window_size=2)
        with self.assertRaises(ValueError):
            windowed.window_result()
class StreamingClassificationMetricsTests(unittest.TestCase):
    def test_binary_scores_across_chunks_and_window(self):
        metric = StreamingClassificationMetrics(window_size=2)
        metric.update([0, 1], [0, 0])
        metric.update([1, 0], [1, 1])

        self.assertEqual(
            metric.result(),
            {"precision": 0.5, "recall": 0.5, "f1": 0.5},
        )
        recent = metric.window_result()
        self.assertEqual(recent["precision"], 0.5)
        self.assertEqual(recent["recall"], 1.0)
        self.assertAlmostEqual(recent["f1"], 2 / 3)

    def test_macro_scores_with_three_classes(self):
        metric = StreamingClassificationMetrics(average="macro")
        metric.update([0, 1, 2, 2], [0, 0, 2, 1])

        scores = metric.result()
        self.assertAlmostEqual(scores["precision"], 0.5)
        self.assertAlmostEqual(scores["recall"], 0.5)
        self.assertAlmostEqual(scores["f1"], 4 / 9)

    def test_custom_positive_label(self):
        metric = StreamingClassificationMetrics(pos_label="yes")
        metric.update(["yes", "no", "yes"], ["yes", "yes", "no"])

        self.assertEqual(
            metric.result(),
            {"precision": 0.5, "recall": 0.5, "f1": 0.5},
        )

    def test_zero_scores_and_reset(self):
        metric = StreamingClassificationMetrics()
        metric.update([0, 0], [0, 0])
        self.assertEqual(
            metric.result(),
            {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        )

        metric.reset()
        with self.assertRaises(ValueError):
            metric.result()

    def test_invalid_options_and_chunks(self):
        with self.assertRaises(ValueError):
            StreamingClassificationMetrics(average="weighted")

        metric = StreamingClassificationMetrics()
        with self.assertRaises(ValueError):
            metric.update([0, 1], [0])
        with self.assertRaises(ValueError):
            metric.window_result()

if __name__ == "__main__":
    unittest.main()