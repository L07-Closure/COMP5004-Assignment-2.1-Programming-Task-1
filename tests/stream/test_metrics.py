import unittest

from numcompute_stream.metrics import StreamingAccuracy


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

if __name__ == "__main__":
    unittest.main()