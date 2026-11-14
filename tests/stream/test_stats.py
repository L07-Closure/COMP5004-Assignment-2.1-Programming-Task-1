import unittest

import numpy as np

from numcompute_stream.stats import StreamingStats


class StreamingStatsTests(unittest.TestCase):
    def test_statistics_across_chunks(self):
        stats = StreamingStats()
        stats.update_stats([[1, 10], [3, 20]])
        stats.update_stats([[5, 30]])

        np.testing.assert_allclose(stats.mean(), [3, 20])
        np.testing.assert_allclose(stats.variance(), [8 / 3, 200 / 3])
        np.testing.assert_allclose(stats.quantile(0.5), [3, 20])

        histograms = stats.histogram(bins=2)
        self.assertEqual(len(histograms), 2)
        self.assertEqual(histograms[0][0].tolist(), [1, 2])
        self.assertEqual(histograms[1][0].tolist(), [1, 2])

    def test_nan_is_ignored_per_feature(self):
        stats = StreamingStats()
        stats.update_stats([[1, np.nan], [3, 10]])
        stats.update_stats([[np.nan, 30]])

        np.testing.assert_allclose(stats.mean(), [2, 20])
        np.testing.assert_allclose(stats.variance(), [1, 100])
        np.testing.assert_allclose(stats.quantile(0.5), [2, 20])

    def test_multiple_quantile_levels(self):
        stats = StreamingStats()
        stats.update_stats([[1, 10], [3, 20], [5, 30]])

        result = stats.quantile([0, 0.5, 1])
        self.assertEqual(result.shape, (3, 2))
        np.testing.assert_allclose(
            result, [[1, 10], [3, 20], [5, 30]]
        )

    def test_reset_allows_new_feature_count(self):
        stats = StreamingStats()
        stats.update_stats([[1, 2]])
        stats.reset()

        with self.assertRaises(RuntimeError):
            stats.mean()
        stats.update_stats([[9]])
        np.testing.assert_allclose(stats.mean(), [9])

    def test_rejects_invalid_chunks(self):
        stats = StreamingStats()
        stats.update_stats([[1, 2]])

        with self.assertRaises(ValueError):
            stats.update_stats([[1]])
        with self.assertRaises(ValueError):
            stats.update_stats([[1, np.inf]])
        with self.assertRaises(TypeError):
            stats.update_stats([["a", "b"]])

    def test_keeps_its_own_copy_of_chunks(self):
        chunk = np.array([[1.0], [3.0]])
        stats = StreamingStats()
        stats.update_stats(chunk)
        chunk[0, 0] = 100.0

        np.testing.assert_allclose(stats.mean(), [2])
        np.testing.assert_allclose(stats.quantile(0.5), [2])


if __name__ == "__main__":
    unittest.main()