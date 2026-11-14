import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from numcompute_stream.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


class VisualiseTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_metric_plot_can_be_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "accuracy.png"
            fig = plot_metric_over_time(
                [0.5, 0.75], "Accuracy", "Score",
                save_path=path, show=False,
            )

            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(fig.axes[0].get_title(), "Accuracy")
            self.assertEqual(list(fig.axes[0].lines[0].get_xdata()), [1, 2])

    def test_model_comparison_has_two_named_lines(self):
        fig = compare_models(
            [0.5, 0.7], [0.4, 0.8], ["small", "large"], show=False
        )

        lines = fig.axes[0].lines
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].get_label(), "small")
        self.assertEqual(lines[1].get_label(), "large")

    def test_predictions_show_true_and_predicted_labels(self):
        fig = plot_predictions_vs_ground_truth(
            ["cat", "dog"], ["cat", "cat"], show=False
        )

        lines = fig.axes[0].lines
        self.assertEqual(list(lines[0].get_ydata()), ["cat", "dog"])
        self.assertEqual(list(lines[1].get_ydata()), ["cat", "cat"])

    def test_invalid_shapes_and_lengths(self):
        with self.assertRaises(ValueError):
            plot_metric_over_time([[0.5, 0.7]], "A", "Score", show=False)
        with self.assertRaises(ValueError):
            compare_models([0.5], [0.5, 0.7], ["A", "B"], show=False)
        with self.assertRaises(ValueError):
            plot_predictions_vs_ground_truth([0, 1], [0], show=False)


if __name__ == "__main__":
    unittest.main()