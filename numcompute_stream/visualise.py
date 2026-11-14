import matplotlib.pyplot as plt
import numpy as np


def _metric_array(values):
    """Return a nonempty 1D array of finite metric values."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("metric values must be a nonempty 1D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("metric values must be finite")
    return array


def _finish_plot(fig, save_path, show):
    """Save or display a plot and return its Figure."""
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig


def plot_metric_over_time(
    metric_values, title, ylabel, save_path=None, show=True
):
    """Plot one metric across chunks; return a matplotlib Figure."""
    values = _metric_array(metric_values)
    chunks = np.arange(1, len(values) + 1)

    fig, ax = plt.subplots()
    ax.plot(chunks, values, marker="o")
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return _finish_plot(fig, save_path, show)


def compare_models(metric1, metric2, labels, save_path=None, show=True):
    """Compare two 1D metric histories; return a Figure."""
    first = _metric_array(metric1)
    second = _metric_array(metric2)
    if len(first) != len(second):
        raise ValueError("metric histories must have the same length")
    if not isinstance(labels, (list, tuple)) or len(labels) != 2:
        raise ValueError("labels must contain two model names")

    chunks = np.arange(1, len(first) + 1)
    fig, ax = plt.subplots()
    ax.plot(chunks, first, marker="o", label=labels[0])
    ax.plot(chunks, second, marker="o", label=labels[1])
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Metric value")
    ax.set_title("Model comparison")
    ax.legend()
    return _finish_plot(fig, save_path, show)


def plot_predictions_vs_ground_truth(
    y_true, y_pred, save_path=None, show=True
):
    """Plot 1D true and predicted labels for one chunk; return a Figure."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("labels must be 1D")
    if len(true) == 0 or len(true) != len(pred):
        raise ValueError("label arrays must have the same nonzero length")

    samples = np.arange(1, len(true) + 1)
    fig, ax = plt.subplots()
    ax.plot(samples, true, "o-", label="Ground truth")
    ax.plot(samples, pred, "x--", label="Prediction")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Class")
    ax.set_title("Predictions vs ground truth")
    ax.legend()
    return _finish_plot(fig, save_path, show)