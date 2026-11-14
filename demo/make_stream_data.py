from pathlib import Path

import numpy as np


rng = np.random.default_rng(7)
X = rng.random((240, 2))

threshold = np.ones(240)
threshold[120:] = 1.2
y = (X[:, 0] + X[:, 1] > threshold).astype(int)

data = np.column_stack((X, y))
path = Path(__file__).with_name("stream_data.csv")
np.savetxt(
    path,
    data,
    delimiter=",",
    header="feature_1,feature_2,label",
    comments="",
    fmt=["%.6f", "%.6f", "%d"],
)
print("Saved", len(data), "rows to", path)