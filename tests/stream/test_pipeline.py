import unittest

import numpy as np

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


class PipelineTests(unittest.TestCase):
    def make_pipeline(self, buffer_size=4):
        return Pipeline([
            ("impute", SimpleImputer()),
            ("model", DecisionTreeClassifier(buffer_size=buffer_size)),
        ])

    def test_fit_and_predict(self):
        pipe = self.make_pipeline()
        pipe.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
        np.testing.assert_array_equal(
            pipe.predict([[0.5], [2.5]]), [0, 1]
        )

    def test_partial_fit_and_predict(self):
        pipe = self.make_pipeline()
        pipe.partial_fit([[0], [1]], [0, 0])
        pipe.partial_fit([[2], [3]], [1, 1])
        np.testing.assert_array_equal(
            pipe.predict([[0.5], [2.5]]), [0, 1]
        )

    def test_transform_uses_fitted_steps(self):
        pipe = Pipeline([
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier()),
        ])
        pipe.fit([[1.0], [3.0]], [0, 1])
        np.testing.assert_allclose(pipe.transform([[np.nan]]), [[0.0]])

    def test_predict_waits_for_first_tree(self):
        pipe = self.make_pipeline(buffer_size=4)
        pipe.partial_fit([[0]], [0])
        with self.assertRaises(RuntimeError):
            pipe.predict([[0]])

    def test_duplicate_step_names(self):
        with self.assertRaises(ValueError):
            Pipeline([("step", SimpleImputer()), ("step", DecisionTreeClassifier())])

    def test_empty_steps(self):
        with self.assertRaises(ValueError):
            Pipeline([])

    def test_transformer_needs_partial_fit(self):
        pipe = Pipeline([
            ("bad", object()),
            ("model", DecisionTreeClassifier()),
        ])
        with self.assertRaises(TypeError):
            pipe.partial_fit([[1]], [0])

    def test_wrong_feature_count_reaches_transformer(self):
        pipe = self.make_pipeline()
        pipe.fit([[0], [1]], [0, 1])
        with self.assertRaises(ValueError):
            pipe.predict([[0, 1]])


if __name__ == "__main__":
    unittest.main()