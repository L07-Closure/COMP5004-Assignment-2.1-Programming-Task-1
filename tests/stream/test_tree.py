import unittest

import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier


class DecisionTreeTests(unittest.TestCase):
    def setUp(self):
        self.X = [[0], [1], [2], [3]]
        self.y = [0, 0, 1, 1]

    def test_fit_and_predict(self):
        tree = DecisionTreeClassifier()
        tree.fit(self.X, self.y)
        np.testing.assert_array_equal(tree.predict([[0.5], [2.5]]), [0, 1])

    def test_entropy(self):
        tree = DecisionTreeClassifier(criterion="entropy")
        tree.fit(self.X, self.y)
        np.testing.assert_array_equal(tree.predict([[0.5], [2.5]]), [0, 1])

    def test_max_depth_zero(self):
        tree = DecisionTreeClassifier(max_depth=0)
        tree.fit(self.X, self.y)
        np.testing.assert_array_equal(tree.predict([[0], [3]]), [0, 0])

    def test_constant_feature(self):
        tree = DecisionTreeClassifier()
        tree.fit([[1], [1], [1]], [1, 1, 0])
        self.assertEqual(tree.predict([[1]])[0], 1)

    def test_string_labels(self):
        tree = DecisionTreeClassifier()
        tree.fit(self.X, ["no", "no", "yes", "yes"])
        np.testing.assert_array_equal(
            tree.predict([[0.5], [2.5]]), ["no", "yes"]
        )

    def test_nan_feature(self):
        tree = DecisionTreeClassifier()
        tree.fit([[0.0], [np.nan], [2.0]], [0, 1, 1])
        self.assertEqual(tree.predict([[np.nan]])[0], 1)

    def test_all_nan_column(self):
        tree = DecisionTreeClassifier()
        tree.fit(
            [[np.nan, 0], [np.nan, 1], [np.nan, 2], [np.nan, 3]],
            self.y,
        )
        np.testing.assert_array_equal(
            tree.predict([[np.nan, 0.5], [np.nan, 2.5]]), [0, 1]
        )

    def test_predict_before_fit(self):
        tree = DecisionTreeClassifier()
        with self.assertRaises(RuntimeError):
            tree.predict([[1]])

    def test_wrong_predict_shape(self):
        tree = DecisionTreeClassifier()
        tree.fit(self.X, self.y)
        with self.assertRaises(ValueError):
            tree.predict([[0, 1]])

    def test_buffer_waits_then_retrains(self):
        tree = DecisionTreeClassifier(buffer_size=4)
        tree.partial_fit(self.X[:2], self.y[:2])
        with self.assertRaises(RuntimeError):
            tree.predict([[0]])

        tree.partial_fit(self.X[2:], self.y[2:])
        np.testing.assert_array_equal(tree.predict([[0], [3]]), [0, 1])
        self.assertEqual(len(tree._buffer_X), 0)

    def test_old_tree_used_between_retrains(self):
        tree = DecisionTreeClassifier(buffer_size=4)
        tree.partial_fit(self.X, self.y)
        tree.partial_fit([[2.5]], [0])
        self.assertEqual(tree.predict([[2.5]])[0], 1)

    def test_invalid_buffer_size(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(buffer_size=0)

    def test_infinite_feature(self):
        tree = DecisionTreeClassifier()
        with self.assertRaises(ValueError):
            tree.fit([[0], [np.inf]], [0, 1])


if __name__ == "__main__":
    unittest.main()