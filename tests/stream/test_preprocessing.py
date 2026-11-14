import unittest

import numpy as np

from numcompute_stream.preprocessing import StandardScaler, SimpleImputer
from numcompute_stream.preprocessing import OneHotEncoder

class StandardScalerTests(unittest.TestCase):
    def test_fit_transform(self):
        scaler = StandardScaler()
        result = scaler.fit_transform([[1.0], [3.0], [5.0]])
        expected = [-np.sqrt(1.5), 0.0, np.sqrt(1.5)]
        np.testing.assert_allclose(result[:, 0], expected)

    def test_chunks_match_full_fit(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        full = StandardScaler().fit(X)
        streamed = StandardScaler()
        streamed.partial_fit(X[:2])
        streamed.partial_fit(X[2:])
        np.testing.assert_allclose(streamed.mean_, full.mean_)
        np.testing.assert_allclose(streamed.transform(X), full.transform(X))

    def test_nan_is_ignored(self):
        scaler = StandardScaler()
        scaler.partial_fit([[1.0, 2.0], [3.0, np.nan]])
        scaler.partial_fit([[5.0, 4.0]])
        np.testing.assert_array_equal(scaler.count_, [3, 2])
        np.testing.assert_allclose(scaler.mean_, [3.0, 3.0])

    def test_zero_variance(self):
        scaler = StandardScaler().fit([[2.0], [2.0]])
        np.testing.assert_allclose(scaler.transform([[2.0], [3.0]]), [[0.0], [1.0]])

    def test_all_nan_column(self):
        scaler = StandardScaler().fit([[np.nan, 1.0], [np.nan, 2.0]])
        result = scaler.transform([[np.nan, 1.0]])
        self.assertTrue(np.isnan(result[0, 0]))
        self.assertTrue(np.isfinite(result[0, 1]))

    def test_transform_before_fit(self):
        with self.assertRaises(RuntimeError):
            StandardScaler().transform([[1.0]])

    def test_wrong_input_shape(self):
        scaler = StandardScaler().fit([[1.0], [2.0]])
        with self.assertRaises(ValueError):
            scaler.transform([1.0, 2.0])

    def test_feature_count_cannot_change(self):
        scaler = StandardScaler().partial_fit([[1.0], [2.0]])
        with self.assertRaises(ValueError):
            scaler.partial_fit([[1.0, 2.0]])

    def test_fit_resets_old_statistics(self):
        scaler = StandardScaler().fit([[1.0], [3.0]])
        scaler.fit([[10.0], [20.0]])
        np.testing.assert_allclose(scaler.mean_, [15.0])
        np.testing.assert_array_equal(scaler.count_, [2])

    def test_empty_chunk(self):
        with self.assertRaises(ValueError):
            StandardScaler().partial_fit(np.empty((0, 1)))

    def test_infinite_value(self):
        with self.assertRaises(ValueError):
            StandardScaler().partial_fit([[np.inf]])

class SimpleImputerTests(unittest.TestCase):
    def test_mean_fill(self):
        imputer = SimpleImputer().fit(
            [[1.0, np.nan], [3.0, 10.0], [5.0, 14.0]]
        )
        np.testing.assert_allclose(
            imputer.transform([[np.nan, np.nan]]), [[3.0, 12.0]]
        )

    def test_chunks_match_full_fit(self):
        X = np.array([[1.0, np.nan], [3.0, 10.0], [5.0, 14.0]])
        full = SimpleImputer().fit(X)
        streamed = SimpleImputer()
        streamed.partial_fit(X[:2])
        streamed.partial_fit(X[2:])
        np.testing.assert_allclose(streamed.mean_, full.mean_)
        np.testing.assert_allclose(streamed.transform(X), full.transform(X))

    def test_constant_fill(self):
        imputer = SimpleImputer(strategy="constant", fill_value=-1)
        imputer.fit([[np.nan, 2.0]])
        np.testing.assert_allclose(
            imputer.transform([[np.nan, 2.0]]), [[-1.0, 2.0]]
        )

    def test_all_nan_column_then_new_value(self):
        imputer = SimpleImputer(fill_value=0).partial_fit([[np.nan]])
        np.testing.assert_allclose(imputer.transform([[np.nan]]), [[0.0]])
        imputer.partial_fit([[4.0]])
        np.testing.assert_allclose(imputer.transform([[np.nan]]), [[4.0]])

    def test_fit_resets_estimate(self):
        imputer = SimpleImputer().fit([[1.0], [3.0]])
        imputer.fit([[10.0], [20.0]])
        np.testing.assert_allclose(imputer.mean_, [15.0])

    def test_transform_before_fit(self):
        with self.assertRaises(RuntimeError):
            SimpleImputer().transform([[np.nan]])

    def test_wrong_feature_count(self):
        imputer = SimpleImputer().fit([[1.0]])
        with self.assertRaises(ValueError):
            imputer.transform([[1.0, 2.0]])

    def test_empty_chunk(self):
        with self.assertRaises(ValueError):
            SimpleImputer().partial_fit(np.empty((0, 1)))

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            SimpleImputer(strategy="median")

    def test_infinite_value(self):
        with self.assertRaises(ValueError):
            SimpleImputer().partial_fit([[np.inf]])

class OneHotEncoderTests(unittest.TestCase):
    def test_new_category_is_added(self):
        encoder = OneHotEncoder().partial_fit([["red"], ["blue"]])
        encoder.partial_fit([["green"]])
        self.assertEqual(
            encoder.categories_[0].tolist(), ["red", "blue", "green"]
        )
        np.testing.assert_array_equal(
            encoder.transform([["green"]]), [[0.0, 0.0, 1.0]]
        )

    def test_existing_category_keeps_its_position(self):
        encoder = OneHotEncoder().partial_fit([["red"], ["blue"]])
        before = encoder.transform([["red"]])
        encoder.partial_fit([["green"]])
        np.testing.assert_array_equal(
            encoder.transform([["red"]])[:, :2], before
        )

    def test_two_columns_expand(self):
        encoder = OneHotEncoder().partial_fit([["red", "small"]])
        encoder.partial_fit([["blue", "large"]])
        np.testing.assert_array_equal(
            encoder.transform([["red", "large"]]),
            [[1.0, 0.0, 0.0, 1.0]],
        )

    def test_none_and_nan_are_one_category(self):
        encoder = OneHotEncoder().partial_fit([[None]])
        encoder.partial_fit([[np.nan]])
        self.assertEqual(encoder.transform([[None]]).shape, (1, 1))

    def test_unknown_can_be_ignored_before_update(self):
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.partial_fit([["red"]])
        np.testing.assert_array_equal(
            encoder.transform([["blue"]]), [[0.0]]
        )

    def test_wrong_feature_count(self):
        encoder = OneHotEncoder().partial_fit([["red"]])
        with self.assertRaises(ValueError):
            encoder.partial_fit([["blue", "large"]])

    def test_mixed_types_across_chunks(self):
        encoder = OneHotEncoder().partial_fit([["red"]])
        with self.assertRaises(TypeError):
            encoder.partial_fit([[1]])

    def test_empty_chunk(self):
        encoder = OneHotEncoder()
        with self.assertRaises(ValueError):
            encoder.partial_fit(np.empty((0, 1), dtype=object))

    def test_fit_resets_categories(self):
        encoder = OneHotEncoder().partial_fit([["red"]])
        encoder.fit([["blue"]])
        self.assertEqual(encoder.categories_[0].tolist(), ["blue"])

    def test_new_category_keeps_other_column_positions(self):
            encoder = OneHotEncoder().partial_fit(
                [["red", "small"], ["blue", "small"]]
            )
            before = encoder.transform([["red", "small"]])
    
            encoder.partial_fit([["green", "small"]])
    
            np.testing.assert_array_equal(
                encoder.transform([["red", "small"]])[:, :3], before
            )
            np.testing.assert_array_equal(
                encoder.transform([["green", "small"]]),
                [[0.0, 0.0, 1.0, 1.0]],
            )
            
if __name__ == "__main__":
    unittest.main()