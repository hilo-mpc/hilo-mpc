import unittest
import warnings

import numpy as np

from hilo_mpc.util.data import DataSet


class TestDataSetCreation(unittest.TestCase):
    """Tests for DataSet instantiation"""

    def test_dataset_creation_with_features_and_labels(self):
        """DataSet can be created with feature and label names"""
        ds = DataSet(features=['x1', 'x2'], labels=['y'])
        self.assertIsNotNone(ds)

    def test_dataset_empty_initially(self):
        """DataSet has zero samples when first created"""
        ds = DataSet(features=['x'], labels=['y'])
        self.assertEqual(len(ds), 0)

    def test_dataset_multiple_labels(self):
        """DataSet accepts multiple label names"""
        ds = DataSet(features=['x1', 'x2'], labels=['y1', 'y2', 'y3'])
        self.assertIsNotNone(ds)

    def test_dataset_with_add_time(self):
        """DataSet with add_time=True includes time vector"""
        ds = DataSet(features=['x'], labels=['y'], add_time=True)
        self.assertIsNotNone(ds)

    def test_dataset_features_property(self):
        """DataSet exposes feature names via .features"""
        ds = DataSet(features=['x1', 'x2'], labels=['y'])
        self.assertEqual(list(ds.features), ['x1', 'x2'])

    def test_dataset_labels_property(self):
        """DataSet exposes label names via .labels"""
        ds = DataSet(features=['x'], labels=['y1', 'y2'])
        self.assertEqual(list(ds.labels), ['y1', 'y2'])


class TestDataSetDataAddition(unittest.TestCase):
    """Tests for adding data to DataSet

    Note: add_data expects arrays in (n_features, n_samples) format,
    i.e. each column is a data sample.
    """

    def setUp(self):
        self.ds = DataSet(features=['x1', 'x2'], labels=['y'])

    def _make_data(self, n=3):
        """Return x: (2, n), y: (1, n) arrays"""
        x = np.arange(n * 2, dtype=float).reshape(2, n)
        y = np.arange(n, dtype=float).reshape(1, n)
        return x, y

    def test_add_data_from_arrays(self):
        """DataSet accepts (n_features, n_samples) arrays"""
        x, y = self._make_data(3)
        self.ds.add_data(x, y)
        self.assertEqual(len(self.ds), 3)

    def test_add_data_length_increases(self):
        """Adding data increments the sample count"""
        self.assertEqual(len(self.ds), 0)
        x, y = self._make_data(1)
        self.ds.add_data(x, y)
        self.assertEqual(len(self.ds), 1)
        self.ds.add_data(x, y)
        self.assertEqual(len(self.ds), 2)

    def test_raw_data_retrievable(self):
        """DataSet.raw_data returns a (x, y) tuple"""
        x, y = self._make_data(5)
        self.ds.add_data(x, y)
        raw = self.ds.raw_data
        self.assertIsInstance(raw, tuple)
        self.assertEqual(len(raw), 2)

    def test_raw_data_shape(self):
        """raw_data returns arrays with correct dimensions"""
        x, y = self._make_data(5)
        self.ds.add_data(x, y)
        x_raw, y_raw = self.ds.raw_data
        self.assertEqual(x_raw.shape[0], 2)  # 2 features
        self.assertEqual(y_raw.shape[0], 1)  # 1 label
        self.assertEqual(x_raw.shape[1], 5)  # 5 samples

    def test_add_data_values_preserved(self):
        """Raw data values match what was added"""
        x = np.array([[1., 2., 3.], [4., 5., 6.]])
        y = np.array([[7., 8., 9.]])
        self.ds.add_data(x, y)
        x_raw, y_raw = self.ds.raw_data
        np.testing.assert_allclose(x_raw, x)
        np.testing.assert_allclose(y_raw, y)


class TestDataSetSelection(unittest.TestCase):
    """Tests for DataSet train/test data selection"""

    def setUp(self):
        ds = DataSet(features=['x'], labels=['y'])
        n = 10
        x = np.arange(n, dtype=float).reshape(1, n)
        y = 2. * x
        ds.add_data(x, y)
        self.ds = ds
        self.n = n

    def test_select_train_data_downsample(self):
        """select_train_data with 'downsample' selects a subset of indices"""
        self.ds.select_train_data('downsample', downsample_factor=2)
        x_train, y_train = self.ds.train_data
        # Every 2nd sample selected from 10 → 5 training samples
        self.assertEqual(x_train.shape[1], 5)

    def test_select_test_data_downsample(self):
        """select_test_data with 'downsample' selects a subset of indices"""
        self.ds.select_test_data('downsample', downsample_factor=5)
        x_test, y_test = self.ds.test_data
        # Every 5th sample selected from 10 → 2 test samples
        self.assertEqual(x_test.shape[1], 2)

    def test_train_data_feature_count(self):
        """Training data has the correct number of features"""
        self.ds.select_train_data('downsample', downsample_factor=2)
        x_train, _ = self.ds.train_data
        self.assertEqual(x_train.shape[0], 1)  # 1 feature

    def test_train_data_label_count(self):
        """Training data has the correct number of labels"""
        self.ds.select_train_data('downsample', downsample_factor=2)
        _, y_train = self.ds.train_data
        self.assertEqual(y_train.shape[0], 1)  # 1 label

    def test_euclidean_distance_selection(self):
        """select_train_data with 'euclidean_distance' selects samples"""
        self.ds.select_train_data('euclidean_distance', distance_threshold=2.0)
        x_train, _ = self.ds.train_data
        # At least 1 sample selected
        self.assertGreater(x_train.shape[1], 0)

    def test_no_selection_returns_empty_train_data(self):
        """Without calling select_train_data, train indices list is empty"""
        self.assertEqual(len(self.ds._train_index), 0)


if __name__ == '__main__':
    unittest.main()
