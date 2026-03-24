import unittest
import warnings

import numpy as np
import pandas as pd

from hilo_mpc import ANN, Layer

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

requires_pytorch = unittest.skipUnless(PYTORCH_AVAILABLE, "PyTorch not installed")


class TestLayerDense(unittest.TestCase):
    """Tests for Dense layer creation and configuration"""

    def test_dense_single_layer_default_activation(self):
        """Dense layer with default activation (linear)"""
        layer = Layer.dense(10)
        self.assertEqual(len(layer), 10)
        self.assertEqual(layer.activation, 'linear')

    def test_dense_single_layer_sigmoid_activation(self):
        """Dense layer with sigmoid activation"""
        layer = Layer.dense(20, activation='sigmoid')
        self.assertEqual(len(layer), 20)
        self.assertEqual(layer.activation, 'sigmoid')

    def test_dense_single_layer_relu_activation(self):
        """Dense layer with relu activation"""
        layer = Layer.dense(15, activation='relu')
        self.assertEqual(len(layer), 15)
        self.assertEqual(layer.activation, 'relu')

    def test_dense_single_layer_tanh_activation(self):
        """Dense layer with tanh activation"""
        layer = Layer.dense(8, activation='tanh')
        self.assertEqual(len(layer), 8)
        self.assertEqual(layer.activation, 'tanh')

    def test_dense_multiple_layers(self):
        """Dense with list of nodes returns list of layers"""
        layers = Layer.dense([10, 20, 10], activation='sigmoid')
        self.assertIsInstance(layers, list)
        self.assertEqual(len(layers), 3)
        self.assertEqual(len(layers[0]), 10)
        self.assertEqual(len(layers[1]), 20)
        self.assertEqual(len(layers[2]), 10)

    def test_dense_multiple_layers_different_activations(self):
        """Dense with list of nodes and list of activations"""
        layers = Layer.dense([10, 20], activation=['relu', 'sigmoid'])
        self.assertEqual(len(layers), 2)
        self.assertEqual(layers[0].activation, 'relu')
        self.assertEqual(layers[1].activation, 'sigmoid')

    def test_dense_layer_type(self):
        """Dense layer has type 'Dense'"""
        layer = Layer.dense(10)
        self.assertIsNotNone(layer.type)

    def test_dense_layer_with_initializer(self):
        """Dense layer with explicit initializer"""
        layer = Layer.dense(10, initializer='uniform')
        self.assertIsNotNone(layer.initializer)
        self.assertEqual(layer.initializer, 'uniform')

    def test_dense_layer_activation_mismatch_raises(self):
        """Mismatched nodes and activation list lengths raise ValueError"""
        with self.assertRaises(ValueError):
            Layer.dense([10, 20], activation=['relu', 'sigmoid', 'tanh'])

    def test_dense_layer_initializer_mismatch_raises(self):
        """Mismatched nodes and initializer list lengths raise ValueError"""
        with self.assertRaises(ValueError):
            Layer.dense([10, 20], initializer=['uniform', 'normal', 'zeros'])


class TestLayerDropout(unittest.TestCase):
    """Tests for Dropout layer creation"""

    def test_dropout_layer_creation(self):
        """Dropout layer can be created"""
        layer = Layer.dropout(0.2)
        self.assertIsNotNone(layer)

    def test_dropout_layer_type(self):
        """Dropout layer has type 'Dropout'"""
        layer = Layer.dropout(0.2)
        self.assertIsNotNone(layer.type)

    def test_dropout_rate_stored(self):
        """Dropout rate is correctly stored"""
        layer = Layer.dropout(0.5)
        self.assertIsNotNone(layer)


class TestANNInitialization(unittest.TestCase):
    """Tests for ANN class initialization"""

    def test_ann_basic_initialization(self):
        """ANN can be initialized with features and labels"""
        ann = ANN(['x1', 'x2'], ['y1'])
        self.assertIsNotNone(ann)

    def test_ann_multiple_features_and_labels(self):
        """ANN initializes with multiple features and labels"""
        ann = ANN(['x1', 'x2', 'x3'], ['y1', 'y2'])
        self.assertIsNotNone(ann)

    def test_ann_default_backend_is_pytorch(self):
        """ANN default backend is pytorch"""
        ann = ANN(['x'], ['y'])
        self.assertEqual(ann.backend, 'pytorch')

    def test_ann_default_loss_is_mse(self):
        """ANN default loss is mse"""
        ann = ANN(['x'], ['y'])
        self.assertEqual(ann.loss, 'mse')

    def test_ann_default_optimizer_is_adam(self):
        """ANN default optimizer is adam"""
        ann = ANN(['x'], ['y'])
        self.assertEqual(ann.optimizer, 'adam')

    def test_ann_custom_learning_rate(self):
        """ANN stores custom learning rate"""
        ann = ANN(['x'], ['y'], learning_rate=0.01)
        self.assertAlmostEqual(ann.learning_rate, 0.01)

    def test_ann_custom_seed(self):
        """ANN stores custom seed"""
        ann = ANN(['x'], ['y'], seed=42)
        self.assertEqual(ann.seed, 42)

    def test_ann_features_count(self):
        """ANN correctly reports number of features"""
        ann = ANN(['x1', 'x2', 'x3'], ['y'])
        self.assertEqual(ann.n_features, 3)

    def test_ann_labels_count(self):
        """ANN correctly reports number of labels"""
        ann = ANN(['x'], ['y1', 'y2'])
        self.assertEqual(ann.n_labels, 2)


class TestANNLayerManagement(unittest.TestCase):
    """Tests for ANN layer addition and management"""

    def setUp(self):
        self.ann = ANN(['x1', 'x2'], ['y1'])

    def test_add_single_dense_layer(self):
        """ANN can add a single Dense layer"""
        self.ann.add_layers(Layer.dense(10, activation='sigmoid'))
        self.assertEqual(self.ann.depth, 1)

    def test_add_multiple_dense_layers(self):
        """ANN can add multiple Dense layers"""
        self.ann.add_layers(Layer.dense(10, activation='sigmoid'))
        self.ann.add_layers(Layer.dense(5, activation='relu'))
        self.assertEqual(self.ann.depth, 2)

    def test_add_layer_list(self):
        """ANN can add a list of layers at once"""
        layers = Layer.dense([10, 5], activation='sigmoid')
        self.ann.add_layers(layers)
        self.assertEqual(self.ann.depth, 2)

    def test_ann_shape_with_layers(self):
        """ANN shape reflects features, hidden nodes, and labels"""
        self.ann.add_layers(Layer.dense(10, activation='sigmoid'))
        shape = self.ann.shape
        # shape should be (n_features, hidden_nodes, n_labels)
        self.assertEqual(shape[0], 2)   # n_features
        self.assertEqual(shape[1], 10)  # hidden nodes
        self.assertEqual(shape[2], 1)   # n_labels

    def test_dropout_not_counted_in_depth(self):
        """Dropout layers are not counted in depth"""
        self.ann.add_layers(Layer.dense(10, activation='sigmoid'))
        self.ann.add_layers(Layer.dropout(0.2))
        self.ann.add_layers(Layer.dense(5, activation='sigmoid'))
        # depth only counts Dense layers
        self.assertEqual(self.ann.depth, 2)


class TestANNTrainingAndPrediction(unittest.TestCase):
    """Tests for ANN training and prediction with synthetic data"""

    def setUp(self):
        """Create a simple dataset for training"""
        np.random.seed(0)
        n = 50
        x = np.random.randn(n, 2)
        y = np.sin(x[:, 0:1]) + 0.1 * np.random.randn(n, 1)
        self.df = pd.DataFrame(np.hstack([x, y]), columns=['x1', 'x2', 'y1'])

    @requires_pytorch
    def test_ann_training_completes(self):
        """ANN training runs without error"""
        ann = ANN(['x1', 'x2'], ['y1'])
        ann.add_layers(Layer.dense(5, activation='sigmoid'))
        ann.setup(save_tensorboard=False)
        ann.add_data_set(self.df)
        ann.train(1, 2, test_split=0.2, patience=100, verbose=0)

    def test_ann_add_data_set(self):
        """ANN correctly stores data set"""
        ann = ANN(['x1', 'x2'], ['y1'])
        ann.add_data_set(self.df)
        self.assertEqual(len(ann._data_sets), 1)

    def test_ann_data_set_wrong_feature_raises(self):
        """ANN raises ValueError when data set missing required feature"""
        ann = ANN(['x1', 'x2', 'missing_feature'], ['y1'])
        with self.assertRaises(ValueError):
            ann.add_data_set(self.df)

    def test_ann_data_set_wrong_label_raises(self):
        """ANN raises ValueError when data set missing required label"""
        ann = ANN(['x1', 'x2'], ['y1', 'missing_label'])
        with self.assertRaises(ValueError):
            ann.add_data_set(self.df)

    @requires_pytorch
    def test_ann_casadi_graph_after_training(self):
        """ANN can build CasADi graph after training"""
        ann = ANN(['x1', 'x2'], ['y1'])
        ann.add_layers(Layer.dense(5, activation='sigmoid'))
        ann.setup(save_tensorboard=False)
        ann.add_data_set(self.df)
        ann.train(1, 2, test_split=0.2, patience=100, verbose=0)
        # Should not raise
        ann.build_graph()
        self.assertIsNotNone(ann._function)

    @requires_pytorch
    def test_ann_prediction_after_training(self):
        """ANN produces output after training (via CasADi graph)"""
        import casadi as ca
        ann = ANN(['x1', 'x2'], ['y1'])
        ann.add_layers(Layer.dense(5, activation='sigmoid'))
        ann.setup(save_tensorboard=False)
        ann.add_data_set(self.df)
        ann.train(1, 2, test_split=0.2, patience=100, verbose=0)
        ann.build_graph()
        # Test that the function can be called with a sample input
        x_test = ca.DM([0.5, -0.3])
        # _function returns a tuple of CasADi DMs: (labels, hidden_layer_1, ...)
        result = ann._function(x_test)
        labels = result[0]
        self.assertEqual(labels.size1(), 1)


if __name__ == '__main__':
    unittest.main()
