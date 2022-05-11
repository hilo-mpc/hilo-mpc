from typing import Tuple
from unittest import TestCase, skip
import warnings

import numpy as np

from hilo_mpc import GP, Mean, Kernel


# TODO: Try to improve numerical stability of GPs
# TODO: GPs with multiple features (automatic relevance detection (ard) True and False)
# TODO: GPs with multiple labels -> MultiOutputGP (or maybe GPArray)
# TODO: Hyperprior for multiple features
# TODO: GPs with constrained hyperparameters
# TODO: GPs with different inference methods
# TODO: GPs with different likelihoods
class GaussianProcessInitialization(TestCase):
    """"""
    def test_gaussian_process_features_and_labels_no_list(self) -> None:
        """

        :return:
        """
        gp = GP('x', 'y')
        self.assertIsInstance(gp.features, list)
        self.assertIsInstance(gp.labels, list)

    def test_gaussian_process_more_than_one_label(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            GP(['x'], ['y', 'z'])
        # FIXME: Rename 'MultiOutputGP' to 'GPArray'
        self.assertEqual(str(context.exception), "Training a GP on multiple labels is not supported. Please use "
                                                 "'MultiOutputGP' to train GPs on multiple labels.")

    def test_gaussian_process_check_features_and_labels(self) -> None:
        """

        :return:
        """
        gp = GP(['x', 'y'], 'z')

        self.assertEqual(gp.n_features, 2)
        self.assertEqual(gp.n_labels, 1)
        self.assertEqual(gp.features, ['x', 'y'])
        self.assertEqual(gp.labels, ['z'])

    def test_gaussian_process_initial_training_data(self) -> None:
        """

        :return:
        """
        gp = GP(['x', 'y'], 'z')

        X_train = gp.X_train
        y_train = gp.y_train

        self.assertEqual(X_train.values.size, 0)
        self.assertTrue(X_train.SX.is_empty())
        self.assertTrue(X_train.MX.is_empty())
        self.assertEqual(y_train.values.size, 0)
        self.assertTrue(y_train.SX.is_empty())
        self.assertTrue(y_train.MX.is_empty())

    def test_gaussian_process_likelihood_string_gaussian(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.likelihood import Gaussian

        gp = GP(['x', 'y'], ['z'], likelihood='Gaussian')
        self.assertIsInstance(gp.likelihood, Gaussian)
        self.assertEqual(gp.likelihood.name, 'Gaussian')

    def test_gaussian_process_likelihood_string_logistic(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.likelihood import Logistic

        # FIXME: Once the logistic likelihood is implemented, we can remove the with statement
        with self.assertRaises(NotImplementedError):
            gp = GP(['x', 'y'], ['z'], likelihood='Logistic')
            self.assertIsInstance(gp.likelihood, Logistic)
            self.assertEqual(gp.likelihood.name, 'Logistic')

    def test_gaussian_process_likelihood_string_laplacian(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.likelihood import Laplacian

        # FIXME: Once the Laplacian likelihood is implemented, we can remove the with statement
        with self.assertRaises(NotImplementedError):
            gp = GP(['x', 'y'], ['z'], likelihood='Laplacian')
            self.assertIsInstance(gp.likelihood, Laplacian)
            self.assertEqual(gp.likelihood.name, 'Laplacian')

    def test_gaussian_process_likelihood_string_students_t(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.likelihood import StudentsT

        # FIXME: Once the student's t likelihood is implemented, we can remove the with statement
        with self.assertRaises(NotImplementedError):
            gp = GP(['x', 'y'], ['z'], likelihood='Students t')
            self.assertIsInstance(gp.likelihood, StudentsT)
            self.assertEqual(gp.likelihood.name, 'Students_T')

    def test_gaussian_process_likelihood_string_not_recognized(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            GP(['x', 'y'], ['z'], likelihood='Gumbel')
        self.assertEqual(str(context.exception), "Likelihood 'Gumbel' not recognized")

    def test_gaussian_process_inference_string_exact(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.inference import ExactInference

        gp = GP(['x', 'y'], ['z'], inference='exact')
        self.assertIsInstance(gp.inference, ExactInference)

    def test_gaussian_process_inference_string_laplace(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.inference import Laplace

        # FIXME: Once the Laplace inference is implemented, we can remove the with statement
        with self.assertRaises(TypeError):
            gp = GP(['x', 'y'], ['z'], inference='Laplace')
            self.assertIsInstance(gp.inference, Laplace)

    def test_gaussian_process_inference_string_expectation_propagation(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.inference import ExpectationPropagation

        # FIXME: Once the expectation propagation inference is implemented, we can remove the with statement
        with self.assertRaises(TypeError):
            gp = GP(['x', 'y'], ['z'], inference='Expectation propagation')
            self.assertIsInstance(gp.inference, ExpectationPropagation)

    def test_gaussian_process_inference_string_variational_bayes(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.inference import VariationalBayes

        # FIXME: Once the variational Bayes inference is implemented, we can remove the with statement
        with self.assertRaises(TypeError):
            gp = GP(['x', 'y'], ['z'], inference='Variational Bayes')
            self.assertIsInstance(gp.inference, VariationalBayes)

    def test_gaussian_process_inference_string_kullback_leibler(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.inference import KullbackLeibler

        # FIXME: Once the Kullback Leibler inference is implemented, we can remove the with statement
        with self.assertRaises(TypeError):
            gp = GP(['x', 'y'], ['z'], inference='Kullback Leibler')
            self.assertIsInstance(gp.inference, KullbackLeibler)

    def test_gaussian_process_inference_string_not_recognized(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            GP(['x', 'y'], ['z'], inference='Sampling')
        self.assertEqual(str(context.exception), "Inference 'Sampling' not recognized")

    # def test_gaussian_process_noise_variance_hyperprior_laplace(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # FIXME: Fix behavior when supplying hyperpriors this way (both for PositiveParameter and Parameter class)
    #     gp = GP(['x', 'y'], ['z'], hyperprior='Laplace', hyperprior_parameters={'mean': 0., 'variance': 1.})
    #     # TODO: Finish once bugs are fixed

    def test_gaussian_process_default_options(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.inference import ExactInference
        from hilo_mpc.modules.machine_learning.gp.likelihood import Gaussian
        from hilo_mpc.modules.machine_learning.gp.mean import ZeroMean
        from hilo_mpc.modules.machine_learning.gp.kernel import SquaredExponentialKernel

        gp = GP(['x', 'y'], 'z')

        self.assertIsInstance(gp.inference, ExactInference)
        self.assertIsInstance(gp.likelihood, Gaussian)
        self.assertIsInstance(gp.mean, ZeroMean)
        self.assertIsInstance(gp.kernel, SquaredExponentialKernel)

        self.assertEqual(len(gp.hyperparameters), 3)
        self.assertEqual(len(gp.hyperparameter_names), 3)

        self.assertIn(gp.noise_variance, gp.hyperparameters)
        self.assertEqual(gp.noise_variance, gp.hyperparameters[0])
        self.assertIn(gp.noise_variance.name, gp.hyperparameter_names)
        self.assertEqual(gp.noise_variance.name, gp.hyperparameter_names[0])
        self.assertEqual(gp.noise_variance.name, 'GP.noise_variance')
        self.assertFalse(gp.noise_variance.fixed)
        self.assertIsNone(gp.noise_variance.prior)
        np.testing.assert_allclose(gp.noise_variance.value, np.ones((1, 1)))
        np.testing.assert_allclose(gp.noise_variance.log, np.zeros((1, 1)))


class TestGaussianProcessTrainingData(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        self.gp = GP(['x', 'y'], 'z', solver='ipopt')
        self.X_train = np.array([[0., .5, 1. / np.sqrt(2.), np.sqrt(3.) / 2., 1., 0.],
                                 [1., np.sqrt(3.) / 2., 1. / np.sqrt(2.), .5, 0., -1.]])
        self.y_train = np.array([[0., np.pi / 6., np.pi / 4., np.pi / 3., np.pi / 2., np.pi]])

    def test_gaussian_process_wrong_dimension_in_x(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.gp.X_train = np.array([[1., 2., 3., 4., 5.]])
        self.assertEqual(str(context.exception), "Dimension mismatch. Supplied dimension for the features is 1, but "
                                                 "required dimension is 2.")

    def test_gaussian_process_wrong_dimension_in_y(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.gp.y_train = np.array([[2., 3., 4., 5., 6.], [2., 3., 4., 5., 6.]])
        self.assertEqual(str(context.exception), "Dimension mismatch. Supplied dimension for the labels is 2, but "
                                                 "required dimension is 1.")

    def test_gaussian_process_wrong_dimension_in_observations(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.gp.set_training_data(np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 0.]]),
                                      np.array([[.1, .2, .3, .4]]))
        self.assertEqual(str(context.exception), "Number of observations in training matrix and target vector do not "
                                                 "match!")

    def test_gaussian_process_fit_again_warning(self) -> None:
        """

        :return:
        """
        gp = self.gp
        X_train = self.X_train
        y_train = self.y_train

        gp.set_training_data(X_train, y_train)
        gp.setup()
        gp.fit_model()

        with self.assertWarns(UserWarning) as context:
            gp.set_training_data(X_train, y_train)
        self.assertEqual(len(context.warnings), 1)
        self.assertEqual(str(context.warning), "Gaussian process was already executed. Use the fit_model() method again"
                                               " to optimize with respect to the newly set training data.")

    def test_gaussian_process_set_up_again_warning(self) -> None:
        """

        :return:
        """
        gp = self.gp
        X_train = self.X_train
        y_train = self.y_train

        gp.set_training_data(X_train, y_train)
        gp.setup()
        gp.fit_model()

        with self.assertWarns(UserWarning) as context:
            gp.set_training_data(X_train[:, :-1], y_train[:, :-1])
        self.assertEqual(len(context.warnings), 2)
        self.assertEqual(str(context.warnings[0].message), "Gaussian process was already executed. Use the fit_model() "
                                                           "method again to optimize with respect to the newly set "
                                                           "training data.")
        self.assertEqual(str(context.warnings[1].message), "Dimensions of training data set changed. Please run setup()"
                                                           " method again.")


class TestGaussianProcessSetupAndModelFitting(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        self.gp = GP(['x', 'y'], 'z', solver='ipopt')
        self.X_train = np.array([[0., .5, 1. / np.sqrt(2.), np.sqrt(3.) / 2., 1., 0.],
                                 [1., np.sqrt(3.) / 2., 1. / np.sqrt(2.), .5, 0., -1.]])
        self.y_train = np.array([[0., np.pi / 6., np.pi / 4., np.pi / 3., np.pi / 2., np.pi]])

    def test_gaussian_process_no_training_data_supplied_x(self) -> None:
        """

        :return:
        """
        with self.assertRaises(RuntimeError) as context:
            self.gp.setup()
        self.assertEqual(str(context.exception), "The training data has not been set. Please run the method "
                                                 "set_training_data() to proceed.")

    def test_gaussian_process_no_training_data_supplied_y(self) -> None:
        """

        :return:
        """
        gp = self.gp

        gp.X_train = self.X_train
        with self.assertRaises(RuntimeError) as context:
            self.gp.setup()
        self.assertEqual(str(context.exception), "The training data has not been set. Please run the method "
                                                 "set_training_data() to proceed.")

    def test_gaussian_process_log_marginal_likelihood(self) -> None:
        """

        :return:
        """
        gp = self.gp

        gp.set_training_data(self.X_train, self.y_train)
        gp.setup()
        np.testing.assert_approx_equal(gp.log_marginal_likelihood(), -9.82229944)

    def test_gaussian_process_hyperparameter_hyperprior_log_marginal_likelihood(self) -> None:
        """

        :return:
        """
        from hilo_mpc.util.probability import LaplacePrior

        gp = self.gp

        length_scales = gp.kernel.length_scales
        length_scales.prior = 'Laplace'
        length_scales.prior.mean = 0.
        length_scales.prior.variance = 1.
        self.assertIsInstance(length_scales.prior, LaplacePrior)
        self.assertEqual(length_scales.prior.name, 'Laplace')
        np.testing.assert_approx_equal(length_scales.prior.mean, 0.)
        np.testing.assert_approx_equal(length_scales.prior.variance, 1.)

        gp.set_training_data(self.X_train, self.y_train)
        gp.setup()
        np.testing.assert_approx_equal(gp.log_marginal_likelihood(), -10.16887303)

    # def test_gaussian_process_hyperparameter_is_not_log(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     gp = self.gp
    #     self.assertEqual(len(gp.hyperparameters), 3)
    #     self.assertEqual(len(gp.hyperparameter_names), 3)
    #
    #     gp.mean = Mean.constant()
    #     self.assertEqual(len(gp.hyperparameters), 4)
    #     self.assertEqual(len(gp.hyperparameter_names), 4)
    #     self.assertIn('Const.bias', gp.hyperparameter_names)
    #
    #     gp.set_training_data(self.X_train, self.y_train)
    #     # FIXME: Make kernel, mean, ... properties of the GaussianProcess class. Inside we also need to update affected
    #     #  attributes like self._hyp_is_log for example (right now kernel, mean, ... are attributes of the class and
    #     #  affected attributes will not be updated.
    #     gp.setup()
    #     np.testing.assert_approx_equal(gp.log_marginal_likelihood(), )

    # def test_gaussian_process_hyperparameter_ard(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     gp = self.gp
    #
    #     # FIXME: We also need to update active_dims of the corresponding kernel if we change the length scales like this
    #     length_scales = gp.kernel.length_scales
    #     length_scales.value = [1., 1., 1.]
    #     self.assertEqual(length_scales.value.shape, (3, 1))
    #     self.assertEqual(length_scales.SX.shape, (3, 1))
    #     self.assertEqual(length_scales.MX.shape, (3, 1))
    #     np.testing.assert_equal(length_scales.value, np.ones((3, 1)))
    #     np.testing.assert_allclose(length_scales.log, np.zeros((3, 1)))
    #
    #     gp.set_training_data(self.X_train, self.y_train)
    #     gp.setup()
    #     np.testing.assert_approx_equal(gp.log_marginal_likelihood(), )

    def test_gaussian_process_fixed_parameter_is_log_no_variance(self) -> None:
        """

        :return:
        """
        # NOTE: This test would need to be adapted if we provided other objectives as well
        from hilo_mpc.util.probability import DeltaPrior

        gp = self.gp

        length_scales = gp.kernel.length_scales
        self.assertFalse(length_scales.fixed)
        length_scales.fixed = True
        self.assertTrue(length_scales.fixed)
        self.assertIsInstance(length_scales.prior, DeltaPrior)
        self.assertEqual(length_scales.prior.name, 'Delta')

        gp.set_training_data(self.X_train, self.y_train)
        gp.setup()

        np.testing.assert_allclose(gp.hyperparameters[1].value, np.ones((1, 1)))
        lml_before = gp.log_marginal_likelihood()
        gp.fit_model()
        np.testing.assert_allclose(gp.hyperparameters[1].value, np.ones((1, 1)))
        lml_after = gp.log_marginal_likelihood()
        self.assertGreater(lml_after, lml_before)

    def test_gaussian_process_fixed_parameter_is_log(self) -> None:
        """

        :return:
        """
        from hilo_mpc.util.probability import DeltaPrior

        gp = self.gp

        noise_variance = gp.noise_variance
        self.assertFalse(noise_variance.fixed)
        noise_variance.fixed = True
        self.assertTrue(noise_variance.fixed)
        self.assertIsInstance(noise_variance.prior, DeltaPrior)
        self.assertEqual(noise_variance.prior.name, 'Delta')

        gp.set_training_data(self.X_train, self.y_train)
        gp.setup()

        np.testing.assert_allclose(gp.hyperparameters[0].value, np.ones((1, 1)))
        lml_before = gp.log_marginal_likelihood()
        gp.fit_model()
        np.testing.assert_allclose(gp.hyperparameters[0].value, np.ones((1, 1)))
        lml_after = gp.log_marginal_likelihood()
        self.assertGreater(lml_after, lml_before)

    # def test_gaussian_process_fixed_parameter_is_not_log(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     from hilo_mpc.util.probability import DeltaPrior
    #
    #     gp = self.gp
    #     self.assertEqual(len(gp.hyperparameters), 3)
    #     self.assertEqual(len(gp.hyperparameter_names), 3)
    #
    #     gp.mean = Mean.constant(hyperprior='Delta')
    #     self.assertIsInstance(gp.mean.bias.prior, DeltaPrior)
    #     self.assertEqual(gp.mean.bias.prior.name, 'Delta')
    #     self.assertEqual(len(gp.hyperparameters), 4)
    #     self.assertEqual(len(gp.hyperparameter_names), 4)
    #     self.assertIn('Const.bias', gp.hyperparameter_names)
    #     self.assertTrue(gp.mean.bias.fixed)
    #
    #     gp.set_training_data(self.X_train, self.y_train)
    #     # FIXME: Make kernel, mean, ... properties of the GaussianProcess class. Inside we also need to update affected
    #     #  attributes like self._hyp_is_log for example (right now kernel, mean, ... are attributes of the class and
    #     #  affected attributes will not be updated.
    #     gp.setup()
    #
    #     np.testing.assert_allclose(gp.hyperparameters[0].value, np.ones((1, 1)))
    #     lml_before = gp.log_marginal_likelihood()
    #     gp.fit_model()
    #     np.testing.assert_allclose(gp.hyperparameters[0].value, np.ones((1, 1)))
    #     lml_after = gp.log_marginal_likelihood()
    #     self.assertGreater(lml_after, lml_before)

    # def test_gaussian_process_fixed_parameter_ard(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     from hilo_mpc.util.probability import DeltaPrior
    #
    #     gp = self.gp
    #
    #     # FIXME: We also need to update active_dims of the corresponding kernel if we change the length scales like this
    #     length_scales = gp.kernel.length_scales
    #     length_scales.value = [1., 1., 1.]
    #     self.assertEqual(length_scales.value.shape, (3, 1))
    #     self.assertEqual(length_scales.SX.shape, (3, 1))
    #     self.assertEqual(length_scales.MX.shape, (3, 1))
    #     np.testing.assert_equal(length_scales.value, np.ones((3, 1)))
    #     np.testing.assert_allclose(length_scales.log, np.zeros((3, 1)))
    #
    #     self.assertFalse(length_scales.fixed)
    #     length_scales.fixed = True
    #     self.assertTrue(length_scales.fixed)
    #     self.assertIsInstance(length_scales.prior, DeltaPrior)
    #     self.assertEqual(length_scales.prior.name, 'Delta')
    #
    #     gp.set_training_data(self.X_train, self.y_train)
    #     gp.setup()
    #
    #     np.testing.assert_allclose(gp.hyperparameters[0].value, np.ones((1, 1)))
    #     lml_before = gp.log_marginal_likelihood()
    #     gp.fit_model()
    #     np.testing.assert_allclose(gp.hyperparameters[0].value, np.ones((1, 1)))
    #     lml_after = gp.log_marginal_likelihood()
    #     self.assertGreater(lml_after, lml_before)

    def test_gaussian_process_is_not_set_up(self) -> None:
        """

        :return:
        """
        self.assertFalse(self.gp.is_setup())

    def test_gaussian_process_is_set_up(self) -> None:
        """

        :return:
        """
        gp = self.gp
        gp.set_training_data(self.X_train, self.y_train)
        gp.setup()
        self.assertTrue(gp.is_setup())

    def test_gaussian_process_fit_model_not_set_up(self) -> None:
        """

        :return:
        """
        gp = self.gp

        gp.set_training_data(self.X_train, self.y_train)
        with self.assertRaises(RuntimeError) as context:
            gp.fit_model()
        self.assertEqual(str(context.exception),
                         "The GP has not been set up yet. Please run the setup() method before fitting.")

    def test_gaussian_process_fit_model(self) -> None:
        """

        :return:
        """
        gp = self.gp

        gp.set_training_data(self.X_train, self.y_train)
        gp.setup()

        lml_before = gp.log_marginal_likelihood()
        gp.fit_model()
        lml_after = gp.log_marginal_likelihood()
        self.assertGreater(lml_after, lml_before)
        np.testing.assert_array_less(gp.noise_variance.value, np.array([[1e-3]]))

    def test_gaussian_process_fit_model_hyperprior(self) -> None:
        """

        :return:
        """
        from hilo_mpc.util.probability import GaussianPrior

        gp = self.gp

        noise_variance = gp.noise_variance
        noise_variance.prior = 'Gaussian'
        noise_variance.prior.mean = .2
        noise_variance.prior.variance = .01
        self.assertIsInstance(noise_variance.prior, GaussianPrior)
        self.assertEqual(noise_variance.prior.name, 'Gaussian')
        np.testing.assert_approx_equal(noise_variance.prior.mean, .2)
        np.testing.assert_approx_equal(noise_variance.prior.variance, .01)

        y_train = self.y_train
        y_train += np.array([[.23757934, .55730318, .02598826, .06349002, .26647032, -.137302]])

        gp.set_training_data(self.X_train, y_train)
        gp.setup()

        lml_before = gp.log_marginal_likelihood()
        gp.fit_model()
        lml_after = gp.log_marginal_likelihood()
        self.assertGreater(lml_after, lml_before)
        np.testing.assert_allclose(noise_variance.value, np.array([[1.406995]]))


class TestGaussianProcessPrediction(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        gp = GP(['x', 'y'], 'z', solver='ipopt')
        X_train = np.array([[0., .5, 1. / np.sqrt(2.), np.sqrt(3.) / 2., 1., 0.],
                            [1., np.sqrt(3.) / 2., 1. / np.sqrt(2.), .5, 0., -1.]])
        y_train = np.array([[0., np.pi / 6., np.pi / 4., np.pi / 3., np.pi / 2., np.pi]])
        y_train += np.array([[0.05850223,  0.09876431, -0.05570195,  0.15573265,  0.03278181, -0.06901315]])

        gp.set_training_data(X_train, y_train)
        gp.setup()
        gp.fit_model()

        self.gp = gp

    def test_gaussian_process_predict_not_set_up(self) -> None:
        """

        :return:
        """
        gp = GP(['x', 'y'], 'z', solver='ipopt')
        with self.assertRaises(RuntimeError) as context:
            gp.predict(np.array([0., 1., 2., 3.]))
        self.assertEqual(str(context.exception),
                         "The GP has not been set up yet. Please run the setup() method before predicting.")

    def test_gaussian_process_predict(self) -> None:
        """

        :return:
        """
        gp = self.gp

        X_test = np.array([[.25, .6, .75, .9, .4], [.9, .75, .6, .25, -.5]])
        mean, var = gp.predict(X_test)
        mean_nf, var_nf = gp.predict(X_test, noise_free=True)

        np.testing.assert_allclose(mean, mean_nf)
        np.testing.assert_array_less(var_nf, var)

    # def test_gaussian_process_predict_symbolic(self) -> None:
    #     """
    #
    #     :return:
    #     """

    # def test_gaussian_process_predict_quantiles_test_data(self) -> None:
    #     """
    #
    #     :return:
    #     """

    # def test_gaussian_process_predict_quantiles_mean_var(self) -> None:
    #     """
    #
    #     :return:
    #     """

    def test_gaussian_process_predict_quantiles_none(self) -> None:
        """

        :return:
        """
        self.assertIsNone(self.gp.predict_quantiles())


# class TestOneFeatureOneLabel(TestCase):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         self.x = randn(.8, (20, 1))
#         self.y = np.sin(3 * self.x) + .1 * randn(.9, (20, 1))
#         self.x_s = np.linspace(-3, 3, 61).reshape(1, -1)
#         self.kernel = None
#         self.solver = None
#
#     def tearDown(self) -> None:
#         """
#
#         :return:
#         """
#         # NOTE: Parameter values for comparison are taken from the gpml toolbox for MATLAB using its default solver
#         #  unless otherwise noted
#         if self.kernel is None:
#             np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
#                                        np.array([.0085251, .5298217, .8114553]), rtol=1e-5)
#         else:
#             if self.kernel.acronym == 'Const':
#                 np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
#                                            np.array([.7009480, 3.0498634]), rtol=1e-5)
#             elif self.kernel.acronym == 'E':
#                 # NOTE: Hyperparameters should always be positive
#                 # NOTE: Noise variance is almost 0
#                 np.testing.assert_array_less(self.gp.noise_variance.value, 1e-5 * np.ones((1, 1)))
#                 np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters][1:],
#                                            np.array([.9769135, .5935212]), rtol=1e-5)
#             elif self.kernel.acronym == 'M32':
#                 np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
#                                            np.array([.0088262, .8329355, .9398366]), rtol=1e-5)
#             elif self.kernel.acronym == 'M52':
#                 np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
#                                            np.array([.0086694, .7180206, .9571137]), rtol=1e-5)
#             elif self.kernel.acronym == 'Matern':
#                 np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
#                                            np.array([.0086179, .6666981, .9421087]), rtol=1e-5)
#             elif self.kernel.acronym == 'RQ':
#                 # NOTE: Here, we get a slightly better optimum than with the default solver of the gpml toolbox, but
#                 #  the parameter \alpha is quite high. Don't know if a value that high makes much sense.
#                 # NOTE: The last value is always fluctuating very much on different computers since the sensitivity of
#                 #  that value w.r.t. to objective value seems to be very low, so we are ignoring it for now.
#                 np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters][:-1],
#                                            np.array([.00852509, .529822, .811455]), rtol=1e-5)
#             elif self.kernel.acronym == 'PP':
#                 if self.degree == 0:
#                     # NOTE: Here, we get a better optimum than with the default solver of the gpml toolbox for MATLAB
#                     #  (-1.29015 vs -1.78466). Also, if we use the optimal parameter values in the gpml toolbox, no
#                     #  optimization is taking place. But the value for the noise variance is super low. I don't know if
#                     #  such a low value makes much sense. Maybe we can find a better example or there is some way to
#                     #  make sure that the hyperparameters don't get too low.
#                     np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
#                                                np.array([3.71835e-29, .994594, .427023]), rtol=1e-5)
#                 elif self.degree == 1:
#                     np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
#                                                np.array([.0088391, 1.6079331, .6545336]), rtol=1e-5)
#                 elif self.degree == 2:
#                     np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
#                                                np.array([.0088244, 2.0883167, .852502]), rtol=1e-5)
#                 elif self.degree == 3:
#                     np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
#                                                np.array([.0086766, 2.247363, .7873581]), rtol=1e-5)
#             elif self.kernel.acronym == 'Poly':
#                 np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
#                                            np.array([.0980796, 1.3112287, .5083423]), rtol=1e-5)
#             elif self.kernel.acronym == 'Lin':
#                 np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
#                                            np.array([.6627861, .008198]), rtol=1e-4)
#             elif self.kernel.acronym == 'NN':
#                 np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
#                                            np.array([.0095177, 5.7756069, .1554265]), rtol=1e-5)
#             elif self.kernel.acronym == 'Periodic':
#                 np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
#                                            np.array([.4975112, .159969, .5905631, .8941061]), rtol=1e-3)
#
#     def test_gp_regression(self) -> None:
#         """
#
#         :return:
#         """
#         gp = GP(['x'], ['y'], kernel=self.kernel, noise_variance=np.exp(-2), solver=self.solver)
#         gp.set_training_data(self.x.T, self.y.T)
#         gp.setup()
#         with self.assertWarns(UserWarning) as context:
#             gp.fit_model()
#             warnings.warn("Dummy warning!!!")
#         self.assertEqual(len(context.warnings), 1)  # this will catch unsuccessful optimizations (among other warnings)
#
#         self.gp = gp
#
#
# class TestOneFeatureOneLabelConst(TestOneFeatureOneLabel):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         super().setUp()
#         self.y += 3.
#         self.kernel = Kernel.constant()
#
#
# @skip("This test is not necessary, since the GP in 'TestOneFeatureOneLabel' already defaults to the squared exponential"
#       " kernel")
# class TestOneFeatureOneLabelSE(TestOneFeatureOneLabel):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         super().setUp()
#         self.kernel = Kernel.squared_exponential()
#
#
# class TestOneFeatureOneLabelE(TestOneFeatureOneLabel):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         super().setUp()
#         self.kernel = Kernel.exponential()
#
#
# class TestOneFeatureOneLabelM32(TestOneFeatureOneLabel):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         super().setUp()
#         self.kernel = Kernel.matern_32()
#
#
# class TestOneFeatureOneLabelM52(TestOneFeatureOneLabel):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         super().setUp()
#         self.kernel = Kernel.matern_52()
#
#
# class TestOneFeatureOneLabelM72(TestOneFeatureOneLabel):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         from hilo_mpc import MaternKernel
#
#         super().setUp()
#         self.kernel = MaternKernel(3)
#
#
# class TestOneFeatureOneLabelRQ(TestOneFeatureOneLabel):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         super().setUp()
#         self.kernel = Kernel.rational_quadratic()
#         self.solver = 'BFGS'
#
#
# class TestOneFeatureOneLabelPP0(TestOneFeatureOneLabel):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         # TODO: Find out what's happening here. We get a super small noise variance.
#         super().setUp()
#         self.kernel = Kernel.piecewise_polynomial(0)
#         self.solver = 'Powell'
#         self.degree = 0
#
#
# class TestOneFeatureOneLabelPP1(TestOneFeatureOneLabel):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         super().setUp()
#         self.kernel = Kernel.piecewise_polynomial(1)
#         self.solver = 'BFGS'
#         self.degree = 1
#
#
# class TestOneFeatureOneLabelPP2(TestOneFeatureOneLabel):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         super().setUp()
#         self.kernel = Kernel.piecewise_polynomial(2)
#         self.degree = 2
#
#
# class TestOneFeatureOneLabelPP3(TestOneFeatureOneLabel):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         super().setUp()
#         self.kernel = Kernel.piecewise_polynomial(3)
#         self.degree = 3
#
#
# class TestOneFeatureOneLabelPoly(TestOneFeatureOneLabel):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         super().setUp()
#         self.kernel = Kernel.polynomial(3)
#
#
# class TestOneFeatureOneLabelLin(TestOneFeatureOneLabel):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         super().setUp()
#         self.kernel = Kernel.linear()
#
#
# class TestOneFeatureOneLabelNN(TestOneFeatureOneLabel):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         super().setUp()
#         self.kernel = Kernel.neural_network()
#         self.solver = 'CG'
#
#
# class TestOneFeatureOneLabelPeriodic(TestOneFeatureOneLabel):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         super().setUp()
#         self.kernel = Kernel.periodic()
#
#
# class RasmussenSimpleRegression(TestCase):
#     """"""
#     def setUp(self) -> None:
#         """
#
#         :return:
#         """
#         self.kernel = Kernel.matern_32(length_scales=.25)
#         self.mean = Mean.linear(coefficient=.5, bounds={'coefficient': 'fixed'}) + Mean.one()
#
#         self.x = randn(0.3, (20, 1))
#         K = self.kernel(self.x.T)
#         mu = self.mean(self.x.T)
#         self.y = np.linalg.cholesky(K) @ randn(.15, (20, 1)) + mu.T + .1 * randn(.2, (20, 1))
#         self.x_s = np.linspace(-1.9, 1.9, 101).reshape(1, -1)
#
#     def tearDown(self) -> None:
#         """
#
#         :return:
#         """
#         np.testing.assert_approx_equal(self.lml, -11.9706317)
#         np.testing.assert_allclose([parameter.value for parameter in self.gp_1.hyperparameters],
#                                    np.array([.0222571, .3703166, 3.9427837]), rtol=1e-6)
#         np.testing.assert_allclose([parameter.value for parameter in self.gp_2.hyperparameters],
#                                    np.array([.02183, 1.1918832, 1.4624534, .2201449, .4017997]), rtol=1e-6)
#         # NOTE: Here, we get a better optimum than with the default solver of the gpml toolbox for MATLAB
#         #  (-2.26587 vs -3.36002), but only with the solvers 'Nelder-Mead' and 'Powell', which don't require gradients.
#         #  Also, if we use the optimal parameter values in the gpml toolbox, no optimization is taking place.
#         #  Maybe we can find a better example.
#         np.testing.assert_allclose([parameter.value for parameter in self.gp_3.hyperparameters],
#                                    np.array([.0100000, 1.00023, 1., .260644, .719711]), rtol=1e-5)
#
#     def test_gp_regression(self):
#         """
#
#         :return:
#         """
#         gp = GP(['x'], ['y'], mean=self.mean, kernel=self.kernel, noise_variance=.1 ** 2)
#         gp.set_training_data(self.x.T, self.y.T)
#         gp.setup()
#         self.lml = gp.log_marginal_likelihood()
#
#         gp = GP(['x'], ['y'], noise_variance=.1 ** 2)
#         gp.set_training_data(self.x.T, self.y.T)
#         gp.setup()
#         gp.fit_model()
#         self.gp_1 = gp
#
#         mean = Mean.linear(coefficient=0.) + Mean.constant(bias=0.)
#         gp = GP(['x'], ['y'], mean=mean, noise_variance=.1 ** 2)
#         gp.set_training_data(self.x.T, self.y.T)
#         gp.setup()
#         gp.fit_model()
#         self.gp_2 = gp
#
#         mean_1 = Mean.linear(coefficient=0., hyperprior='Gaussian')
#         mean_1.coefficient.prior.mean = 1.
#         mean_1.coefficient.prior.variance = .01 ** 2
#         mean_2 = Mean.constant(bias=0., hyperprior='Laplace')
#         mean_2.bias.prior.mean = 1.
#         mean_2.bias.prior.variance = .01 ** 2
#         mean = mean_1 + mean_2
#         gp = GP(['x'], ['y'], mean=mean, noise_variance=.1 ** 2, solver='Powell')
#         gp.noise_variance.prior = 'Delta'
#         gp.set_training_data(self.x.T, self.y.T)
#         gp.setup()
#         gp.fit_model()
#         self.gp_3 = gp


def randn(seed: float, shape: Tuple[int, int]):
    """

    :param seed:
    :param shape:
    :return:
    """
    n = np.prod(shape)
    N = int(np.ceil(n / 2) * 2)

    a = 7 ** 5
    m = 2 ** 31 - 1

    q = np.fix(m / a)
    r = np.remainder(m, a)
    u = np.zeros((N + 1, 1))
    u[0] = np.fix(seed * 2 ** 31)
    for k in range(1, N + 1):
        u[k] = a * np.remainder(u[k - 1], q) - r * np.fix(u[k - 1] / q)
        if u[k] < 0.:
            u[k] += m
    u = u[1:N + 1] / (2 ** 31)

    N2 = int(N / 2)
    w = np.sqrt(-2 * np.log(u[:N2]))
    x = np.concatenate([w * np.cos(2 * np.pi * u[N2:N]), w * np.sin(2 * np.pi * u[N2:N])])
    return np.reshape(x[:n], shape)
