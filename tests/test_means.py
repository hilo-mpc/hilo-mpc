from unittest import TestCase

import casadi as ca
import numpy as np

from hilo_mpc import Mean


class TestConstantMean(TestCase):
    """"""
    def test_constant_mean_no_hyperprior(self) -> None:
        """

        :return:
        """
        mean = Mean.constant()

        self.assertIsNone(mean.active_dims)
        self.assertFalse(hasattr(mean, 'log'))
        np.testing.assert_equal(mean.bias.value, np.ones((1, 1)))
        self.assertEqual(len(mean.hyperparameters), 1)
        self.assertEqual(mean.hyperparameter_names, ['Const.bias'])

    def test_constant_mean_fixed(self) -> None:
        """

        :return:
        """
        mean = Mean.constant(hyperprior='Delta')

        self.assertTrue(mean.bias.fixed)

    # def test_constant_mean_hyperprior_gaussian(self):
    #     """
    #
    #     :return:
    #     """
    #     # FIXME: Fix behavior when supplying hyperpriors this way (both for PositiveParameter and Parameter class)
    #     mean = Mean.constant(hyperprior='Gaussian', hyperprior_parameters={'mean': 1., 'variance': .01 ** 2})
    #     # TODO: Finish once bug is fixed

    def test_constant_mean_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        mean = Mean.constant()

        x = ca.SX.sym('x')
        mu = mean(x)

        self.assertIsInstance(mu, ca.SX)
        self.assertEqual(mu, mean.bias.SX)

    # def test_constant_mean_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     mean = Mean.constant()
    #
    #     x = ca.MX.sym('x')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     mu = mean(x)
    #
    #     self.assertIsInstance(mu, ca.MX)
    #     self.assertEqual(mu, mean.bias.MX)

    def test_constant_mean_numeric_call(self) -> None:
        """

        :return:
        """
        mean = Mean.constant()

        x = np.array([[1., 2., 3., 4., 5.]])
        mu = mean(x)

        self.assertIsInstance(mu, np.ndarray)
        np.testing.assert_equal(mu, np.ones((1, 5)))

        mean.bias.value = 2.
        mu = mean(x)
        np.testing.assert_equal(mu, 2. * np.ones((1, 5)))


class TestZeroMean(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        self.mean = Mean.zero()

    def test_zero_mean(self) -> None:
        """

        :return:
        """
        mean = self.mean

        self.assertIsNone(mean.active_dims)
        self.assertFalse(hasattr(mean, 'log'))
        np.testing.assert_equal(mean.bias, 0.)
        self.assertEqual(len(mean.hyperparameters), 0)

    def test_zero_mean_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        mean = self.mean

        x = ca.SX.sym('x')
        mu = mean(x)

        self.assertIsInstance(mu, ca.SX)
        self.assertTrue(mu.is_zero())

    def test_zero_mean_symbolic_call_mx(self) -> None:
        """

        :return:
        """
        mean = self.mean

        x = ca.MX.sym('x')
        mu = mean(x)
        mu_fun = ca.Function('mu_fun', [x], [mu])

        self.assertIsInstance(mu, ca.MX)
        self.assertTrue(mu_fun(ca.SX.sym('x')).is_zero())

    def test_zero_mean_numeric_call(self) -> None:
        """

        :return:
        """
        mean = self.mean

        x = np.array([[1., 2., 3., 4., 5.]])
        mu = mean(x)

        self.assertIsInstance(mu, np.ndarray)
        np.testing.assert_equal(mu, np.zeros((1, 5)))


class TestOneMean(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        self.mean = Mean.one()

    def test_one_mean(self) -> None:
        """

        :return:
        """
        mean = self.mean

        self.assertIsNone(mean.active_dims)
        self.assertFalse(hasattr(mean, 'log'))
        np.testing.assert_equal(mean.bias, 1.)
        self.assertEqual(len(mean.hyperparameters), 0)

    def test_one_mean_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        mean = self.mean

        x = ca.SX.sym('x')
        mu = mean(x)

        self.assertIsInstance(mu, ca.SX)
        self.assertTrue(mu.is_one())

    def test_one_mean_symbolic_call_mx(self) -> None:
        """

        :return:
        """
        mean = self.mean

        x = ca.MX.sym('x')
        mu = mean(x)
        mu_fun = ca.Function('mu_fun', [x], [mu])

        self.assertIsInstance(mu, ca.MX)
        self.assertTrue(mu_fun(ca.SX.sym('x')).is_one())

    def test_one_mean_numeric_call(self) -> None:
        """

        :return:
        """
        mean = self.mean

        x = np.array([[1., 2., 3., 4., 5.]])
        mu = mean(x)

        self.assertIsInstance(mu, np.ndarray)
        np.testing.assert_equal(mu, np.ones((1, 5)))


class TestPolynomialMean(TestCase):
    """"""
    def test_polynomial_mean_no_hyperprior(self) -> None:
        """

        :return:
        """
        mean = Mean.polynomial(2)

        self.assertIsNone(mean.active_dims)
        self.assertFalse(hasattr(mean, 'log'))
        np.testing.assert_equal(mean.coefficient.value, np.ones((1, 1)))
        self.assertEqual(mean.degree, 2)
        self.assertEqual(len(mean.hyperparameters), 2)
        self.assertEqual(mean.hyperparameter_names, ['Poly.coefficient', 'Poly.offset'])
        np.testing.assert_equal(mean.offset.value, np.ones((1, 1)))

        mean.degree = 3
        self.assertEqual(mean.degree, 3)

    def test_polynomial_mean_fixed(self) -> None:
        """

        :return:
        """
        mean = Mean.polynomial(2, hyperprior='Delta')

        self.assertTrue(mean.coefficient.fixed)
        self.assertTrue(mean.offset.fixed)

    # def test_constant_mean_hyperprior_gaussian_laplace(self):
    #     """
    #
    #     :return:
    #     """
    #     # FIXME: Fix behavior when supplying hyperpriors this way (both for PositiveParameter and Parameter class)
    #     mean = Mean.polynomial(
    #         2,
    #         hyperprior={'coefficient': 'Gaussian', 'offset': 'Laplace'},
    #         hyperprior_parameters={'coefficient': {'mean': 1., 'variance': .01 ** 2},
    #                                'offset': {'mean': 1., 'variance': .01 ** 2}}
    #     )
    #     # TODO: Finish once bug is fixed

    def test_polynomial_mean_hyperprior_wrong_type(self) -> None:
        """

        :return:
        """
        with self.assertRaises(TypeError) as context:
            Mean.polynomial(2, hyperprior=1.)
        self.assertEqual(str(context.exception), "Wrong type 'float' for keyword argument 'hyperprior'")

    def test_polynomial_mean_ard_dimension_mismatch(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            Mean.polynomial(2, active_dims=[0, 1], coefficient=[1., 1., 1.])
        self.assertEqual(str(context.exception),
                         "Dimension mismatch between 'active_dims' (2) and the number of coefficients (3)")

    def test_polynomial_mean_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        mean = Mean.polynomial(2)

        x = ca.SX.sym('x')
        mu = mean(x)

        hess, _ = ca.hessian(mu, x)
        hess_fun = ca.Function('hess_fun', [mean.coefficient.SX], [hess])

        self.assertIsInstance(mu, ca.SX)
        self.assertTrue(ca.depends_on(mu, x))
        self.assertFalse(hess_fun.has_free())
        np.testing.assert_allclose(hess_fun(0.), np.zeros((1, 1)))
        np.testing.assert_allclose(hess_fun(.5), np.array([[.5]]))
        np.testing.assert_allclose(hess_fun(.1), np.array([[.02]]))

    # def test_polynomial_mean_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     mean = Mean.polynomial(2)
    #
    #     x = ca.MX.sym('x')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     mu = mean(x)
    #
    #     hess, _ = ca.hessian(mu, x)
    #     hess_fun = ca.Function('hess_fun', [mean.coefficient.MX], [hess])
    #
    #     self.assertIsInstance(mu, ca.MX)
    #     self.assertTrue(ca.depends_on(mu, x))
    #     self.assertFalse(hess_fun.has_free())
    #     np.testing.assert_allclose(hess_fun(0.), np.zeros((1, 1)))
    #     np.testing.assert_allclose(hess_fun(.5), np.array([[.5]]))
    #     np.testing.assert_allclose(hess_fun(.1), np.array([[.02]]))

    def test_polynomial_mean_numeric_call(self) -> None:
        """

        :return:
        """
        mean = Mean.polynomial(2)

        x = np.array([[1., 2., 3., 4., 5.]])
        mu = mean(x)

        self.assertIsInstance(mu, np.ndarray)
        np.testing.assert_allclose(mu, np.array([[4., 9., 16., 25., 36.]]))

    def test_polynomial_mean_ard(self) -> None:
        """

        :return:
        """
        # FIXME: The following should also work: Mean.polynomial(2, active_dims=[0, 1], coefficient=1.),
        #  Mean.polynomial(2, coefficient=[1., 1.])
        mean = Mean.polynomial(2, active_dims=[0, 1], coefficient=[1., 1.])

        self.assertEqual(mean.active_dims, [0, 1])
        np.testing.assert_equal(mean.coefficient.value, np.ones((2, 1)))
        # TODO: Finish once bug is fixed

    def test_polynomial_mean_ard_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        mean = Mean.polynomial(2, active_dims=[0, 1], coefficient=[1., 1.])

        x = ca.SX.sym('x', 2)
        mu = mean(x)

        hess, _ = ca.hessian(mu, x)
        hess_fun = ca.Function('hess_fun', [mean.coefficient.SX], [hess])

        self.assertIsInstance(mu, ca.SX)
        self.assertTrue(ca.depends_on(mu, x))
        self.assertFalse(hess_fun.has_free())
        np.testing.assert_allclose(hess_fun([0., 0.]), np.zeros((2, 2)))
        np.testing.assert_allclose(hess_fun([.5, 0.]), np.array([[.5, 0.], [0., 0.]]))
        np.testing.assert_allclose(hess_fun([0., .5]), np.array([[0., 0.], [0., .5]]))
        np.testing.assert_allclose(hess_fun([.5, .5]), np.array([[.5, .5], [.5, .5]]))
        np.testing.assert_allclose(hess_fun([.5, .1]), np.array([[.5, .1], [.1, .02]]))
        np.testing.assert_allclose(hess_fun([.1, .5]), np.array([[.02, .1], [.1, .5]]))

    # def test_polynomial_mean_ard_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     mean = Mean.polynomial(2, active_dims=[0, 1], coefficient=[1., 1.])
    #
    #     x = ca.MX.sym('x', 2)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     mu = mean(x)
    #
    #     hess, _ = ca.hessian(mu, x)
    #     hess_fun = ca.Function('hess_fun', [mean.coefficient.MX], [hess])
    #
    #     self.assertIsInstance(mu, ca.MX)
    #     self.assertTrue(ca.depends_on(mu, x))
    #     self.assertFalse(hess_fun.has_free())
    #     np.testing.assert_allclose(hess_fun([0., 0.]), np.zeros((2, 2)))
    #     np.testing.assert_allclose(hess_fun([.5, 0.]), np.array([[.5, 0.], [0., 0.]]))
    #     np.testing.assert_allclose(hess_fun([0., .5]), np.array([[0., 0.], [0., .5]]))
    #     np.testing.assert_allclose(hess_fun([.5, .5]), np.array([[.5, .5], [.5, .5]]))
    #     np.testing.assert_allclose(hess_fun([.5, .1]), np.array([[.5, .1], [.1, .02]]))
    #     np.testing.assert_allclose(hess_fun([.1, .5]), np.array([[.02, .1], [.1, .5]]))

    def test_polynomial_mean_ard_numeric_call(self) -> None:
        """

        :return:
        """
        mean = Mean.polynomial(2, active_dims=[0, 1], coefficient=[1., 1.])

        x = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 0.]])
        mu = mean(x)

        self.assertIsInstance(mu, np.ndarray)
        np.testing.assert_allclose(mu, np.array([[64., 100., 144., 196., 36.]]))

    def test_polynomial_mean_hyperprior_ard(self) -> None:
        """

        :return:
        """
