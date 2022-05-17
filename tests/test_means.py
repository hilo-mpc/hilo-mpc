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

    # def test_polynomial_mean_hyperprior_gaussian_laplace(self):
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

    # def test_polynomial_mean_hyperprior_ard(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # FIXME: Fix behavior when supplying hyperpriors this way (both for PositiveParameter and Parameter class)
    #     mean = Mean.polynomial(2, active_dims=[0, 1], coefficient=[1., 1.], hyperprior={'offset': "Student's t"},
    #                            hyperprior_parameters={'offset': {'mean': 1., 'variance': .01 ** 2, 'nu': 3}})
    #     # TODO: Finish once bug is fixed


class TestLinearMean(TestCase):
    """"""
    def test_linear_mean_no_hyperprior(self) -> None:
        """

        :return:
        """
        mean = Mean.linear()

        self.assertIsNone(mean.active_dims)
        self.assertFalse(hasattr(mean, 'log'))
        np.testing.assert_equal(mean.coefficient.value, np.ones((1, 1)))
        self.assertEqual(mean.degree, 1)
        self.assertEqual(len(mean.hyperparameters), 1)
        self.assertEqual(mean.hyperparameter_names, ['Lin.coefficient'])
        np.testing.assert_equal(mean.offset, 0.)

    def test_linear_mean_fixed(self) -> None:
        """

        :return:
        """
        mean = Mean.linear(hyperprior='Delta')

        self.assertTrue(mean.coefficient.fixed)

    # def test_linear_mean_hyperprior_gaussian(self):
    #     """
    #
    #     :return:
    #     """
    #     # FIXME: Fix behavior when supplying hyperpriors this way (both for PositiveParameter and Parameter class)
    #     mean = Mean.linear(hyperprior={'coefficient': 'Gaussian'},
    #                        hyperprior_parameters={'coefficient': {'mean': 1., 'variance': .01 ** 2}})
    #     # TODO: Finish once bug is fixed

    def test_linear_mean_hyperprior_wrong_type(self) -> None:
        """

        :return:
        """
        with self.assertRaises(TypeError) as context:
            Mean.linear(hyperprior=1.)
        self.assertEqual(str(context.exception), "Wrong type 'float' for keyword argument 'hyperprior'")

    def test_linear_mean_ard_dimension_mismatch(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            Mean.linear(active_dims=[0, 1], coefficient=[1., 1., 1.])
        self.assertEqual(str(context.exception),
                         "Dimension mismatch between 'active_dims' (2) and the number of coefficients (3)")

    def test_linear_mean_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        mean = Mean.linear()

        x = ca.SX.sym('x')
        mu = mean(x)

        grad = ca.gradient(mu, x)

        self.assertIsInstance(mu, ca.SX)
        self.assertTrue(ca.depends_on(mu, x))
        self.assertEqual(grad, mean.coefficient.SX)

    # def test_linear_mean_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     mean = Mean.linear()
    #
    #     x = ca.MX.sym('x')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     mu = mean(x)
    #
    #     grad = ca.gradient(mu, x)
    #
    #     self.assertIsInstance(mu, ca.MX)
    #     self.assertTrue(ca.depends_on(mu, x))
    #     self.assertEqual(grad, mean.coefficient.MX)

    def test_linear_mean_numeric_call(self) -> None:
        """

        :return:
        """
        mean = Mean.linear()

        x = np.array([[1., 2., 3., 4., 5.]])
        mu = mean(x)

        self.assertIsInstance(mu, np.ndarray)
        np.testing.assert_allclose(mu, np.array([[1., 2., 3., 4., 5.]]))

    def test_linear_mean_ard(self) -> None:
        """

        :return:
        """
        # FIXME: The following should also work: Mean.linear(active_dims=[0, 1], coefficient=1.),
        #  Mean.linear(coefficient=[1., 1.])
        mean = Mean.linear(active_dims=[0, 1], coefficient=[1., 1.])

        self.assertEqual(mean.active_dims, [0, 1])
        np.testing.assert_equal(mean.coefficient.value, np.ones((2, 1)))
        # TODO: Finish once bug is fixed

    def test_linear_mean_ard_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        mean = Mean.linear(active_dims=[0, 1], coefficient=[1., 1.])

        x = ca.SX.sym('x', 2)
        mu = mean(x)

        grad = ca.gradient(mu, x)

        self.assertIsInstance(mu, ca.SX)
        self.assertTrue(ca.depends_on(mu, x))
        # NOTE: Checking equality only works for scalars in this way -> use of the elements-method
        self.assertEqual(grad.elements(), mean.coefficient.SX.elements())

    # def test_linear_mean_ard_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     mean = Mean.linear(active_dims=[0, 1], coefficient=[1., 1.])
    #
    #     x = ca.MX.sym('x', 2)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     mu = mean(x)
    #
    #     grad = ca.gradient(mu, x)
    #
    #     self.assertIsInstance(mu, ca.MX)
    #     self.assertTrue(ca.depends_on(mu, x))
    #     self.assertEqual(grad, mean.coefficient.MX)

    def test_linear_mean_ard_numeric_call(self) -> None:
        """

        :return:
        """
        mean = Mean.linear(active_dims=[0, 1], coefficient=[1., 1.])

        x = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 0.]])
        mu = mean(x)

        self.assertIsInstance(mu, np.ndarray)
        np.testing.assert_allclose(mu, np.array([[7., 9., 11., 13., 5.]]))

    # def test_linear_mean_hyperprior_ard(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # FIXME: Fix behavior when supplying hyperpriors this way (both for PositiveParameter and Parameter class)
    #     mean = Mean.linear(active_dims=[0, 1], coefficient=[1., 1.], hyperprior={'coefficient': 'Laplace'},
    #                        hyperprior_parameters={'coefficient': {'mean': 1., 'variance': .01 ** 2}})
    #     # TODO: Finish once bug is fixed


class TestMeanOperators(TestCase):
    """"""
    def test_mean_sum(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.mean import LinearMean, ConstantMean

        mean = LinearMean() + ConstantMean()

        self.assertIsNone(mean.active_dims)
        self.assertEqual(len(mean.hyperparameters), 2)
        self.assertEqual(mean.hyperparameter_names, ['Lin.coefficient', 'Const.bias'])
        self.assertIsInstance(mean.mean_1, LinearMean)
        self.assertIsInstance(mean.mean_2, ConstantMean)

        x = np.array([[1., 2., 3., 4., 5.]])
        mu = mean(x)

        np.testing.assert_allclose(mu, np.array([[2., 3., 4., 5., 6.]]))

    def test_mean_scale(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.mean import ConstantMean

        mean = 2. * ConstantMean()

        self.assertIsNone(mean.active_dims)
        self.assertEqual(len(mean.hyperparameters), 1)
        self.assertEqual(mean.hyperparameter_names, ['Const.bias'])
        self.assertIsInstance(mean.mean_1, ConstantMean)
        self.assertIsNone(mean.mean_2)
        np.testing.assert_equal(mean.scale, 2.)

        x = np.array([[1., 2., 3., 4., 5.]])
        mu = mean(x)

        np.testing.assert_allclose(mu, np.array([[2., 2., 2., 2., 2.]]))

    def test_mean_scale_from_the_right(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.mean import ConstantMean

        mean = ConstantMean() * 2.

        self.assertIsNone(mean.active_dims)
        self.assertEqual(len(mean.hyperparameters), 1)
        self.assertEqual(mean.hyperparameter_names, ['Const.bias'])
        self.assertIsInstance(mean.mean_1, ConstantMean)
        self.assertIsNone(mean.mean_2)
        np.testing.assert_equal(mean.scale, 2.)

        x = np.array([[1., 2., 3., 4., 5.]])
        mu = mean(x)

        np.testing.assert_allclose(mu, np.array([[2., 2., 2., 2., 2.]]))

    def test_mean_product(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.mean import ConstantMean

        mean = ConstantMean() * ConstantMean(bias=.5)

        self.assertIsNone(mean.active_dims)
        self.assertEqual(len(mean.hyperparameters), 2)
        self.assertEqual(mean.hyperparameter_names, ['Const_1.bias', 'Const_2.bias'])
        self.assertIsInstance(mean.mean_1, ConstantMean)
        self.assertIsInstance(mean.mean_2, ConstantMean)

        x = np.array([[1., 2., 3., 4., 5.]])
        mu = mean(x)

        np.testing.assert_allclose(mu, np.array([[.5, .5, .5, .5, .5]]))

    def test_mean_power(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.mean import ConstantMean

        mean = ConstantMean(bias=.5) ** 2

        self.assertIsNone(mean.active_dims)
        self.assertEqual(len(mean.hyperparameters), 1)
        self.assertEqual(mean.hyperparameter_names, ['Const.bias'])
        self.assertIsInstance(mean.mean_1, ConstantMean)
        self.assertIsNone(mean.mean_2)
        np.testing.assert_equal(mean.power, 2)

        x = np.array([[1., 2., 3., 4., 5.]])
        mu = mean(x)

        np.testing.assert_allclose(mu, np.array([[.25, .25, .25, .25, .25]]))

    def test_mean_multi_op_sum_power(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.mean import LinearMean, ConstantMean, PolynomialMean

        mean = (LinearMean() + ConstantMean()) ** 2

        self.assertIsNone(mean.active_dims)
        self.assertEqual(len(mean.hyperparameters), 2)
        self.assertEqual(mean.hyperparameter_names, ['Lin.coefficient', 'Const.bias'])
        self.assertIsNone(mean.mean_1.active_dims)
        self.assertEqual(len(mean.mean_1.hyperparameters), 2)
        self.assertEqual(mean.mean_1.hyperparameter_names, ['Lin.coefficient', 'Const.bias'])
        self.assertIsInstance(mean.mean_1.mean_1, LinearMean)
        self.assertIsInstance(mean.mean_1.mean_2, ConstantMean)
        self.assertIsNone(mean.mean_2)
        np.testing.assert_equal(mean.power, 2)

        poly_mean = PolynomialMean(2)

        x = np.array([[1., 2., 3., 4., 5.]])
        mu = mean(x)
        poly_mu = poly_mean(x)

        np.testing.assert_allclose(mu, poly_mu)
        np.testing.assert_allclose(mu, np.array([[4., 9., 16., 25., 36.]]))

    def test_mean_multi_op_sum_1(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.mean import LinearMean

        mean = LinearMean() + LinearMean(coefficient=.5) + LinearMean(coefficient=2.)

        self.assertIsNone(mean.active_dims)
        self.assertEqual(len(mean.hyperparameters), 3)
        self.assertEqual(mean.hyperparameter_names, ['Lin_1.coefficient', 'Lin_2.coefficient', 'Lin_3.coefficient'])
        self.assertIsNone(mean.mean_1.active_dims)
        self.assertEqual(len(mean.mean_1.hyperparameters), 2)
        self.assertEqual(mean.mean_1.hyperparameter_names, ['Lin_1.coefficient', 'Lin_2.coefficient'])
        self.assertIsInstance(mean.mean_1.mean_1, LinearMean)
        self.assertIsInstance(mean.mean_1.mean_2, LinearMean)
        self.assertIsInstance(mean.mean_2, LinearMean)

        x = np.array([[1., 2., 3., 4., 5.]])
        mu = mean(x)

        np.testing.assert_allclose(mu, np.array([[3.5, 7., 10.5, 14., 17.5]]))

    def test_mean_multi_op_product(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.mean import LinearMean

        mean = LinearMean() * LinearMean(coefficient=.5) * LinearMean(coefficient=2.)

        self.assertIsNone(mean.active_dims)
        self.assertEqual(len(mean.hyperparameters), 3)
        self.assertEqual(mean.hyperparameter_names, ['Lin_1.coefficient', 'Lin_2.coefficient', 'Lin_3.coefficient'])
        self.assertIsNone(mean.mean_1.active_dims)
        self.assertEqual(len(mean.mean_1.hyperparameters), 2)
        self.assertEqual(mean.mean_1.hyperparameter_names, ['Lin_1.coefficient', 'Lin_2.coefficient'])
        self.assertIsInstance(mean.mean_1.mean_1, LinearMean)
        self.assertIsInstance(mean.mean_1.mean_2, LinearMean)
        self.assertIsInstance(mean.mean_2, LinearMean)

        x = np.array([[1., 2., 3., 4., 5.]])
        mu = mean(x)

        np.testing.assert_allclose(mu, np.array([[1., 8., 27., 64., 125.]]))

    def test_mean_multi_op_sum_2(self) -> None:
        """

        :return:
        """
        # TODO: Add __sub__ dunder method which uses Scale and Sum classes?
        from hilo_mpc.modules.machine_learning.gp.mean import LinearMean

        mean = LinearMean() + (LinearMean(coefficient=.5) + LinearMean(coefficient=2.))

        self.assertIsNone(mean.active_dims)
        self.assertEqual(len(mean.hyperparameters), 3)
        self.assertEqual(mean.hyperparameter_names, ['Lin_1.coefficient', 'Lin_2.coefficient', 'Lin_3.coefficient'])
        self.assertIsInstance(mean.mean_1, LinearMean)
        self.assertIsNone(mean.mean_2.active_dims)
        self.assertEqual(len(mean.mean_2.hyperparameters), 2)
        self.assertEqual(mean.mean_2.hyperparameter_names, ['Lin_2.coefficient', 'Lin_3.coefficient'])
        self.assertIsInstance(mean.mean_2.mean_1, LinearMean)
        self.assertIsInstance(mean.mean_2.mean_2, LinearMean)

        x = np.array([[1., 2., 3., 4., 5.]])
        mu = mean(x)

        np.testing.assert_allclose(mu, np.array([[3.5, 7., 10.5, 14., 17.5]]))

    def test_mean_multi_op_sum_of_products(self) -> None:
        """

        :return:
        """
        from hilo_mpc.modules.machine_learning.gp.mean import LinearMean

        mean = (LinearMean() * LinearMean(coefficient=.7)) + (LinearMean() * LinearMean(coefficient=.3))

        self.assertIsNone(mean.active_dims)
        self.assertEqual(len(mean.hyperparameters), 4)
        self.assertEqual(mean.hyperparameter_names,
                         ['Lin_1.coefficient', 'Lin_2.coefficient', 'Lin_3.coefficient', 'Lin_4.coefficient'])
        self.assertIsNone(mean.mean_1.active_dims)
        self.assertEqual(len(mean.mean_1.hyperparameters), 2)
        self.assertEqual(mean.mean_1.hyperparameter_names, ['Lin_1.coefficient', 'Lin_2.coefficient'])
        self.assertIsInstance(mean.mean_1.mean_1, LinearMean)
        self.assertIsInstance(mean.mean_1.mean_2, LinearMean)
        self.assertIsNone(mean.mean_2.active_dims)
        self.assertEqual(len(mean.mean_2.hyperparameters), 2)
        self.assertEqual(mean.mean_2.hyperparameter_names, ['Lin_3.coefficient', 'Lin_4.coefficient'])
        self.assertIsInstance(mean.mean_2.mean_1, LinearMean)
        self.assertIsInstance(mean.mean_2.mean_2, LinearMean)

        x = np.array([[1., 2., 3., 4., 5.]])
        mu = mean(x)

        np.testing.assert_allclose(mu, np.array([[1., 4., 9., 16., 25.]]))
