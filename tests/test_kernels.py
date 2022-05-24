from unittest import TestCase

import casadi as ca
import numpy as np

from hilo_mpc import Kernel


# TODO: Align function arguments for kernels with function arguments for means (hyperprior, ...)
class TestConstantKernel(TestCase):
    """"""
    def test_constant_kernel_no_hyperprior(self) -> None:
        """

        :return:
        """
        kernel = Kernel.constant()

        self.assertIsNone(kernel.active_dims)
        self.assertTrue(hasattr(kernel.bias, 'log'))
        np.testing.assert_equal(kernel.bias.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.bias.log, np.zeros((1, 1)))
        self.assertEqual(len(kernel.hyperparameters), 1)
        self.assertEqual(kernel.hyperparameter_names, ['Const.bias'])

    def test_constant_kernel_fixed(self) -> None:
        """

        :return:
        """
        # TODO: Change according to test_means.py when first TODO is finished
        kernel = Kernel.constant()
        kernel.bias.fixed = True

        self.assertTrue(kernel.bias.fixed)

    # def test_constant_kernel_hyperprior_gaussian(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create according to test_means.py when first TODO is finished

    def test_constant_kernel_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.constant()

        x = ca.SX.sym('x')
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.bias.SX))
        self.assertFalse(ca.depends_on(cov, x))

    # def test_constant_kernel_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.constant()
    #
    #     x = ca.MX.sym('x')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.bias.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    def test_constant_kernel_numeric_call(self) -> None:
        """

        :return:
        """
        kernel = Kernel.constant()

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_equal(cov, np.ones((5, 5)))

        kernel.bias.value = 2.
        cov = kernel(x)
        np.testing.assert_equal(cov, 4. * np.ones((5, 5)))

    def test_constant_kernel_symbolic_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.constant()

        x = ca.SX.sym('x')
        y = np.array([[2.]])
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_constant_kernel_symbolic_call_x_x_bar_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.constant()

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.bias.SX))
        self.assertFalse(ca.depends_on(cov, x))
        self.assertFalse(ca.depends_on(cov, y))

    # def test_constant_kernel_symbolic_call_x_x_bar_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.constant()
    #
    #     x = ca.MX.sym('x')
    #     y = ca.MX.sym('y')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.bias.MX))
    #     self.assertFalse(ca.depends_on(cov, x))
    #     self.assertFalse(ca.depends_on(cov, y))

    def test_constant_kernel_numeric_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.constant()

        x = np.array([[2.]])
        y = ca.SX.sym('y')
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_constant_kernel_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.constant()

        x = np.array([[1., 2., 3., 4., 5.]])
        y = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_equal(cov, np.ones((5, 5)))

        kernel.bias.value = 2.
        cov = kernel(x, y)
        np.testing.assert_equal(cov, 4. * np.ones((5, 5)))


class TestSquaredExponentialKernel(TestCase):
    """"""
    def test_squared_exponential_kernel_no_hyperprior(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential()

        self.assertIsNone(kernel.active_dims)
        np.testing.assert_equal(kernel.alpha, .5)
        np.testing.assert_equal(kernel.gamma, 2.)
        self.assertEqual(len(kernel.hyperparameters), 2)
        self.assertEqual(kernel.hyperparameter_names, ['SE.length_scales', 'SE.signal_variance'])
        self.assertTrue(hasattr(kernel.length_scales, 'log'))
        np.testing.assert_equal(kernel.length_scales.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.length_scales.log, np.zeros((1, 1)))
        self.assertTrue(hasattr(kernel.signal_variance, 'log'))
        np.testing.assert_equal(kernel.signal_variance.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.signal_variance.log, np.zeros((1, 1)))

    def test_squared_exponential_kernel_fixed(self) -> None:
        """

        :return:
        """
        # TODO: Change according to test_means.py when first TODO is finished
        kernel = Kernel.squared_exponential()
        kernel.length_scales.fixed = True
        kernel.signal_variance.fixed = True

        self.assertTrue(kernel.length_scales.fixed)
        self.assertTrue(kernel.signal_variance.fixed)

    # def test_squared_exponential_kernel_hyperprior_gaussian(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create according to test_means.py when first TODO is finished

    def test_squared_exponential_kernel_ard_no_active_dims(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            Kernel.squared_exponential(ard=True)
        self.assertEqual(str(context.exception),
                         "The key word 'ard' can only be set to True if the key word 'active_dims' was supplied")

    def test_squared_exponential_kernel_ard_dimension_mismatch(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            Kernel.squared_exponential(active_dims=[0, 1], length_scales=[1., 1., 1.])
        self.assertEqual(str(context.exception),
                         "Dimension mismatch between 'active_dims' (2) and the number of length_scales (3)")

    def test_squared_exponential_kernel_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential()

        x = ca.SX.sym('x')
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertFalse(ca.depends_on(cov, x))

    # def test_squared_exponential_kernel_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.squared_exponential()
    #
    #     x = ca.MX.sym('x')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    def test_squared_exponential_kernel_numeric_call(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential()

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[1., .60653066, .135335283, .0111089965, .000335462628],
                                                  [.60653066, 1., .60653066, .135335283, .0111089965],
                                                  [.135335283, .60653066, 1., .60653066, .135335283],
                                                  [.0111089965, .135335283, .60653066, 1., .60653066],
                                                  [.000335462628, .0111089965, .135335283, .60653066, 1.]]))

    def test_squared_exponential_kernel_symbolic_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential()

        x = ca.SX.sym('x')
        y = np.array([[2.]])
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_squared_exponential_kernel_symbolic_call_x_x_bar_dimension_mismatch(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential()

        x = ca.SX.sym('x', 2)
        y = ca.SX.sym('y')
        # FIXME: Convert to ValueError
        with self.assertRaises(AssertionError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar do not have the same input space dimensions")

    def test_squared_exponential_kernel_symbolic_call_x_x_bar_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential()

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertTrue(ca.depends_on(cov, x))
        self.assertTrue(ca.depends_on(cov, y))

    # def test_squared_exponential_kernel_symbolic_call_x_x_bar_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.squared_exponential()
    #
    #     x = ca.MX.sym('x')
    #     y = ca.MX.sym('y')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertTrue(ca.depends_on(cov, x))
    #     self.assertTrue(ca.depends_on(cov, y))

    def test_squared_exponential_kernel_numeric_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential()

        x = np.array([[2.]])
        y = ca.SX.sym('y')
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_squared_exponential_kernel_numeric_call_x_x_bar_dimension_mismatch(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential()

        x = np.array([[1.], [2.]])
        y = np.array([[1.]])
        # FIXME: Convert to ValueError
        with self.assertRaises(AssertionError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar do not have the same input space dimensions")

    def test_squared_exponential_kernel_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential()

        x = np.array([[1., 2., 3., 4., 5.]])
        y = np.array([[1., 2., 3., 4., 5.]]) / 2.
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(
            cov,
            np.array([[.8824969, 1., .8824969, .60653066, .32465247],
                      [.32465247, .60653066, .8824969, 1., .8824969],
                      [.04393693, .13533528, .32465247, .60653066, .8824969],
                      [.0021874911, .0111089965, .04393693, .13533528, .32465247],
                      [.0000400652974, .000335462628, .0021874911, .0111089965, .04393693]])
                                   )

    def test_squared_exponential_kernel_ard(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential(active_dims=[0, 1], length_scales=[1., 1.])

        self.assertEqual(kernel.active_dims, [0, 1])
        np.testing.assert_equal(kernel.length_scales.value, np.ones((2, 1)))

        kernel = Kernel.squared_exponential(active_dims=[0, 1, 2], ard=True)

        self.assertEqual(kernel.active_dims, [0, 1, 2])
        np.testing.assert_equal(kernel.length_scales.value, np.ones((3, 1)))

    # def test_squared_exponential_kernel_ard_call_dimension_mismatch(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create similar test for means
    #     kernel = Kernel.squared_exponential(active_dims=[0, 1], length_scales=[1., 1.])
    #
    #     x = ca.SX.sym('x')
    #     # FIXME: This will result in another error that could be unclear to the user. We should probably catch it and
    #     #  return a more informative error message.
    #     cov = kernel(x)

    def test_squared_exponential_kernel_ard_call_dimension_mismatch(self) -> None:
        """

        :return:
        """
        # TODO: Create similar test for means
        kernel = Kernel.squared_exponential(length_scales=[1., 1.])

        x = ca.SX.sym('x')
        with self.assertRaises(ValueError) as context:
            kernel(x)
        self.assertEqual(str(context.exception), "Length scales vector dimension does not equal input space dimension.")

    def test_squared_exponential_kernel_ard_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential(length_scales=[1., 1.])

        x = ca.SX.sym('x', 2)
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertFalse(ca.depends_on(cov, x))

    def test_squared_exponential_kernel_ard_symbolic_call_sx_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential(active_dims=[0, 2], length_scales=[1., 1.])

        x = ca.SX.sym('x', 3)
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertFalse(ca.depends_on(cov, x))

    # def test_squared_exponential_kernel_ard_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.squared_exponential(length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 2)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    # def test_squared_exponential_kernel_ard_symbolic_call_mx_not_all_active(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.squared_exponential(active_dims=[0, 2], length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 3)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    def test_squared_exponential_kernel_ard_numeric_call(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential(length_scales=[1., 1.])

        x = np.array([[1., 2., 3., 4., 5.], [.1, .2, .3, .4, .5]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[1., .603505575, .132655465, .010620173, .00030967104],
                                                  [.603505575, 1., .603505575, .132655465, .010620173],
                                                  [.132655465, .603505575, 1., .603505575, .132655465],
                                                  [.010620173, .132655465, .603505575, 1., .603505575],
                                                  [.00030967104, .010620173, .132655465, .603505575, 1.]
                                                  ]))

    def test_squared_exponential_kernel_ard_numeric_call_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential(active_dims=[0, 2], length_scales=[1., 1.])

        x = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 0.], [.1, .2, .3, .4, .5]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[1., .603505575, .132655465, .010620173, .00030967104],
                                                  [.603505575, 1., .603505575, .132655465, .010620173],
                                                  [.132655465, .603505575, 1., .603505575, .132655465],
                                                  [.010620173, .132655465, .603505575, 1., .603505575],
                                                  [.00030967104, .010620173, .132655465, .603505575, 1.]
                                                  ]))

    def test_squared_exponential_kernel_ard_symbolic_call_sx_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential(length_scales=[1., 1.])

        x = ca.SX.sym('x', 2)
        y = ca.SX.sym('y', 2)
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        for length_scale in kernel.length_scales.SX.elements():
            self.assertTrue(ca.depends_on(cov, length_scale))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        for xk in x.elements():
            self.assertTrue(ca.depends_on(cov, xk))
        for yk in y.elements():
            self.assertTrue(ca.depends_on(cov, yk))

    def test_squared_exponential_kernel_ard_symbolic_call_sx_x_x_bar_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential(active_dims=[0, 2], length_scales=[1., 1.])

        x = ca.SX.sym('x', 3)
        y = ca.SX.sym('y', 3)
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        for length_scale in kernel.length_scales.SX.elements():
            self.assertTrue(ca.depends_on(cov, length_scale))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        for k, xk in enumerate(x.elements()):
            if k == 1:
                self.assertFalse(ca.depends_on(cov, xk))
            else:
                self.assertTrue(ca.depends_on(cov, xk))
        for k, yk in enumerate(y.elements()):
            if k == 1:
                self.assertFalse(ca.depends_on(cov, yk))
            else:
                self.assertTrue(ca.depends_on(cov, yk))

    # def test_squared_exponential_kernel_ard_symbolic_call_mx_x_x_bar(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.squared_exponential(length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 2)
    #     y = ca.MX.sym('y', 2)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     for length_scale in kernel.length_scales.MX.elements():
    #         self.assertTrue(ca.depends_on(cov, length_scale))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     for xk in x.elements():
    #         self.assertTrue(ca.depends_on(cov, xk))
    #     for yk in y.elements():
    #         self.assertTrue(ca.depends_on(cov, yk))

    # def test_squared_exponential_kernel_ard_symbolic_call_mx_x_x_bar_not_all_active(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.squared_exponential(active_dims=[0, 2], length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 3)
    #     y = ca.MX.sym('y', 3)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     for length_scale in kernel.length_scales.MX.elements():
    #         self.assertTrue(ca.depends_on(cov, length_scale))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     for k, xk in enumerate(x.elements()):
    #         if k == 1:
    #             self.assertFalse(ca.depends_on(cov, xk))
    #         else:
    #             self.assertTrue(ca.depends_on(cov, xk))
    #     for k, yk in enumerate(y.elements()):
    #         if k == 1:
    #             self.assertFalse(ca.depends_on(cov, yk))
    #         else:
    #             self.assertTrue(ca.depends_on(cov, yk))

    def test_squared_exponential_kernel_ard_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential(length_scales=[1., 1.])

        x = np.array([[1., 1.1, 1.2], [1.3, 1.4, 1.5]])
        y = np.array([[1.6, 1.7, 1.8], [1.9, 2., 2.1]])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.69767633, .61262639, .52729242],
                                                  [.77880078, .69767633, .61262639],
                                                  [.85214379, .77880078, .69767633]]))

    def test_squared_exponential_kernel_ard_numeric_call_x_x_bar_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.squared_exponential(active_dims=[0, 2], length_scales=[1., 1.])

        x = np.array([[1., 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]])
        y = np.array([[1.9, 2., 2.1], [2.2, 2.3, 2.4], [2.5, 2.6, 2.7]])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.44485807, .36787944, .29819728],
                                                  [.52729242, .44485807, .36787944],
                                                  [.61262639, .52729242, .44485807]]))


class TestExponentialKernel(TestCase):
    """"""
    def test_exponential_kernel_no_hyperprior(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential()

        self.assertIsNone(kernel.active_dims)
        self.assertEqual(len(kernel.hyperparameters), 2)
        self.assertEqual(kernel.hyperparameter_names, ['E.length_scales', 'E.signal_variance'])
        self.assertTrue(hasattr(kernel.length_scales, 'log'))
        np.testing.assert_equal(kernel.length_scales.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.length_scales.log, np.zeros((1, 1)))
        self.assertTrue(hasattr(kernel.signal_variance, 'log'))
        np.testing.assert_equal(kernel.signal_variance.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.signal_variance.log, np.zeros((1, 1)))

    def test_exponential_kernel_fixed(self) -> None:
        """

        :return:
        """
        # TODO: Change according to test_means.py when first TODO is finished
        kernel = Kernel.exponential()
        kernel.length_scales.fixed = True
        kernel.signal_variance.fixed = True

        self.assertTrue(kernel.length_scales.fixed)
        self.assertTrue(kernel.signal_variance.fixed)

    # def test_exponential_kernel_hyperprior_gaussian(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create according to test_means.py when first TODO is finished

    def test_exponential_kernel_ard_no_active_dims(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            Kernel.exponential(ard=True)
        self.assertEqual(str(context.exception),
                         "The key word 'ard' can only be set to True if the key word 'active_dims' was supplied")

    def test_exponential_kernel_ard_dimension_mismatch(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            Kernel.exponential(active_dims=[0, 1], length_scales=[1., 1., 1.])
        self.assertEqual(str(context.exception),
                         "Dimension mismatch between 'active_dims' (2) and the number of length_scales (3)")

    def test_exponential_kernel_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential()

        x = ca.SX.sym('x')
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertFalse(ca.depends_on(cov, x))

    # def test_exponential_kernel_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.exponential()
    #
    #     x = ca.MX.sym('x')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    def test_exponential_kernel_numeric_call(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential()

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[1., .36787944, .13533528, .04978707, .01831564],
                                                  [.36787944, 1., .36787944, .13533528, .04978707],
                                                  [.13533528, .36787944, 1., .36787944, .13533528],
                                                  [.04978707, .13533528, .36787944, 1., .36787944],
                                                  [.01831564, .04978707, .13533528, .36787944, 1.]]))

    def test_exponential_kernel_symbolic_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential()

        x = ca.SX.sym('x')
        y = np.array([[2.]])
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_exponential_kernel_symbolic_call_x_x_bar_dimension_mismatch(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential()

        x = ca.SX.sym('x', 2)
        y = ca.SX.sym('y')
        # FIXME: Convert to ValueError
        with self.assertRaises(AssertionError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar do not have the same input space dimensions")

    def test_exponential_kernel_symbolic_call_x_x_bar_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential()

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertTrue(ca.depends_on(cov, x))
        self.assertTrue(ca.depends_on(cov, y))

    # def test_exponential_kernel_symbolic_call_x_x_bar_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.exponential()
    #
    #     x = ca.MX.sym('x')
    #     y = ca.MX.sym('y')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertTrue(ca.depends_on(cov, x))
    #     self.assertTrue(ca.depends_on(cov, y))

    def test_exponential_kernel_numeric_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential()

        x = np.array([[2.]])
        y = ca.SX.sym('y')
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_exponential_kernel_numeric_call_x_x_bar_dimension_mismatch(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential()

        x = np.array([[1.], [2.]])
        y = np.array([[1.]])
        # FIXME: Convert to ValueError
        with self.assertRaises(AssertionError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar do not have the same input space dimensions")

    def test_exponential_kernel_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential()

        x = np.array([[1., 2., 3., 4., 5.]])
        y = np.array([[1., 2., 3., 4., 5.]]) / 2.
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(
            cov,
            np.array([[.60653066, 1., .60653066, .36787944, .22313016],
                      [.22313016, .36787944, .60653066, 1., .60653066],
                      [.082085, .13533528, .22313016, .36787944, .60653066],
                      [.030197383, .04978707, .082085, .13533528, .22313016],
                      [.011108997, .01831564, .030197383, .04978707, .082085]])
                                   )

    def test_exponential_kernel_ard(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential(active_dims=[0, 1], length_scales=[1., 1.])

        self.assertEqual(kernel.active_dims, [0, 1])
        np.testing.assert_equal(kernel.length_scales.value, np.ones((2, 1)))

        kernel = Kernel.exponential(active_dims=[0, 1, 2], ard=True)

        self.assertEqual(kernel.active_dims, [0, 1, 2])
        np.testing.assert_equal(kernel.length_scales.value, np.ones((3, 1)))

    # def test_exponential_kernel_ard_call_dimension_mismatch(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create similar test for means
    #     kernel = Kernel.exponential(active_dims=[0, 1], length_scales=[1., 1.])
    #
    #     x = ca.SX.sym('x')
    #     # FIXME: This will result in another error that could be unclear to the user. We should probably catch it and
    #     #  return a more informative error message.
    #     cov = kernel(x)

    def test_exponential_kernel_ard_call_dimension_mismatch(self) -> None:
        """

        :return:
        """
        # TODO: Create similar test for means
        kernel = Kernel.exponential(length_scales=[1., 1.])

        x = ca.SX.sym('x')
        with self.assertRaises(ValueError) as context:
            kernel(x)
        self.assertEqual(str(context.exception), "Length scales vector dimension does not equal input space dimension.")

    def test_exponential_kernel_ard_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential(length_scales=[1., 1.])

        x = ca.SX.sym('x', 2)
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertFalse(ca.depends_on(cov, x))

    def test_exponential_kernel_ard_symbolic_call_sx_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential(active_dims=[0, 2], length_scales=[1., 1.])

        x = ca.SX.sym('x', 3)
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertFalse(ca.depends_on(cov, x))

    # def test_exponential_kernel_ard_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.exponential(length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 2)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    # def test_exponential_kernel_ard_symbolic_call_mx_not_all_active(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.exponential(active_dims=[0, 2], length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 3)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    def test_exponential_kernel_ard_numeric_call(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential(length_scales=[1., 1.])

        x = np.array([[1., 2., 3., 4., 5.], [.1, .2, .3, .4, .5]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[1., .36604919, .13399201, .04904767, .017953858],
                                                  [.36604919, 1., .36604919, .13399201, .04904767],
                                                  [.13399201, .36604919, 1., .36604919, .13399201],
                                                  [.04904767, .13399201, .36604919, 1., .36604919],
                                                  [.017953858, .04904767, .13399201, .36604919, 1.]
                                                  ]))

    def test_exponential_kernel_ard_numeric_call_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential(active_dims=[0, 2], length_scales=[1., 1.])

        x = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 0.], [.1, .2, .3, .4, .5]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[1., .36604919, .13399201, .04904767, .017953858],
                                                  [.36604919, 1., .36604919, .13399201, .04904767],
                                                  [.13399201, .36604919, 1., .36604919, .13399201],
                                                  [.04904767, .13399201, .36604919, 1., .36604919],
                                                  [.017953858, .04904767, .13399201, .36604919, 1.]
                                                  ]))

    def test_exponential_kernel_ard_symbolic_call_sx_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential(length_scales=[1., 1.])

        x = ca.SX.sym('x', 2)
        y = ca.SX.sym('y', 2)
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        for length_scale in kernel.length_scales.SX.elements():
            self.assertTrue(ca.depends_on(cov, length_scale))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        for xk in x.elements():
            self.assertTrue(ca.depends_on(cov, xk))
        for yk in y.elements():
            self.assertTrue(ca.depends_on(cov, yk))

    def test_exponential_kernel_ard_symbolic_call_sx_x_x_bar_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential(active_dims=[0, 2], length_scales=[1., 1.])

        x = ca.SX.sym('x', 3)
        y = ca.SX.sym('y', 3)
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        for length_scale in kernel.length_scales.SX.elements():
            self.assertTrue(ca.depends_on(cov, length_scale))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        for k, xk in enumerate(x.elements()):
            if k == 1:
                self.assertFalse(ca.depends_on(cov, xk))
            else:
                self.assertTrue(ca.depends_on(cov, xk))
        for k, yk in enumerate(y.elements()):
            if k == 1:
                self.assertFalse(ca.depends_on(cov, yk))
            else:
                self.assertTrue(ca.depends_on(cov, yk))

    # def test_exponential_kernel_ard_symbolic_call_mx_x_x_bar(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.exponential(length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 2)
    #     y = ca.MX.sym('y', 2)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     for length_scale in kernel.length_scales.MX.elements():
    #         self.assertTrue(ca.depends_on(cov, length_scale))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     for xk in x.elements():
    #         self.assertTrue(ca.depends_on(cov, xk))
    #     for yk in y.elements():
    #         self.assertTrue(ca.depends_on(cov, yk))

    # def test_exponential_kernel_ard_symbolic_call_mx_x_x_bar_not_all_active(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.exponential(active_dims=[0, 2], length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 3)
    #     y = ca.MX.sym('y', 3)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     for length_scale in kernel.length_scales.MX.elements():
    #         self.assertTrue(ca.depends_on(cov, length_scale))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     for k, xk in enumerate(x.elements()):
    #         if k == 1:
    #             self.assertFalse(ca.depends_on(cov, xk))
    #         else:
    #             self.assertTrue(ca.depends_on(cov, xk))
    #     for k, yk in enumerate(y.elements()):
    #         if k == 1:
    #             self.assertFalse(ca.depends_on(cov, yk))
    #         else:
    #             self.assertTrue(ca.depends_on(cov, yk))

    def test_exponential_kernel_ard_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential(length_scales=[1., 1.])

        x = np.array([[1., 1.1, 1.2], [1.3, 1.4, 1.5]])
        y = np.array([[1.6, 1.7, 1.8], [1.9, 2., 2.1]])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.42804449, .37159546, .32259073],
                                                  [.49306869, .42804449, .37159546],
                                                  [.56797071, .49306869, .42804449]]))

    def test_exponential_kernel_ard_numeric_call_x_x_bar_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.exponential(active_dims=[0, 2], length_scales=[1., 1.])

        x = np.array([[1., 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]])
        y = np.array([[1.9, 2., 2.1], [2.2, 2.3, 2.4], [2.5, 2.6, 2.7]])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.28004858, .24311673, .21105534],
                                                  [.32259073, .28004858, .24311673],
                                                  [.37159546, .32259073, .28004858]]))


class TestMatern32Kernel(TestCase):
    """"""
    def test_matern_32_kernel_no_hyperprior(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32()

        self.assertIsNone(kernel.active_dims)
        self.assertEqual(len(kernel.hyperparameters), 2)
        self.assertEqual(kernel.hyperparameter_names, ['M32.length_scales', 'M32.signal_variance'])
        self.assertTrue(hasattr(kernel.length_scales, 'log'))
        np.testing.assert_equal(kernel.length_scales.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.length_scales.log, np.zeros((1, 1)))
        self.assertTrue(hasattr(kernel.signal_variance, 'log'))
        np.testing.assert_equal(kernel.signal_variance.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.signal_variance.log, np.zeros((1, 1)))

    def test_matern_32_kernel_fixed(self) -> None:
        """

        :return:
        """
        # TODO: Change according to test_means.py when first TODO is finished
        kernel = Kernel.matern_32()
        kernel.length_scales.fixed = True
        kernel.signal_variance.fixed = True

        self.assertTrue(kernel.length_scales.fixed)
        self.assertTrue(kernel.signal_variance.fixed)

    # def test_matern_32_kernel_hyperprior_gaussian(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create according to test_means.py when first TODO is finished

    def test_matern_32_kernel_ard_no_active_dims(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            Kernel.matern_32(ard=True)
        self.assertEqual(str(context.exception),
                         "The key word 'ard' can only be set to True if the key word 'active_dims' was supplied")

    def test_matern_32_kernel_ard_dimension_mismatch(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            Kernel.matern_32(active_dims=[0, 1], length_scales=[1., 1., 1.])
        self.assertEqual(str(context.exception),
                         "Dimension mismatch between 'active_dims' (2) and the number of length_scales (3)")

    def test_matern_32_kernel_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32()

        x = ca.SX.sym('x')
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertFalse(ca.depends_on(cov, x))

    # def test_matern_32_kernel_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.matern_32()
    #
    #     x = ca.MX.sym('x')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    def test_matern_32_kernel_numeric_call(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32()

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[1., .48335772, .13973135, .03431324, .007767734],
                                                  [.48335772, 1., .48335772, .13973135, .03431324],
                                                  [.13973135, .48335772, 1., .48335772, .13973135],
                                                  [.03431324, .13973135, .48335772, 1., .48335772],
                                                  [.007767734, .03431324, .13973135, .48335772, 1.]]))

    def test_matern_32_kernel_symbolic_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32()

        x = ca.SX.sym('x')
        y = np.array([[2.]])
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_matern_32_kernel_symbolic_call_x_x_bar_dimension_mismatch(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32()

        x = ca.SX.sym('x', 2)
        y = ca.SX.sym('y')
        # FIXME: Convert to ValueError
        with self.assertRaises(AssertionError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar do not have the same input space dimensions")

    def test_matern_32_kernel_symbolic_call_x_x_bar_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32()

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertTrue(ca.depends_on(cov, x))
        self.assertTrue(ca.depends_on(cov, y))

    # def test_matern_32_kernel_symbolic_call_x_x_bar_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.matern_32()
    #
    #     x = ca.MX.sym('x')
    #     y = ca.MX.sym('y')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertTrue(ca.depends_on(cov, x))
    #     self.assertTrue(ca.depends_on(cov, y))

    def test_matern_32_kernel_numeric_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32()

        x = np.array([[2.]])
        y = ca.SX.sym('y')
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_matern_32_kernel_numeric_call_x_x_bar_dimension_mismatch(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32()

        x = np.array([[1.], [2.]])
        y = np.array([[1.]])
        # FIXME: Convert to ValueError
        with self.assertRaises(AssertionError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar do not have the same input space dimensions")

    def test_matern_32_kernel_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32()

        x = np.array([[1., 2., 3., 4., 5.]])
        y = np.array([[1., 2., 3., 4., 5.]]) / 2.
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(
            cov,
            np.array([[.78488765, 1., .78488765, .48335772, .26775661],
                      [.26775661, .48335772, .78488765, 1., .78488765],
                      [.07017579, .13973135, .26775661, .48335772, .78488765],
                      [.01645009, .03431324, .07017579, .13973135, .26775661],
                      [.003624159, .007767734, .01645009, .03431324, .07017579]])
                                   )

    def test_matern_32_kernel_ard(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32(active_dims=[0, 1], length_scales=[1., 1.])

        self.assertEqual(kernel.active_dims, [0, 1])
        np.testing.assert_equal(kernel.length_scales.value, np.ones((2, 1)))

        kernel = Kernel.matern_32(active_dims=[0, 1, 2], ard=True)

        self.assertEqual(kernel.active_dims, [0, 1, 2])
        np.testing.assert_equal(kernel.length_scales.value, np.ones((3, 1)))

    # def test_matern_32_kernel_ard_call_dimension_mismatch(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create similar test for means
    #     kernel = Kernel.matern_32(active_dims=[0, 1], length_scales=[1., 1.])
    #
    #     x = ca.SX.sym('x')
    #     # FIXME: This will result in another error that could be unclear to the user. We should probably catch it and
    #     #  return a more informative error message.
    #     cov = kernel(x)

    def test_matern_32_kernel_ard_call_dimension_mismatch(self) -> None:
        """

        :return:
        """
        # TODO: Create similar test for means
        kernel = Kernel.matern_32(length_scales=[1., 1.])

        x = ca.SX.sym('x')
        with self.assertRaises(ValueError) as context:
            kernel(x)
        self.assertEqual(str(context.exception), "Length scales vector dimension does not equal input space dimension.")

    def test_matern_32_kernel_ard_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32(length_scales=[1., 1.])

        x = ca.SX.sym('x', 2)
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertFalse(ca.depends_on(cov, x))

    def test_matern_32_kernel_ard_symbolic_call_sx_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32(active_dims=[0, 2], length_scales=[1., 1.])

        x = ca.SX.sym('x', 3)
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertFalse(ca.depends_on(cov, x))

    # def test_matern_32_kernel_ard_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.matern_32(length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 2)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    # def test_matern_32_kernel_ard_symbolic_call_mx_not_all_active(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.matern_32(active_dims=[0, 2], length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 3)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    def test_matern_32_kernel_ard_numeric_call(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32(length_scales=[1., 1.])

        x = np.array([[1., 2., 3., 4., 5.], [.1, .2, .3, .4, .5]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[1., .48071535, .13786943, .03357525, .007536611],
                                                  [.48071535, 1., .48071535, .13786943, .03357525],
                                                  [.13786943, .48071535, 1., .48071535, .13786943],
                                                  [.03357525, .13786943, .48071535, 1., .48071535],
                                                  [.007536611, .03357525, .13786943, .48071535, 1.]
                                                  ]))

    def test_matern_32_kernel_ard_numeric_call_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32(active_dims=[0, 2], length_scales=[1., 1.])

        x = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 0.], [.1, .2, .3, .4, .5]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[1., .48071535, .13786943, .03357525, .007536611],
                                                  [.48071535, 1., .48071535, .13786943, .03357525],
                                                  [.13786943, .48071535, 1., .48071535, .13786943],
                                                  [.03357525, .13786943, .48071535, 1., .48071535],
                                                  [.007536611, .03357525, .13786943, .48071535, 1.]
                                                  ]))

    def test_matern_32_kernel_ard_symbolic_call_sx_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32(length_scales=[1., 1.])

        x = ca.SX.sym('x', 2)
        y = ca.SX.sym('y', 2)
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        for length_scale in kernel.length_scales.SX.elements():
            self.assertTrue(ca.depends_on(cov, length_scale))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        for xk in x.elements():
            self.assertTrue(ca.depends_on(cov, xk))
        for yk in y.elements():
            self.assertTrue(ca.depends_on(cov, yk))

    def test_matern_32_kernel_ard_symbolic_call_sx_x_x_bar_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32(active_dims=[0, 2], length_scales=[1., 1.])

        x = ca.SX.sym('x', 3)
        y = ca.SX.sym('y', 3)
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        for length_scale in kernel.length_scales.SX.elements():
            self.assertTrue(ca.depends_on(cov, length_scale))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        for k, xk in enumerate(x.elements()):
            if k == 1:
                self.assertFalse(ca.depends_on(cov, xk))
            else:
                self.assertTrue(ca.depends_on(cov, xk))
        for k, yk in enumerate(y.elements()):
            if k == 1:
                self.assertFalse(ca.depends_on(cov, yk))
            else:
                self.assertTrue(ca.depends_on(cov, yk))

    # def test_matern_32_kernel_ard_symbolic_call_mx_x_x_bar(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.matern_32(length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 2)
    #     y = ca.MX.sym('y', 2)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     for length_scale in kernel.length_scales.MX.elements():
    #         self.assertTrue(ca.depends_on(cov, length_scale))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     for xk in x.elements():
    #         self.assertTrue(ca.depends_on(cov, xk))
    #     for yk in y.elements():
    #         self.assertTrue(ca.depends_on(cov, yk))

    # def test_matern_32_kernel_ard_symbolic_call_mx_x_x_bar_not_all_active(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.matern_32(active_dims=[0, 2], length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 3)
    #     y = ca.MX.sym('y', 3)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     for length_scale in kernel.length_scales.MX.elements():
    #         self.assertTrue(ca.depends_on(cov, length_scale))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     for k, xk in enumerate(x.elements()):
    #         if k == 1:
    #             self.assertFalse(ca.depends_on(cov, xk))
    #         else:
    #             self.assertTrue(ca.depends_on(cov, xk))
    #     for k, yk in enumerate(y.elements()):
    #         if k == 1:
    #             self.assertFalse(ca.depends_on(cov, yk))
    #         else:
    #             self.assertTrue(ca.depends_on(cov, yk))

    def test_matern_32_kernel_ard_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32(length_scales=[1., 1.])

        x = np.array([[1., 1.1, 1.2], [1.3, 1.4, 1.5]])
        y = np.array([[1.6, 1.7, 1.8], [1.9, 2., 2.1]])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.56801943, .48871175, .41705364],
                                                  [.65370269, .56801943, .48871175],
                                                  [.74319105, .65370269, .56801943]]))

    def test_matern_32_kernel_ard_numeric_call_x_x_bar_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_32(active_dims=[0, 2], length_scales=[1., 1.])

        x = np.array([[1., 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]])
        y = np.array([[1.9, 2., 2.1], [2.2, 2.3, 2.4], [2.5, 2.6, 2.7]])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.35346459, .29782077, .24967129],
                                                  [.41705364, .35346459, .29782077],
                                                  [.48871175, .41705364, .35346459]]))


class TestMatern52Kernel(TestCase):
    """"""
    def test_matern_52_kernel_no_hyperprior(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52()

        self.assertIsNone(kernel.active_dims)
        self.assertEqual(len(kernel.hyperparameters), 2)
        self.assertEqual(kernel.hyperparameter_names, ['M52.length_scales', 'M52.signal_variance'])
        self.assertTrue(hasattr(kernel.length_scales, 'log'))
        np.testing.assert_equal(kernel.length_scales.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.length_scales.log, np.zeros((1, 1)))
        self.assertTrue(hasattr(kernel.signal_variance, 'log'))
        np.testing.assert_equal(kernel.signal_variance.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.signal_variance.log, np.zeros((1, 1)))

    def test_matern_52_kernel_fixed(self) -> None:
        """

        :return:
        """
        # TODO: Change according to test_means.py when first TODO is finished
        kernel = Kernel.matern_52()
        kernel.length_scales.fixed = True
        kernel.signal_variance.fixed = True

        self.assertTrue(kernel.length_scales.fixed)
        self.assertTrue(kernel.signal_variance.fixed)

    # def test_matern_52_kernel_hyperprior_gaussian(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create according to test_means.py when first TODO is finished

    def test_matern_52_kernel_ard_no_active_dims(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            Kernel.matern_52(ard=True)
        self.assertEqual(str(context.exception),
                         "The key word 'ard' can only be set to True if the key word 'active_dims' was supplied")

    def test_matern_52_kernel_ard_dimension_mismatch(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            Kernel.matern_52(active_dims=[0, 1], length_scales=[1., 1., 1.])
        self.assertEqual(str(context.exception),
                         "Dimension mismatch between 'active_dims' (2) and the number of length_scales (3)")

    def test_matern_52_kernel_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52()

        x = ca.SX.sym('x')
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertFalse(ca.depends_on(cov, x))

    # def test_matern_52_kernel_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.matern_52()
    #
    #     x = ca.MX.sym('x')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    def test_matern_52_kernel_numeric_call(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52()

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[1., .52399411, .13866022, .02772342, .004777085],
                                                  [.52399411, 1., .52399411, .13866022, .02772342],
                                                  [.13866022, .52399411, 1., .52399411, .13866022],
                                                  [.02772342, .13866022, .52399411, 1., .52399411],
                                                  [.004777085, .02772342, .13866022, .52399411, 1.]]))

    def test_matern_52_kernel_symbolic_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52()

        x = ca.SX.sym('x')
        y = np.array([[2.]])
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_matern_52_kernel_symbolic_call_x_x_bar_dimension_mismatch(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52()

        x = ca.SX.sym('x', 2)
        y = ca.SX.sym('y')
        # FIXME: Convert to ValueError
        with self.assertRaises(AssertionError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar do not have the same input space dimensions")

    def test_matern_52_kernel_symbolic_call_x_x_bar_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52()

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertTrue(ca.depends_on(cov, x))
        self.assertTrue(ca.depends_on(cov, y))

    # def test_matern_52_kernel_symbolic_call_x_x_bar_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.matern_52()
    #
    #     x = ca.MX.sym('x')
    #     y = ca.MX.sym('y')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertTrue(ca.depends_on(cov, x))
    #     self.assertTrue(ca.depends_on(cov, y))

    def test_matern_52_kernel_numeric_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52()

        x = np.array([[2.]])
        y = ca.SX.sym('y')
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_matern_52_kernel_numeric_call_x_x_bar_dimension_mismatch(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52()

        x = np.array([[1.], [2.]])
        y = np.array([[1.]])
        # FIXME: Convert to ValueError
        with self.assertRaises(AssertionError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar do not have the same input space dimensions")

    def test_matern_52_kernel_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52()

        x = np.array([[1., 2., 3., 4., 5.]])
        y = np.array([[1., 2., 3., 4., 5.]]) / 2.
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(
            cov,
            np.array([[.82864914, 1., .82864914, .52399411, .28316327],
                      [.28316327, .52399411, .82864914, 1., .82864914],
                      [.06351021, .13866022, .28316327, .52399411, .82864914],
                      [.01167155, .02772342, .06351021, .13866022, .28316327],
                      [.001911584, .004777085, .01167155, .02772342, .06351021]])
                                   )

    def test_matern_52_kernel_ard(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52(active_dims=[0, 1], length_scales=[1., 1.])

        self.assertEqual(kernel.active_dims, [0, 1])
        np.testing.assert_equal(kernel.length_scales.value, np.ones((2, 1)))

        kernel = Kernel.matern_52(active_dims=[0, 1, 2], ard=True)

        self.assertEqual(kernel.active_dims, [0, 1, 2])
        np.testing.assert_equal(kernel.length_scales.value, np.ones((3, 1)))

    # def test_matern_52_kernel_ard_call_dimension_mismatch(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create similar test for means
    #     kernel = Kernel.matern_52(active_dims=[0, 1], length_scales=[1., 1.])
    #
    #     x = ca.SX.sym('x')
    #     # FIXME: This will result in another error that could be unclear to the user. We should probably catch it and
    #     #  return a more informative error message.
    #     cov = kernel(x)

    def test_matern_52_kernel_ard_call_dimension_mismatch(self) -> None:
        """

        :return:
        """
        # TODO: Create similar test for means
        kernel = Kernel.matern_52(length_scales=[1., 1.])

        x = ca.SX.sym('x')
        with self.assertRaises(ValueError) as context:
            kernel(x)
        self.assertEqual(str(context.exception), "Length scales vector dimension does not equal input space dimension.")

    def test_matern_52_kernel_ard_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52(length_scales=[1., 1.])

        x = ca.SX.sym('x', 2)
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertFalse(ca.depends_on(cov, x))

    def test_matern_52_kernel_ard_symbolic_call_sx_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52(active_dims=[0, 2], length_scales=[1., 1.])

        x = ca.SX.sym('x', 3)
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertFalse(ca.depends_on(cov, x))

    # def test_matern_52_kernel_ard_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.matern_52(length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 2)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    # def test_matern_52_kernel_ard_symbolic_call_mx_not_all_active(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.matern_52(active_dims=[0, 2], length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 3)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    def test_matern_52_kernel_ard_numeric_call(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52(length_scales=[1., 1.])

        x = np.array([[1., 2., 3., 4., 5.], [.1, .2, .3, .4, .5]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[1., .521123, .13659553, .027027814, .004607506],
                                                  [.521123, 1., .521123, .13659553, .027027814],
                                                  [.13659553, .521123, 1., .521123, .13659553],
                                                  [.027027814, .13659553, .521123, 1., .521123],
                                                  [.004607506, .027027814, .13659553, .521123, 1.]
                                                  ]))

    def test_matern_52_kernel_ard_numeric_call_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52(active_dims=[0, 2], length_scales=[1., 1.])

        x = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 0.], [.1, .2, .3, .4, .5]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[1., .521123, .13659553, .027027814, .004607506],
                                                  [.521123, 1., .521123, .13659553, .027027814],
                                                  [.13659553, .521123, 1., .521123, .13659553],
                                                  [.027027814, .13659553, .521123, 1., .521123],
                                                  [.004607506, .027027814, .13659553, .521123, 1.]
                                                  ]))

    def test_matern_52_kernel_ard_symbolic_call_sx_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52(length_scales=[1., 1.])

        x = ca.SX.sym('x', 2)
        y = ca.SX.sym('y', 2)
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        for length_scale in kernel.length_scales.SX.elements():
            self.assertTrue(ca.depends_on(cov, length_scale))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        for xk in x.elements():
            self.assertTrue(ca.depends_on(cov, xk))
        for yk in y.elements():
            self.assertTrue(ca.depends_on(cov, yk))

    def test_matern_52_kernel_ard_symbolic_call_sx_x_x_bar_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52(active_dims=[0, 2], length_scales=[1., 1.])

        x = ca.SX.sym('x', 3)
        y = ca.SX.sym('y', 3)
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        for length_scale in kernel.length_scales.SX.elements():
            self.assertTrue(ca.depends_on(cov, length_scale))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        for k, xk in enumerate(x.elements()):
            if k == 1:
                self.assertFalse(ca.depends_on(cov, xk))
            else:
                self.assertTrue(ca.depends_on(cov, xk))
        for k, yk in enumerate(y.elements()):
            if k == 1:
                self.assertFalse(ca.depends_on(cov, yk))
            else:
                self.assertTrue(ca.depends_on(cov, yk))

    # def test_matern_52_kernel_ard_symbolic_call_mx_x_x_bar(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.matern_52(length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 2)
    #     y = ca.MX.sym('y', 2)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     for length_scale in kernel.length_scales.MX.elements():
    #         self.assertTrue(ca.depends_on(cov, length_scale))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     for xk in x.elements():
    #         self.assertTrue(ca.depends_on(cov, xk))
    #     for yk in y.elements():
    #         self.assertTrue(ca.depends_on(cov, yk))

    # def test_matern_52_kernel_ard_symbolic_call_mx_x_x_bar_not_all_active(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.matern_52(active_dims=[0, 2], length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 3)
    #     y = ca.MX.sym('y', 3)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     for length_scale in kernel.length_scales.MX.elements():
    #         self.assertTrue(ca.depends_on(cov, length_scale))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     for k, xk in enumerate(x.elements()):
    #         if k == 1:
    #             self.assertFalse(ca.depends_on(cov, xk))
    #         else:
    #             self.assertTrue(ca.depends_on(cov, xk))
    #     for k, yk in enumerate(y.elements()):
    #         if k == 1:
    #             self.assertFalse(ca.depends_on(cov, yk))
    #         else:
    #             self.assertTrue(ca.depends_on(cov, yk))

    def test_matern_52_kernel_ard_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52(length_scales=[1., 1.])

        x = np.array([[1., 1.1, 1.2], [1.3, 1.4, 1.5]])
        y = np.array([[1.6, 1.7, 1.8], [1.9, 2., 2.1]])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.61445344, .52980338, .45120166],
                                                  [.70249576, .61445344, .52980338],
                                                  [.78984477, .70249576, .61445344]]))

    def test_matern_52_kernel_ard_numeric_call_x_x_bar_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.matern_52(active_dims=[0, 2], length_scales=[1., 1.])

        x = np.array([[1., 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]])
        y = np.array([[1.9, 2., 2.1], [2.2, 2.3, 2.4], [2.5, 2.6, 2.7]])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.38015048, .31728336, .26261993],
                                                  [.45120166, .38015048, .31728336],
                                                  [.52980338, .45120166, .38015048]]))


class TestRationalQuadraticKernel(TestCase):
    """"""
    def test_rational_quadratic_kernel_no_hyperprior(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic()

        self.assertIsNone(kernel.active_dims)
        self.assertTrue(hasattr(kernel.alpha, 'log'))
        np.testing.assert_equal(kernel.alpha.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.alpha.log, np.zeros((1, 1)))
        self.assertEqual(len(kernel.hyperparameters), 3)
        self.assertEqual(kernel.hyperparameter_names, ['RQ.length_scales', 'RQ.signal_variance', 'RQ.alpha'])
        self.assertTrue(hasattr(kernel.length_scales, 'log'))
        np.testing.assert_equal(kernel.length_scales.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.length_scales.log, np.zeros((1, 1)))
        self.assertTrue(hasattr(kernel.signal_variance, 'log'))
        np.testing.assert_equal(kernel.signal_variance.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.signal_variance.log, np.zeros((1, 1)))

    def test_rational_quadratic_kernel_fixed(self) -> None:
        """

        :return:
        """
        # TODO: Change according to test_means.py when first TODO is finished
        kernel = Kernel.rational_quadratic()
        kernel.alpha.fixed = True
        kernel.length_scales.fixed = True
        kernel.signal_variance.fixed = True

        self.assertTrue(kernel.alpha.fixed)
        self.assertTrue(kernel.length_scales.fixed)
        self.assertTrue(kernel.signal_variance.fixed)

    # def test_rational_quadratic_kernel_hyperprior_gaussian(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create according to test_means.py when first TODO is finished

    def test_rational_quadratic_kernel_ard_no_active_dims(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            Kernel.rational_quadratic(ard=True)
        self.assertEqual(str(context.exception),
                         "The key word 'ard' can only be set to True if the key word 'active_dims' was supplied")

    def test_rational_quadratic_kernel_ard_dimension_mismatch(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            Kernel.rational_quadratic(active_dims=[0, 1], length_scales=[1., 1., 1.])
        self.assertEqual(str(context.exception),
                         "Dimension mismatch between 'active_dims' (2) and the number of length_scales (3)")

    def test_rational_quadratic_kernel_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic()

        x = ca.SX.sym('x')
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.alpha.SX))
        self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertFalse(ca.depends_on(cov, x))

    # def test_rational_quadratic_kernel_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.rational_quadratic()
    #
    #     x = ca.MX.sym('x')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.alpha.MX))
    #     self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    def test_rational_quadratic_kernel_numeric_call(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic(alpha=.5)

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[1., .70710678, .4472136, .31622777, .24253563],
                                                  [.70710678, 1., .70710678, .4472136, .31622777],
                                                  [.4472136, .70710678, 1., .70710678, .4472136],
                                                  [.31622777, .4472136, .70710678, 1., .70710678],
                                                  [.24253563, .31622777, .4472136, .70710678, 1.]]))

    def test_rational_quadratic_kernel_symbolic_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic()

        x = ca.SX.sym('x')
        y = np.array([[2.]])
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_rational_quadratic_kernel_symbolic_call_x_x_bar_dimension_mismatch(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic()

        x = ca.SX.sym('x', 2)
        y = ca.SX.sym('y')
        # FIXME: Convert to ValueError
        with self.assertRaises(AssertionError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar do not have the same input space dimensions")

    def test_rational_quadratic_kernel_symbolic_call_x_x_bar_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic()

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.alpha.SX))
        self.assertTrue(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertTrue(ca.depends_on(cov, x))
        self.assertTrue(ca.depends_on(cov, y))

    # def test_rational_quadratic_kernel_symbolic_call_x_x_bar_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.rational_quadratic()
    #
    #     x = ca.MX.sym('x')
    #     y = ca.MX.sym('y')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.alpha.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertTrue(ca.depends_on(cov, x))
    #     self.assertTrue(ca.depends_on(cov, y))

    def test_rational_quadratic_kernel_numeric_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic()

        x = np.array([[2.]])
        y = ca.SX.sym('y')
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_rational_quadratic_kernel_numeric_call_x_x_bar_dimension_mismatch(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic()

        x = np.array([[1.], [2.]])
        y = np.array([[1.]])
        # FIXME: Convert to ValueError
        with self.assertRaises(AssertionError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar do not have the same input space dimensions")

    def test_rational_quadratic_kernel_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic(alpha=.5)

        x = np.array([[1., 2., 3., 4., 5.]])
        y = np.array([[1., 2., 3., 4., 5.]]) / 2.
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(
            cov,
            np.array([[.89442719, 1., .89442719, .70710678, .5547002],
                      [.5547002, .70710678, .89442719, 1., .89442719],
                      [.37139068, .4472136, .5547002, .70710678, .89442719],
                      [.27472113, .31622777, .37139068, .4472136, .5547002],
                      [.21693046, .24253563, .27472113, .31622777, .37139068]])
                                   )

    def test_rational_quadratic_kernel_ard(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic(active_dims=[0, 1], length_scales=[1., 1.])

        self.assertEqual(kernel.active_dims, [0, 1])
        np.testing.assert_equal(kernel.length_scales.value, np.ones((2, 1)))

        kernel = Kernel.rational_quadratic(active_dims=[0, 1, 2], ard=True)

        self.assertEqual(kernel.active_dims, [0, 1, 2])
        np.testing.assert_equal(kernel.length_scales.value, np.ones((3, 1)))

    # def test_rational_quadratic_kernel_ard_call_dimension_mismatch(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create similar test for means
    #     kernel = Kernel.rational_quadratic(active_dims=[0, 1], length_scales=[1., 1.])
    #
    #     x = ca.SX.sym('x')
    #     # FIXME: This will result in another error that could be unclear to the user. We should probably catch it and
    #     #  return a more informative error message.
    #     cov = kernel(x)

    def test_rational_quadratic_kernel_ard_call_dimension_mismatch(self) -> None:
        """

        :return:
        """
        # TODO: Create similar test for means
        kernel = Kernel.rational_quadratic(length_scales=[1., 1.])

        x = ca.SX.sym('x')
        with self.assertRaises(ValueError) as context:
            kernel(x)
        self.assertEqual(str(context.exception), "Length scales vector dimension does not equal input space dimension.")

    def test_rational_quadratic_kernel_ard_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic(length_scales=[1., 1.])

        x = ca.SX.sym('x', 2)
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.alpha.SX))
        self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertFalse(ca.depends_on(cov, x))

    def test_rational_quadratic_kernel_ard_symbolic_call_sx_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic(active_dims=[0, 2], length_scales=[1., 1.])

        x = ca.SX.sym('x', 3)
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.alpha.SX))
        self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertFalse(ca.depends_on(cov, x))

    # def test_rational_quadratic_kernel_ard_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.rational_quadratic(length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 2)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.alpha.MX))
    #     self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    # def test_rational_quadratic_kernel_ard_symbolic_call_mx_not_all_active(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.rational_quadratic(active_dims=[0, 2], length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 3)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.alpha.MX))
    #     self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    def test_rational_quadratic_kernel_ard_numeric_call(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic(length_scales=[1., 1.], alpha=.5)

        x = np.array([[1., 2., 3., 4., 5.], [.1, .2, .3, .4, .5]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[1., .70534562, .4454354, .31481428, .24140227],
                                                  [.70534562, 1., .70534562, .4454354, .31481428],
                                                  [.4454354, .70534562, 1., .70534562, .4454354],
                                                  [.31481428, .4454354, .70534562, 1., .70534562],
                                                  [.24140227, .31481428, .4454354, .70534562, 1.]
                                                  ]))

    def test_rational_quadratic_kernel_ard_numeric_call_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic(active_dims=[0, 2], length_scales=[1., 1.], alpha=.5)

        x = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 0.], [.1, .2, .3, .4, .5]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[1., .70534562, .4454354, .31481428, .24140227],
                                                  [.70534562, 1., .70534562, .4454354, .31481428],
                                                  [.4454354, .70534562, 1., .70534562, .4454354],
                                                  [.31481428, .4454354, .70534562, 1., .70534562],
                                                  [.24140227, .31481428, .4454354, .70534562, 1.]
                                                  ]))

    def test_rational_quadratic_kernel_ard_symbolic_call_sx_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic(length_scales=[1., 1.])

        x = ca.SX.sym('x', 2)
        y = ca.SX.sym('y', 2)
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.alpha.SX))
        for length_scale in kernel.length_scales.SX.elements():
            self.assertTrue(ca.depends_on(cov, length_scale))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        for xk in x.elements():
            self.assertTrue(ca.depends_on(cov, xk))
        for yk in y.elements():
            self.assertTrue(ca.depends_on(cov, yk))

    def test_rational_quadratic_kernel_ard_symbolic_call_sx_x_x_bar_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic(active_dims=[0, 2], length_scales=[1., 1.])

        x = ca.SX.sym('x', 3)
        y = ca.SX.sym('y', 3)
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.alpha.SX))
        for length_scale in kernel.length_scales.SX.elements():
            self.assertTrue(ca.depends_on(cov, length_scale))
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        for k, xk in enumerate(x.elements()):
            if k == 1:
                self.assertFalse(ca.depends_on(cov, xk))
            else:
                self.assertTrue(ca.depends_on(cov, xk))
        for k, yk in enumerate(y.elements()):
            if k == 1:
                self.assertFalse(ca.depends_on(cov, yk))
            else:
                self.assertTrue(ca.depends_on(cov, yk))

    # def test_rational_quadratic_kernel_ard_symbolic_call_mx_x_x_bar(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.rational_quadratic(length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 2)
    #     y = ca.MX.sym('y', 2)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.alpha.MX))
    #     for length_scale in kernel.length_scales.MX.elements():
    #         self.assertTrue(ca.depends_on(cov, length_scale))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     for xk in x.elements():
    #         self.assertTrue(ca.depends_on(cov, xk))
    #     for yk in y.elements():
    #         self.assertTrue(ca.depends_on(cov, yk))

    # def test_rational_quadratic_kernel_ard_symbolic_call_mx_x_x_bar_not_all_active(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.rational_quadratic(active_dims=[0, 2], length_scales=[1., 1.])
    #
    #     x = ca.MX.sym('x', 3)
    #     y = ca.MX.sym('y', 3)
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.alpha.MX))
    #     for length_scale in kernel.length_scales.MX.elements():
    #         self.assertTrue(ca.depends_on(cov, length_scale))
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     for k, xk in enumerate(x.elements()):
    #         if k == 1:
    #             self.assertFalse(ca.depends_on(cov, xk))
    #         else:
    #             self.assertTrue(ca.depends_on(cov, xk))
    #     for k, yk in enumerate(y.elements()):
    #         if k == 1:
    #             self.assertFalse(ca.depends_on(cov, yk))
    #         else:
    #             self.assertTrue(ca.depends_on(cov, yk))

    def test_rational_quadratic_kernel_ard_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic(length_scales=[1., 1.], alpha=.5)

        x = np.array([[1., 1.1, 1.2], [1.3, 1.4, 1.5]])
        y = np.array([[1.6, 1.7, 1.8], [1.9, 2., 2.1]])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.76249285, .71066905, .66226618],
                                                  [.81649658, .76249285, .71066905],
                                                  [.87038828, .81649658, .76249285]]))

    def test_rational_quadratic_kernel_ard_numeric_call_x_x_bar_not_all_active(self) -> None:
        """

        :return:
        """
        kernel = Kernel.rational_quadratic(active_dims=[0, 2], length_scales=[1., 1.], alpha=.5)

        x = np.array([[1., 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]])
        y = np.array([[1.9, 2., 2.1], [2.2, 2.3, 2.4], [2.5, 2.6, 2.7]])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.61780206, .57735027, .54073807],
                                                  [.66226618, .61780206, .57735027],
                                                  [.71066905, .66226618, .61780206]]))


class TestPiecewisePolynomialKernel(TestCase):
    """"""
    def test_piecewise_polynomial_kernel_no_hyperprior(self) -> None:
        """

        :return:
        """
        kernel = Kernel.piecewise_polynomial(1)

        self.assertIsNone(kernel.active_dims)
        self.assertEqual(kernel.degree, 1)
        self.assertEqual(len(kernel.hyperparameters), 2)
        self.assertEqual(kernel.hyperparameter_names, ['PP.length_scales', 'PP.signal_variance'])
        self.assertTrue(hasattr(kernel.length_scales, 'log'))
        np.testing.assert_equal(kernel.length_scales.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.length_scales.log, np.zeros((1, 1)))
        self.assertTrue(hasattr(kernel.signal_variance, 'log'))
        np.testing.assert_equal(kernel.signal_variance.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.signal_variance.log, np.zeros((1, 1)))

    def test_piecewise_polynomial_kernel_wrong_degree(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            Kernel.piecewise_polynomial(4)
        self.assertEqual(str(context.exception),
                         "The parameter 'degree' has to be one of the following integers: 0, 1, 2, 3")

    def test_piecewise_polynomial_kernel_degree_setter(self) -> None:
        """

        :return:
        """
        kernel = Kernel.piecewise_polynomial(1)
        self.assertEqual(kernel.degree, 1)

        kernel.degree = 2
        self.assertEqual(kernel.degree, 2)

        with self.assertRaises(ValueError) as context:
            kernel.degree = 4
        self.assertEqual(str(context.exception),
                         "The property 'degree' has to be one of the following integers: 0, 1, 2, 3")

    def test_piecewise_polynomial_kernel_fixed(self) -> None:
        """

        :return:
        """
        # TODO: Change according to test_means.py when first TODO is finished
        kernel = Kernel.piecewise_polynomial(1)
        kernel.length_scales.fixed = True
        kernel.signal_variance.fixed = True

        self.assertTrue(kernel.length_scales.fixed)
        self.assertTrue(kernel.signal_variance.fixed)

    # def test_piecewise_polynomial_kernel_hyperprior_gaussian(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create according to test_means.py when first TODO is finished

    def test_piecewise_polynomial_kernel_ard_no_active_dims(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            Kernel.piecewise_polynomial(1, ard=True)
        self.assertEqual(str(context.exception),
                         "The key word 'ard' can only be set to True if the key word 'active_dims' was supplied")

    def test_piecewise_polynomial_kernel_ard_dimension_mismatch(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            Kernel.piecewise_polynomial(1, active_dims=[0, 1], length_scales=[1., 1., 1.])
        self.assertEqual(str(context.exception),
                         "Dimension mismatch between 'active_dims' (2) and the number of length_scales (3)")

    def test_piecewise_polynomial_kernel_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        x = ca.SX.sym('x')

        val = None
        for k in range(4):
            kernel = Kernel.piecewise_polynomial(k, signal_variance=.5)
            cov = kernel(x)

            self.assertIsInstance(cov, ca.SX)
            self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
            self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
            self.assertFalse(ca.depends_on(cov, x))

            fun = ca.Function('fun', [kernel.signal_variance.SX], [cov])

            if val is not None:
                np.testing.assert_allclose(fun(kernel.signal_variance.log), val)
            val = fun(kernel.signal_variance.log)

    # def test_piecewise_polynomial_kernel_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     x = ca.MX.sym('x')
    #
    #     val = None
    #     for k in range(4):
    #         kernel = Kernel.piecewise_polynomial(k, signal_variance=.5)
    #         # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #         cov = kernel(x)
    #
    #         self.assertIsInstance(cov, ca.MX)
    #         self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #         self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #         self.assertFalse(ca.depends_on(cov, x))
    #
    #         fun = ca.Function('fun', [kernel.signal_variance.MX], [cov])
    #
    #         if val is not None:
    #             np.testing.assert_allclose(fun(kernel.signal_variance.log), val)
    #         val = fun(kernel.signal_variance.log)

    def test_piecewise_polynomial_kernel_numeric_call(self) -> None:
        """

        :return:
        """
        x = np.array([[1., 2., 3., 4., 5.]])

        val = [.25, .15625, .0859375, .04638672]
        for k in range(4):
            kernel = Kernel.piecewise_polynomial(k, length_scales=2., signal_variance=.5)
            cov = kernel(x)

            self.assertIsInstance(cov, np.ndarray)
            np.testing.assert_allclose(cov, np.array([[.5, val[k], 0., 0., 0.],
                                                      [val[k], .5, val[k], 0., 0.],
                                                      [0., val[k], .5, val[k], 0.],
                                                      [0., 0., val[k], .5, val[k]],
                                                      [0., 0., 0., val[k], .5]]))

    def test_piecewise_polynomial_kernel_symbolic_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.piecewise_polynomial(0)

        x = ca.SX.sym('x')
        y = np.array([[2.]])
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_piecewise_polynomial_kernel_symbolic_call_x_x_bar_dimension_mismatch(self) -> None:
        """

        :return:
        """
        kernel = Kernel.piecewise_polynomial(0)

        x = ca.SX.sym('x', 2)
        y = ca.SX.sym('y')
        # FIXME: Convert to ValueError
        with self.assertRaises(AssertionError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar do not have the same input space dimensions")

    def test_piecewise_polynomial_kernel_symbolic_call_x_x_bar_sx(self) -> None:
        """

        :return:
        """
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')

        val = None
        for k in range(4):
            kernel = Kernel.piecewise_polynomial(k, length_scales=2., signal_variance=.5)
            cov = kernel(x, y)

            self.assertIsInstance(cov, ca.SX)
            self.assertTrue(ca.depends_on(cov, kernel.length_scales.SX))
            self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
            self.assertTrue(ca.depends_on(cov, x))
            self.assertTrue(ca.depends_on(cov, y))

            fun = ca.Function('fun', [kernel.length_scales.SX, kernel.signal_variance.SX, x, y], [cov])

            if val is not None:
                np.testing.assert_array_less(fun(kernel.length_scales.log, kernel.signal_variance.log, 1., 2.), val)
            val = fun(kernel.length_scales.log, kernel.signal_variance.log, 1., 2.)

    # def test_piecewise_polynomial_kernel_symbolic_call_x_x_bar_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     x = ca.MX.sym('x')
    #     y = ca.MX.sym('y')
    #
    #     val = None
    #     for k in range(4):
    #         kernel = Kernel.piecewise_polynomial(k, length_scales=2., signal_variance=.5)
    #         # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #         cov = kernel(x, y)
    #
    #         self.assertIsInstance(cov, ca.MX)
    #         self.assertTrue(ca.depends_on(cov, kernel.length_scales.MX))
    #         self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #         self.assertTrue(ca.depends_on(cov, x))
    #         self.assertTrue(ca.depends_on(cov, y))
    #
    #         fun = ca.Function('fun', [kernel.length_scales.MX, kernel.signal_variance.MX, x, y], [cov])
    #
    #         if val is not None:
    #             np.testing.assert_array_less(fun(kernel.length_scales.log, kernel.signal_variance.log, 1., 2.), val)
    #         val = fun(kernel.length_scales.log, kernel.signal_variance.log, 1., 2.)

    def test_piecewise_polynomial_kernel_numeric_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.piecewise_polynomial(0)

        x = np.array([[2.]])
        y = ca.SX.sym('y')
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_piecewise_polynomial_kernel_numeric_call_x_x_bar_dimension_mismatch(self) -> None:
        """

        :return:
        """
        kernel = Kernel.piecewise_polynomial(0)

        x = np.array([[1.], [2.]])
        y = np.array([[1.]])
        # FIXME: Convert to ValueError
        with self.assertRaises(AssertionError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar do not have the same input space dimensions")

    def test_piecewise_polynomial_kernel_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        x = np.array([[1., 2., 3., 4., 5.]])
        y = np.array([[1., 2., 3., 4., 5.]]) / 2.

        val = [.5, .3125, .171875, .09277344]
        for k in range(4):
            kernel = Kernel.piecewise_polynomial(k)
            cov = kernel(x, y)

            self.assertIsInstance(cov, np.ndarray)
            np.testing.assert_allclose(cov, np.array([[val[k], 1., val[k], 0., 0.],
                                                      [0., 0., val[k], 1., val[k]],
                                                      [0., 0., 0., 0., val[k]],
                                                      [0., 0., 0., 0., 0.],
                                                      [0., 0., 0., 0., 0.]]))

    def test_piecewise_polynomial_kernel_ard(self) -> None:
        """

        :return:
        """
        kernel = Kernel.piecewise_polynomial(0, active_dims=[0, 1], length_scales=[1., 1.])

        self.assertEqual(kernel.active_dims, [0, 1])
        np.testing.assert_equal(kernel.length_scales.value, np.ones((2, 1)))

        kernel = Kernel.piecewise_polynomial(0, active_dims=[0, 1, 2], ard=True)

        self.assertEqual(kernel.active_dims, [0, 1, 2])
        np.testing.assert_equal(kernel.length_scales.value, np.ones((3, 1)))

    # def test_piecewise_polynomial_kernel_ard_call_dimension_mismatch(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create similar test for means
    #     kernel = Kernel.piecewise_polynomial(0, active_dims=[0, 1], length_scales=[1., 1.])
    #
    #     x = ca.SX.sym('x')
    #     # FIXME: This will result in another error that could be unclear to the user. We should probably catch it and
    #     #  return a more informative error message.
    #     cov = kernel(x)

    def test_piecewise_polynomial_kernel_ard_call_dimension_mismatch(self) -> None:
        """

        :return:
        """
        # TODO: Create similar test for means
        kernel = Kernel.piecewise_polynomial(0, length_scales=[1., 1.])

        x = ca.SX.sym('x')
        with self.assertRaises(ValueError) as context:
            kernel(x)
        self.assertEqual(str(context.exception), "Length scales vector dimension does not equal input space dimension.")

    def test_piecewise_polynomial_kernel_ard_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        x = ca.SX.sym('x', 2)

        val = None
        for k in range(4):
            kernel = Kernel.piecewise_polynomial(k, length_scales=[1., 1.], signal_variance=.5)
            cov = kernel(x)

            self.assertIsInstance(cov, ca.SX)
            self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
            self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
            self.assertFalse(ca.depends_on(cov, x))

            fun = ca.Function('fun', [kernel.signal_variance.SX], [cov])

            if val is not None:
                np.testing.assert_allclose(fun(kernel.signal_variance.log), val)
            val = fun(kernel.signal_variance.log)

    def test_piecewise_polynomial_kernel_ard_symbolic_call_sx_not_all_active(self) -> None:
        """

        :return:
        """
        x = ca.SX.sym('x', 3)

        val = None
        for k in range(4):
            kernel = Kernel.piecewise_polynomial(k, active_dims=[0, 2], length_scales=[1., 1.], signal_variance=.5)
            cov = kernel(x)

            self.assertIsInstance(cov, ca.SX)
            self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
            self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
            self.assertFalse(ca.depends_on(cov, x))

            fun = ca.Function('fun', [kernel.signal_variance.SX], [cov])

            if val is not None:
                np.testing.assert_allclose(fun(kernel.signal_variance.log), val)
            val = fun(kernel.signal_variance.log)

    # def test_piecewise_polynomial_kernel_ard_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     x = ca.MX.sym('x', 2)
    #
    #     val = None
    #     for k in range(4):
    #         kernel = Kernel.piecewise_polynomial(k, length_scales=[1., 1.], signal_variance=.5)
    #         # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #         cov = kernel(x)
    #
    #         self.assertIsInstance(cov, ca.MX)
    #         self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #         self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #         self.assertFalse(ca.depends_on(cov, x))
    #
    #         fun = ca.Function('fun', [kernel.signal_variance.MX], [cov])
    #
    #         if val is not None:
    #             np.testing.assert_allclose(fun(kernel.signal_variance.log), val)
    #         val = fun(kernel.signal_variance.log)

    # def test_piecewise_polynomial_kernel_ard_symbolic_call_mx_not_all_active(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     x = ca.MX.sym('x', 3)
    #
    #     val = None
    #     for k in range(4):
    #         kernel = Kernel.piecewise_polynomial(k, active_dims=[0, 2], length_scales=[1., 1.], signal_variance=.5)
    #         # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #         cov = kernel(x)
    #
    #         self.assertIsInstance(cov, ca.MX)
    #         self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #         self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #         self.assertFalse(ca.depends_on(cov, x))
    #
    #         fun = ca.Function('fun', [kernel.signal_variance.MX], [cov])
    #
    #         if val is not None:
    #             np.testing.assert_allclose(fun(kernel.signal_variance.log), val)
    #         val = fun(kernel.signal_variance.log)

    def test_piecewise_polynomial_kernel_ard_numeric_call(self) -> None:
        """

        :return:
        """
        x = np.array([[1., 2., 3., 4., 5.], [.1, .2, .3, .4, .5]])

        val = [.12375622, .09219916, .052774, .02888485]
        for k in range(4):
            kernel = Kernel.piecewise_polynomial(k, length_scales=[2., 2.], signal_variance=.5)
            cov = kernel(x)

            self.assertIsInstance(cov, np.ndarray)
            np.testing.assert_allclose(cov, np.array([[.5, val[k], 0., 0., 0.],
                                                      [val[k], .5, val[k], 0., 0.],
                                                      [0., val[k], .5, val[k], 0.],
                                                      [0., 0., val[k], .5, val[k]],
                                                      [0., 0., 0., val[k], .5]
                                                      ]))

    def test_piecewise_polynomial_kernel_ard_numeric_call_not_all_active(self) -> None:
        """

        :return:
        """
        x = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 0.], [.1, .2, .3, .4, .5]])

        val = [.12375622, .09219916, .052774, .02888485]
        for k in range(4):
            kernel = Kernel.piecewise_polynomial(k, active_dims=[0, 2], length_scales=[2., 2.], signal_variance=.5)
            cov = kernel(x)

            self.assertIsInstance(cov, np.ndarray)
            np.testing.assert_allclose(cov, np.array([[.5, val[k], 0., 0., 0.],
                                                      [val[k], .5, val[k], 0., 0.],
                                                      [0., val[k], .5, val[k], 0.],
                                                      [0., 0., val[k], .5, val[k]],
                                                      [0., 0., 0., val[k], .5]
                                                      ]))

    def test_piecewise_polynomial_kernel_ard_symbolic_call_sx_x_x_bar(self) -> None:
        """

        :return:
        """
        x = ca.SX.sym('x', 2)
        y = ca.SX.sym('y', 2)

        x_val = np.array([[1.], [1.]])
        y_val = np.array([[2.], [2.]])

        val = None
        for k in range(4):
            kernel = Kernel.piecewise_polynomial(k, length_scales=[2., 2.], signal_variance=.5)
            cov = kernel(x, y)

            self.assertIsInstance(cov, ca.SX)
            for length_scale in kernel.length_scales.SX.elements():
                self.assertTrue(ca.depends_on(cov, length_scale))
            self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
            for xk in x.elements():
                self.assertTrue(ca.depends_on(cov, xk))
            for yk in y.elements():
                self.assertTrue(ca.depends_on(cov, yk))

            fun = ca.Function('fun', [kernel.length_scales.SX, kernel.signal_variance.SX, x, y], [cov])

            if val is not None:
                np.testing.assert_array_less(fun(kernel.length_scales.log, kernel.signal_variance.log, x_val, y_val),
                                             val)
            val = fun(kernel.length_scales.log, kernel.signal_variance.log, x_val, y_val)

    def test_piecewise_polynomial_kernel_ard_symbolic_call_sx_x_x_bar_not_all_active(self) -> None:
        """

        :return:
        """
        x = ca.SX.sym('x', 3)
        y = ca.SX.sym('y', 3)

        x_val = np.array([[1.], [1.], [1.]])
        y_val = np.array([[2.], [2.], [2.]])

        val = None
        for k in range(4):
            kernel = Kernel.piecewise_polynomial(k, active_dims=[0, 2], length_scales=[2., 2.], signal_variance=.5)
            cov = kernel(x, y)

            self.assertIsInstance(cov, ca.SX)
            for length_scale in kernel.length_scales.SX.elements():
                self.assertTrue(ca.depends_on(cov, length_scale))
            self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
            for kk, xk in enumerate(x.elements()):
                if kk == 1:
                    self.assertFalse(ca.depends_on(cov, xk))
                else:
                    self.assertTrue(ca.depends_on(cov, xk))
            for kk, yk in enumerate(y.elements()):
                if kk == 1:
                    self.assertFalse(ca.depends_on(cov, yk))
                else:
                    self.assertTrue(ca.depends_on(cov, yk))

            fun = ca.Function('fun', [kernel.length_scales.SX, kernel.signal_variance.SX, x, y], [cov])

            if val is not None:
                np.testing.assert_array_less(fun(kernel.length_scales.log, kernel.signal_variance.log, x_val, y_val),
                                             val)
            val = fun(kernel.length_scales.log, kernel.signal_variance.log, x_val, y_val)

    # def test_piecewise_polynomial_kernel_ard_symbolic_call_mx_x_x_bar(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     x = ca.MX.sym('x', 2)
    #     y = ca.MX.sym('y', 2)
    #
    #     x_val = np.array([[1.], [1.]])
    #     y_val = np.array([[2.], [2.]])
    #
    #     val = None
    #     for k in range(4):
    #         kernel = Kernel.piecewise_polynomial(k, length_scales=[2., 2.], signal_variance=.5)
    #         # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #         cov = kernel(x, y)
    #
    #         self.assertIsInstance(cov, ca.MX)
    #         for length_scale in kernel.length_scales.MX.elements():
    #             self.assertTrue(ca.depends_on(cov, length_scale))
    #         self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #         for xk in x.elements():
    #             self.assertTrue(ca.depends_on(cov, xk))
    #         for yk in y.elements():
    #             self.assertTrue(ca.depends_on(cov, yk))
    #
    #         fun = ca.Function('fun', [kernel.length_scales.MX, kernel.signal_variance.MX, x, y], [cov])
    #
    #         if val is not None:
    #             np.testing.assert_array_less(fun(kernel.length_scales.log, kernel.signal_variance.log, x_val, y_val),
    #                                          val)
    #         val = fun(kernel.length_scales.log, kernel.signal_variance.log, x_val, y_val)

    # def test_piecewise_polynomial_kernel_ard_symbolic_call_mx_x_x_bar_not_all_active(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     x = ca.MX.sym('x', 3)
    #     y = ca.MX.sym('y', 3)
    #
    #     x_val = np.array([[1.], [1.], [1.]])
    #     y_val = np.array([[2.], [2.], [2.]])
    #
    #     val = None
    #     for k in range(4):
    #         kernel = Kernel.piecewise_polynomial(k, active_dims=[0, 2], length_scales=[2., 2.], signal_variance=.5)
    #         # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #         cov = kernel(x, y)
    #
    #         self.assertIsInstance(cov, ca.MX)
    #         for length_scale in kernel.length_scales.MX.elements():
    #             self.assertTrue(ca.depends_on(cov, length_scale))
    #         self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #         for kk, xk in enumerate(x.elements()):
    #             if kk == 1:
    #                 self.assertFalse(ca.depends_on(cov, xk))
    #             else:
    #                 self.assertTrue(ca.depends_on(cov, xk))
    #         for kk, yk in enumerate(y.elements()):
    #             if kk == 1:
    #                 self.assertFalse(ca.depends_on(cov, yk))
    #             else:
    #                 self.assertTrue(ca.depends_on(cov, yk))
    #
    #         fun = ca.Function('fun', [kernel.length_scales.MX, kernel.signal_variance.MX, x, y], [cov])
    #
    #         if val is not None:
    #             np.testing.assert_array_less(fun(kernel.length_scales.log, kernel.signal_variance.log, x_val, y_val),
    #                                          val)
    #         val = fun(kernel.length_scales.log, kernel.signal_variance.log, x_val, y_val)

    def test_piecewise_polynomial_kernel_ard_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        x = np.array([[1., 1.1, 1.2], [1.3, 1.4, 1.5]])
        y = np.array([[1.6, 1.7, 1.8], [1.9, 2., 2.1]])

        kernel = Kernel.piecewise_polynomial(0, length_scales=[2., 2.])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.33147186, .25505051, .18862915],
                                                  [.41789322, .33147186, .25505051],
                                                  [.51431458, .41789322, .33147186]]))

        kernel = Kernel.piecewise_polynomial(1, length_scales=[2., 2.])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.29633527, .1938447, .11609147],
                                                  [.42160556, .29633527, .1938447],
                                                  [.56378911, .42160556, .29633527]]))

        kernel = Kernel.piecewise_polynomial(2, length_scales=[2., 2.])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.20561221, .11328793, .05454821],
                                                  [.33421706, .20561221, .11328793],
                                                  [.4939008, .33421706, .20561221]]))

        kernel = Kernel.piecewise_polynomial(3, length_scales=[2., 2.])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.13687316, .06332761, .024456803],
                                                  [.25519039, .13687316, .06332761],
                                                  [.41890106, .25519039, .13687316]]))

    def test_piecewise_polynomial_kernel_ard_numeric_call_x_x_bar_not_all_active(self) -> None:
        """

        :return:
        """
        x = np.array([[1., 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]])
        y = np.array([[1.9, 2., 2.1], [2.2, 2.3, 2.4], [2.5, 2.6, 2.7]])

        kernel = Kernel.piecewise_polynomial(0, active_dims=[0, 2], length_scales=[2., 2.])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.13220779, .08578644, .04936508],
                                                  [.18862915, .13220779, .08578644],
                                                  [.25505051, .18862915, .13220779]]))

        kernel = Kernel.piecewise_polynomial(1, active_dims=[0, 2], length_scales=[2., 2.])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.06197292, .028174593, .0100188],
                                                  [.11609147, .06197292, .028174593],
                                                  [.1938447, .11609147, .06197292]]))

        kernel = Kernel.piecewise_polynomial(2, active_dims=[0, 2], length_scales=[2., 2.])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.022053282, .006992586, .0015308248],
                                                  [.05454821, .022053282, .006992586],
                                                  [.11328793, .05454821, .022053282]]))

        kernel = Kernel.piecewise_polynomial(3, active_dims=[0, 2], length_scales=[2., 2.])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.007474003, .00165027, .0002221374],
                                                  [.024456803, .007474003, .00165027],
                                                  [.06332761, .024456803, .007474003]]))


class TestPolynomialKernel(TestCase):
    """"""
    def test_polynomial_kernel_no_hyperprior(self) -> None:
        """

        :return:
        """
        kernel = Kernel.polynomial(1)

        self.assertIsNone(kernel.active_dims)
        self.assertEqual(kernel.degree, 1)
        self.assertEqual(len(kernel.hyperparameters), 2)
        self.assertEqual(kernel.hyperparameter_names, ['Poly.signal_variance', 'Poly.offset'])
        self.assertTrue(hasattr(kernel.signal_variance, 'log'))
        np.testing.assert_equal(kernel.signal_variance.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.signal_variance.log, np.zeros((1, 1)))
        self.assertTrue(hasattr(kernel.offset, 'log'))
        np.testing.assert_equal(kernel.offset.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.offset.log, np.zeros((1, 1)))

    def test_polynomial_kernel_degree_setter(self) -> None:
        """

        :return:
        """
        kernel = Kernel.polynomial(1)
        self.assertEqual(kernel.degree, 1)

        kernel.degree = 2
        self.assertEqual(kernel.degree, 2)

    def test_polynomial_kernel_fixed(self) -> None:
        """

        :return:
        """
        # TODO: Change according to test_means.py when first TODO is finished
        kernel = Kernel.polynomial(1)
        kernel.signal_variance.fixed = True
        kernel.offset.fixed = True

        self.assertTrue(kernel.signal_variance.fixed)
        self.assertTrue(kernel.offset.fixed)

    # def test_polynomial_kernel_hyperprior_gaussian(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create according to test_means.py when first TODO is finished

    def test_polynomial_kernel_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.polynomial(2)

        x = ca.SX.sym('x')
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertTrue(ca.depends_on(cov, kernel.offset.SX))
        self.assertTrue(ca.depends_on(cov, x))

    # def test_polynomial_kernel_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.polynomial(2)
    #
    #     x = ca.MX.sym('x')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.offset.MX))
    #     self.assertTrue(ca.depends_on(cov, x))

    def test_polynomial_kernel_numeric_call(self) -> None:
        """

        :return:
        """
        kernel = Kernel.polynomial(2)

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[4., 9., 16., 25., 36.],
                                                  [9., 25., 49., 81., 121.],
                                                  [16., 49., 100., 169., 256.],
                                                  [25., 81., 169., 289., 441.],
                                                  [36., 121., 256., 441., 676.]]))

    def test_polynomial_kernel_symbolic_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.polynomial(2)

        x = ca.SX.sym('x')
        y = np.array([[2.]])
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_polynomial_kernel_symbolic_call_x_x_bar_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.polynomial(2)

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertTrue(ca.depends_on(cov, kernel.offset.SX))
        self.assertTrue(ca.depends_on(cov, x))
        self.assertTrue(ca.depends_on(cov, y))

    # def test_polynomial_kernel_symbolic_call_x_x_bar_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.polynomial(2)
    #
    #     x = ca.MX.sym('x')
    #     y = ca.MX.sym('y')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.offset.MX))
    #     self.assertTrue(ca.depends_on(cov, x))
    #     self.assertTrue(ca.depends_on(cov, y))

    def test_polynomial_kernel_numeric_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.polynomial(2)

        x = np.array([[2.]])
        y = ca.SX.sym('y')
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_polynomial_kernel_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.polynomial(2)

        x = np.array([[1., 2., 3., 4., 5.]])
        y = np.array([[.6, .7, .8, .9, 0.]])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[2.56, 2.89, 3.24, 3.61, 1.],
                                                  [4.84, 5.76, 6.76, 7.84, 1.],
                                                  [7.84, 9.61, 11.56, 13.69, 1.],
                                                  [11.56, 14.44, 17.64, 21.16, 1.],
                                                  [16., 20.25, 25., 30.25, 1.]]))


class TestLinearKernel(TestCase):
    """"""
    def test_linear_kernel_no_hyperprior(self) -> None:
        """

        :return:
        """
        kernel = Kernel.linear()

        self.assertIsNone(kernel.active_dims)
        # TODO: Disable degree setter
        self.assertEqual(kernel.degree, 1)
        self.assertEqual(len(kernel.hyperparameters), 1)
        self.assertEqual(kernel.hyperparameter_names, ['Lin.signal_variance'])
        self.assertTrue(hasattr(kernel.signal_variance, 'log'))
        np.testing.assert_equal(kernel.signal_variance.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.signal_variance.log, np.zeros((1, 1)))
        self.assertFalse(hasattr(kernel.offset, 'log'))
        np.testing.assert_equal(kernel.offset, 0.)

    def test_linear_kernel_fixed(self) -> None:
        """

        :return:
        """
        # TODO: Change according to test_means.py when first TODO is finished
        kernel = Kernel.linear()
        kernel.signal_variance.fixed = True

        self.assertTrue(kernel.signal_variance.fixed)

    # def test_linear_kernel_hyperprior_gaussian(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create according to test_means.py when first TODO is finished

    def test_linear_kernel_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.linear()

        x = ca.SX.sym('x')
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertTrue(ca.depends_on(cov, x))

    # def test_linear_kernel_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.linear()
    #
    #     x = ca.MX.sym('x')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertTrue(ca.depends_on(cov, x))

    def test_linear_kernel_numeric_call(self) -> None:
        """

        :return:
        """
        kernel = Kernel.linear()

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[1., 2., 3., 4., 5.],
                                                  [2., 4., 6., 8., 10.],
                                                  [3., 6., 9., 12., 15.],
                                                  [4., 8., 12., 16., 20.],
                                                  [5., 10., 15., 20., 25.]]))

    def test_linear_kernel_symbolic_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.linear()

        x = ca.SX.sym('x')
        y = np.array([[2.]])
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_linear_kernel_symbolic_call_x_x_bar_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.linear()

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertTrue(ca.depends_on(cov, x))
        self.assertTrue(ca.depends_on(cov, y))

    # def test_linear_kernel_symbolic_call_x_x_bar_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.linear()
    #
    #     x = ca.MX.sym('x')
    #     y = ca.MX.sym('y')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertTrue(ca.depends_on(cov, x))
    #     self.assertTrue(ca.depends_on(cov, y))

    def test_linear_kernel_numeric_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.linear()

        x = np.array([[2.]])
        y = ca.SX.sym('y')
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_linear_kernel_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.linear()

        x = np.array([[1., 2., 3., 4., 5.]])
        y = np.array([[.6, .7, .8, .9, 0.]])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.6, .7, .8, .9, 0.],
                                                  [1.2, 1.4, 1.6, 1.8, 0.],
                                                  [1.8, 2.1, 2.4, 2.7, 0.],
                                                  [2.4, 2.8, 3.2, 3.6, 0.],
                                                  [3., 3.5, 4., 4.5, 0.]]))


class TestNeuralNetworkKernel(TestCase):
    """"""
    def test_neural_network_kernel_no_hyperprior(self) -> None:
        """

        :return:
        """
        kernel = Kernel.neural_network()

        self.assertIsNone(kernel.active_dims)
        # TODO: Disable degree setter
        self.assertEqual(len(kernel.hyperparameters), 2)
        self.assertEqual(kernel.hyperparameter_names, ['NN.signal_variance', 'NN.weight_variance'])
        self.assertTrue(hasattr(kernel.signal_variance, 'log'))
        np.testing.assert_equal(kernel.signal_variance.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.signal_variance.log, np.zeros((1, 1)))
        self.assertTrue(hasattr(kernel.weight_variance, 'log'))
        np.testing.assert_equal(kernel.weight_variance.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.weight_variance.log, np.zeros((1, 1)))

    def test_neural_network_kernel_fixed(self) -> None:
        """

        :return:
        """
        # TODO: Change according to test_means.py when first TODO is finished
        kernel = Kernel.neural_network()
        kernel.signal_variance.fixed = True
        kernel.weight_variance.fixed = True

        self.assertTrue(kernel.signal_variance.fixed)
        self.assertTrue(kernel.weight_variance.fixed)

    # def test_neural_network_kernel_hyperprior_gaussian(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create according to test_means.py when first TODO is finished

    def test_neural_network_kernel_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.neural_network()

        x = ca.SX.sym('x')
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertTrue(ca.depends_on(cov, kernel.weight_variance.SX))
        self.assertTrue(ca.depends_on(cov, x))

    # def test_neural_network_kernel_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.neural_network()
    #
    #     x = ca.MX.sym('x')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.weight_variance.MX))
    #     self.assertTrue(ca.depends_on(cov, x))

    def test_neural_network_kernel_numeric_call(self) -> None:
        """

        :return:
        """
        kernel = Kernel.neural_network()

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.72972766, .78539816, .77024433, .74832717, .72972766],
                                                  [.78539816, .98511078, 1.03849312, 1.04719755, 1.04364093],
                                                  [.77024433, 1.03849312, 1.14109666, 1.17807174, 1.19012163],
                                                  [.74832717, 1.04719755, 1.17807174, 1.23590017, 1.26160301],
                                                  [.72972766, 1.04364093, 1.19012163, 1.26160301, 1.2977837]]))

    def test_neural_network_kernel_symbolic_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.neural_network()

        x = ca.SX.sym('x')
        y = np.array([[2.]])
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_neural_network_kernel_symbolic_call_x_x_bar_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.neural_network()

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertTrue(ca.depends_on(cov, kernel.weight_variance.SX))
        self.assertTrue(ca.depends_on(cov, x))
        self.assertTrue(ca.depends_on(cov, y))

    # def test_neural_network_kernel_symbolic_call_x_x_bar_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.neural_network()
    #
    #     x = ca.MX.sym('x')
    #     y = ca.MX.sym('y')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.weight_variance.MX))
    #     self.assertTrue(ca.depends_on(cov, x))
    #     self.assertTrue(ca.depends_on(cov, y))

    def test_neural_network_kernel_numeric_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.neural_network()

        x = np.array([[2.]])
        y = ca.SX.sym('y')
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_neural_network_kernel_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.neural_network()

        x = np.array([[1., 2., 3., 4., 5.]])
        y = np.array([[.6, .7, .8, .9, 0.]])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.64514815, .67129112, .69398059, .71338192, .42053434],
                                                  [.62444045, .66991635, .71190145, .75037546, .29284277],
                                                  [.58182321, .63395099, .68275027, .72817205, .21484983],
                                                  [.54879429, .60359622, .65514193, .70337798, .16744808],
                                                  [.52486635, .58095355, .63381639, .68340053, .13650631]]))


class TestPeriodicKernel(TestCase):
    """"""
    def test_periodic_kernel_no_hyperprior(self) -> None:
        """

        :return:
        """
        kernel = Kernel.periodic()

        self.assertIsNone(kernel.active_dims)
        # TODO: Disable degree setter
        self.assertEqual(len(kernel.hyperparameters), 3)
        self.assertEqual(kernel.hyperparameter_names, ['Periodic.signal_variance', 'Periodic.length_scales',
                                                       'Periodic.period'])
        self.assertTrue(hasattr(kernel.signal_variance, 'log'))
        np.testing.assert_equal(kernel.signal_variance.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.signal_variance.log, np.zeros((1, 1)))
        self.assertTrue(hasattr(kernel.length_scales, 'log'))
        np.testing.assert_equal(kernel.length_scales.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.length_scales.log, np.zeros((1, 1)))
        self.assertTrue(hasattr(kernel.period, 'log'))
        np.testing.assert_equal(kernel.period.value, np.ones((1, 1)))
        np.testing.assert_equal(kernel.period.log, np.zeros((1, 1)))

    def test_periodic_kernel_fixed(self) -> None:
        """

        :return:
        """
        # TODO: Change according to test_means.py when first TODO is finished
        kernel = Kernel.periodic()
        kernel.signal_variance.fixed = True
        kernel.length_scales.fixed = True
        kernel.period.fixed = True

        self.assertTrue(kernel.signal_variance.fixed)
        self.assertTrue(kernel.length_scales.fixed)
        self.assertTrue(kernel.period.fixed)

    # def test_periodic_kernel_hyperprior_gaussian(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Create according to test_means.py when first TODO is finished

    def test_periodic_kernel_symbolic_call_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.periodic()

        x = ca.SX.sym('x')
        cov = kernel(x)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertFalse(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertFalse(ca.depends_on(cov, kernel.period.SX))
        self.assertFalse(ca.depends_on(cov, x))

    # def test_periodic_kernel_symbolic_call_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.periodic()
    #
    #     x = ca.MX.sym('x')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertFalse(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertFalse(ca.depends_on(cov, kernel.period.MX))
    #     self.assertFalse(ca.depends_on(cov, x))

    def test_periodic_kernel_numeric_call(self) -> None:
        """

        :return:
        """
        kernel = Kernel.periodic(signal_variance=.5)

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, .5 * np.ones((5, 5)))

    def test_periodic_kernel_symbolic_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.periodic()

        x = ca.SX.sym('x')
        y = np.array([[2.]])
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_periodic_kernel_symbolic_call_x_x_bar_sx(self) -> None:
        """

        :return:
        """
        kernel = Kernel.periodic()

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        cov = kernel(x, y)

        self.assertIsInstance(cov, ca.SX)
        self.assertTrue(ca.depends_on(cov, kernel.signal_variance.SX))
        self.assertTrue(ca.depends_on(cov, kernel.length_scales.SX))
        self.assertTrue(ca.depends_on(cov, kernel.period.SX))
        self.assertTrue(ca.depends_on(cov, x))
        self.assertTrue(ca.depends_on(cov, y))

    # def test_periodic_kernel_symbolic_call_x_x_bar_mx(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kernel = Kernel.periodic()
    #
    #     x = ca.MX.sym('x')
    #     y = ca.MX.sym('y')
    #     # FIXME: This will result in a mixture of SX and MX (should we remove MX completely?)
    #     cov = kernel(x, y)
    #
    #     self.assertIsInstance(cov, ca.MX)
    #     self.assertTrue(ca.depends_on(cov, kernel.signal_variance.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.length_scales.MX))
    #     self.assertTrue(ca.depends_on(cov, kernel.period.MX))
    #     self.assertTrue(ca.depends_on(cov, x))
    #     self.assertTrue(ca.depends_on(cov, y))

    def test_periodic_kernel_numeric_call_x_x_bar_wrong_type(self) -> None:
        """

        :return:
        """
        kernel = Kernel.periodic()

        x = np.array([[2.]])
        y = ca.SX.sym('y')
        # FIXME: Convert to TypeError
        with self.assertRaises(ValueError) as context:
            kernel(x, y)
        self.assertEqual(str(context.exception), "X and X_bar need to have the same type")

    def test_periodic_kernel_numeric_call_x_x_bar(self) -> None:
        """

        :return:
        """
        kernel = Kernel.periodic(signal_variance=.5, length_scales=2., period=.5)

        x = np.array([[1., 2., 3., 4., 5.]])
        y = np.array([[.6, .7, .8, .9, 0.]])
        cov = kernel(x, y)

        self.assertIsInstance(cov, np.ndarray)
        np.testing.assert_allclose(cov, np.array([[.42067575, .3180962, .3180962, .42067575, .5],
                                                  [.42067575, .3180962, .3180962, .42067575, .5],
                                                  [.42067575, .3180962, .3180962, .42067575, .5],
                                                  [.42067575, .3180962, .3180962, .42067575, .5],
                                                  [.42067575, .3180962, .3180962, .42067575, .5]]))


class TestKernelOperators(TestCase):
    """"""
    def test_kernel_sum(self) -> None:
        """

        :return:
        """
        from hilo_mpc import LinearKernel, ConstantKernel

        kernel = LinearKernel() + ConstantKernel()

        self.assertIsNone(kernel.active_dims)
        self.assertEqual(len(kernel.hyperparameters), 2)
        self.assertEqual(kernel.hyperparameter_names, ['Lin.signal_variance', 'Const.bias'])
        self.assertIsInstance(kernel.kernel_1, LinearKernel)
        self.assertIsInstance(kernel.kernel_2, ConstantKernel)

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        np.testing.assert_allclose(cov, np.array([[2., 3., 4., 5., 6.],
                                                  [3., 5., 7., 9., 11.],
                                                  [4., 7., 10., 13., 16.],
                                                  [5., 9., 13., 17., 21.],
                                                  [6., 11., 16., 21., 26.]]))

    # def test_kernel_scale(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Implement Scale operator
    #     from hilo_mpc import ConstantKernel
    #
    #     kernel = 2. * ConstantKernel()
    #
    #     self.assertIsNone(kernel.active_dims)
    #     self.assertEqual(len(kernel.hyperparameters), 1)
    #     self.assertEqual(kernel.hyperparameter_names, ['Const.bias'])
    #     self.assertIsInstance(kernel.kernel_1, ConstantKernel)
    #     self.assertIsNone(kernel.kernel_2)
    #     np.testing.assert_equal(kernel.scale, 2.)
    #
    #     x = np.array([[1., 2., 3., 4., 5.]])
    #     cov = kernel(x)
    #
    #     np.testing.assert_allclose(cov, np.array([[2., 2., 2., 2., 2.]]))

    # def test_kernel_scale_from_the_right(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     # TODO: Implement Scale operator
    #     from hilo_mpc import ConstantKernel
    #
    #     kernel = ConstantKernel() * 2.
    #
    #     self.assertIsNone(kernel.active_dims)
    #     self.assertEqual(len(kernel.hyperparameters), 1)
    #     self.assertEqual(kernel.hyperparameter_names, ['Const.bias'])
    #     self.assertIsInstance(kernel.kernel_1, ConstantKernel)
    #     self.assertIsNone(kernel.kernel_2)
    #     np.testing.assert_equal(kernel.scale, 2.)
    #
    #     x = np.array([[1., 2., 3., 4., 5.]])
    #     cov = kernel(x)
    #
    #     np.testing.assert_allclose(cov, np.array([[2., 2., 2., 2., 2.]]))

    def test_kernel_product(self) -> None:
        """

        :return:
        """
        from hilo_mpc import ConstantKernel

        kernel = ConstantKernel() * ConstantKernel(bias=.5)

        self.assertIsNone(kernel.active_dims)
        self.assertEqual(len(kernel.hyperparameters), 2)
        self.assertEqual(kernel.hyperparameter_names, ['Const_1.bias', 'Const_2.bias'])
        self.assertIsInstance(kernel.kernel_1, ConstantKernel)
        self.assertIsInstance(kernel.kernel_2, ConstantKernel)

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        np.testing.assert_allclose(cov, .25 * np.ones((5, 5)))

    def test_mean_power(self) -> None:
        """

        :return:
        """
        from hilo_mpc import ConstantKernel

        kernel = ConstantKernel(bias=.5) ** 2

        self.assertIsNone(kernel.active_dims)
        self.assertEqual(len(kernel.hyperparameters), 1)
        self.assertEqual(kernel.hyperparameter_names, ['Const.bias'])
        self.assertIsInstance(kernel.kernel_1, ConstantKernel)
        self.assertIsNone(kernel.kernel_2)
        np.testing.assert_equal(kernel.power, 2)

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        np.testing.assert_allclose(cov, .0625 * np.ones((5, 5)))

    # def test_mean_multi_op_sum_power(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     from hilo_mpc import LinearKernel, ConstantKernel, PolynomialKernel
    #
    #     kernel = (LinearKernel() + ConstantKernel()) ** 2
    #
    #     self.assertIsNone(kernel.active_dims)
    #     self.assertEqual(len(kernel.hyperparameters), 2)
    #     self.assertEqual(kernel.hyperparameter_names, ['Lin.signal_variance', 'Const.bias'])
    #     self.assertIsNone(kernel.kernel_1.active_dims)
    #     self.assertEqual(len(kernel.kernel_1.hyperparameters), 2)
    #     self.assertEqual(kernel.kernel_1.hyperparameter_names, ['Lin.signal_variance', 'Const.bias'])
    #     self.assertIsInstance(kernel.kernel_1.kernel_1, LinearKernel)
    #     self.assertIsInstance(kernel.kernel_1.kernel_2, ConstantKernel)
    #     self.assertIsNone(kernel.kernel_2)
    #     np.testing.assert_equal(kernel.power, 2)
    #
    #     poly_kernel = PolynomialKernel(2)
    #
    #     x = np.array([[1., 2., 3., 4., 5.]])
    #     # FIXME: Power doesn't use x_bar
    #     cov = kernel(x)
    #     poly_cov = poly_kernel(x)
    #
    #     # NOTE: The following only holds true, if signal variance, offset and bias all are equal to 1
    #     # TODO: Uncomment once above bug is fixed
    #     np.testing.assert_allclose(cov, poly_cov)
    #     np.testing.assert_allclose(cov, np.array([[4., 9., 16., 25., 36.],
    #                                               [9., 25., 49., 81., 121.],
    #                                               [16., 49., 100., 169., 256.],
    #                                               [25., 81., 169., 289., 441.],
    #                                               [36., 121., 256., 441., 676.]]))

    def test_kernel_multi_op_sum_1(self) -> None:
        """

        :return:
        """
        from hilo_mpc import LinearKernel

        kernel = LinearKernel() + LinearKernel(signal_variance=.5) + LinearKernel(signal_variance=2.)

        self.assertIsNone(kernel.active_dims)
        self.assertEqual(len(kernel.hyperparameters), 3)
        self.assertEqual(kernel.hyperparameter_names,
                         ['Lin_1.signal_variance', 'Lin_2.signal_variance', 'Lin_3.signal_variance'])
        self.assertIsNone(kernel.kernel_1.active_dims)
        self.assertEqual(len(kernel.kernel_1.hyperparameters), 2)
        self.assertEqual(kernel.kernel_1.hyperparameter_names, ['Lin_1.signal_variance', 'Lin_2.signal_variance'])
        self.assertIsInstance(kernel.kernel_1.kernel_1, LinearKernel)
        self.assertIsInstance(kernel.kernel_1.kernel_2, LinearKernel)
        self.assertIsInstance(kernel.kernel_2, LinearKernel)

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        np.testing.assert_allclose(cov, np.array([[3.5, 7., 10.5, 14., 17.5],
                                                  [7., 14., 21., 28., 35.],
                                                  [10.5, 21., 31.5, 42., 52.5],
                                                  [14., 28., 42., 56., 70.],
                                                  [17.5, 35., 52.5, 70., 87.5]]))

    def test_kernel_multi_op_product(self) -> None:
        """

        :return:
        """
        from hilo_mpc import LinearKernel

        kernel = LinearKernel() * LinearKernel(signal_variance=.1) * LinearKernel(signal_variance=.2)

        self.assertIsNone(kernel.active_dims)
        self.assertEqual(len(kernel.hyperparameters), 3)
        self.assertEqual(kernel.hyperparameter_names,
                         ['Lin_1.signal_variance', 'Lin_2.signal_variance', 'Lin_3.signal_variance'])
        self.assertIsNone(kernel.kernel_1.active_dims)
        self.assertEqual(len(kernel.kernel_1.hyperparameters), 2)
        self.assertEqual(kernel.kernel_1.hyperparameter_names, ['Lin_1.signal_variance', 'Lin_2.signal_variance'])
        self.assertIsInstance(kernel.kernel_1.kernel_1, LinearKernel)
        self.assertIsInstance(kernel.kernel_1.kernel_2, LinearKernel)
        self.assertIsInstance(kernel.kernel_2, LinearKernel)

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        np.testing.assert_allclose(cov, np.array([[.02, .16, .54, 1.28, 2.5],
                                                  [.16, 1.28, 4.32, 10.24, 20.],
                                                  [.54, 4.32, 14.58, 34.56, 67.5],
                                                  [1.28, 10.24, 34.56, 81.92, 160.],
                                                  [2.5, 20., 67.5, 160., 312.5]]))

    def test_kernel_multi_op_sum_2(self) -> None:
        """

        :return:
        """
        # TODO: Add __sub__ dunder method which uses Scale and Sum classes?
        from hilo_mpc import LinearKernel

        kernel = LinearKernel() + (LinearKernel(signal_variance=.5) + LinearKernel(signal_variance=2.))

        self.assertIsNone(kernel.active_dims)
        self.assertEqual(len(kernel.hyperparameters), 3)
        self.assertEqual(kernel.hyperparameter_names,
                         ['Lin_1.signal_variance', 'Lin_2.signal_variance', 'Lin_3.signal_variance'])
        self.assertIsInstance(kernel.kernel_1, LinearKernel)
        self.assertIsNone(kernel.kernel_2.active_dims)
        self.assertEqual(len(kernel.kernel_2.hyperparameters), 2)
        self.assertEqual(kernel.kernel_2.hyperparameter_names, ['Lin_2.signal_variance', 'Lin_3.signal_variance'])
        self.assertIsInstance(kernel.kernel_2.kernel_1, LinearKernel)
        self.assertIsInstance(kernel.kernel_2.kernel_2, LinearKernel)

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        np.testing.assert_allclose(cov, np.array([[3.5, 7., 10.5, 14., 17.5],
                                                  [7., 14., 21., 28., 35.],
                                                  [10.5, 21., 31.5, 42., 52.5],
                                                  [14., 28., 42., 56., 70.],
                                                  [17.5, 35., 52.5, 70., 87.5]]))

    def test_kernel_multi_op_sum_of_products(self) -> None:
        """

        :return:
        """
        from hilo_mpc import LinearKernel

        kernel = (LinearKernel() * LinearKernel(signal_variance=.7)) + \
                 (LinearKernel() * LinearKernel(signal_variance=.3))

        self.assertIsNone(kernel.active_dims)
        self.assertEqual(len(kernel.hyperparameters), 4)
        self.assertEqual(
            kernel.hyperparameter_names,
            ['Lin_1.signal_variance', 'Lin_2.signal_variance', 'Lin_3.signal_variance', 'Lin_4.signal_variance']
        )
        self.assertIsNone(kernel.kernel_1.active_dims)
        self.assertEqual(len(kernel.kernel_1.hyperparameters), 2)
        self.assertEqual(kernel.kernel_1.hyperparameter_names, ['Lin_1.signal_variance', 'Lin_2.signal_variance'])
        self.assertIsInstance(kernel.kernel_1.kernel_1, LinearKernel)
        self.assertIsInstance(kernel.kernel_1.kernel_2, LinearKernel)
        self.assertIsNone(kernel.kernel_2.active_dims)
        self.assertEqual(len(kernel.kernel_2.hyperparameters), 2)
        self.assertEqual(kernel.kernel_2.hyperparameter_names, ['Lin_3.signal_variance', 'Lin_4.signal_variance'])
        self.assertIsInstance(kernel.kernel_2.kernel_1, LinearKernel)
        self.assertIsInstance(kernel.kernel_2.kernel_2, LinearKernel)

        x = np.array([[1., 2., 3., 4., 5.]])
        cov = kernel(x)

        np.testing.assert_allclose(cov, np.array([[1., 4., 9., 16., 25.],
                                                  [4., 16., 36., 64., 100.],
                                                  [9., 36., 81., 144., 225.],
                                                  [16., 64., 144., 256., 400.],
                                                  [25., 100., 225., 400., 625.]]))
