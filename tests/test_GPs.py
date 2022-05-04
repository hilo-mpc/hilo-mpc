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
class TestOneFeatureOneLabel(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        self.x = randn(.8, (20, 1))
        self.y = np.sin(3 * self.x) + .1 * randn(.9, (20, 1))
        self.x_s = np.linspace(-3, 3, 61).reshape(1, -1)
        self.kernel = None
        self.solver = None

    def tearDown(self) -> None:
        """

        :return:
        """
        # NOTE: Parameter values for comparison are taken from the gpml toolbox for MATLAB using its default solver
        #  unless otherwise noted
        if self.kernel is None:
            np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
                                       np.array([.0085251, .5298217, .8114553]), rtol=1e-5)
        else:
            if self.kernel.acronym == 'Const':
                np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
                                           np.array([.7009480, 3.0498634]), rtol=1e-5)
            elif self.kernel.acronym == 'E':
                # NOTE: Hyperparameters should always be positive
                self.assertTrue(self.gp.noise_variance.value < 1e-5)  # noise variance is almost 0
                np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters][1:],
                                           np.array([.9769135, .5935212]), rtol=1e-5)
            elif self.kernel.acronym == 'M32':
                np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
                                           np.array([.0088262, .8329355, .9398366]), rtol=1e-5)
            elif self.kernel.acronym == 'M52':
                np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
                                           np.array([.0086694, .7180206, .9571137]), rtol=1e-5)
            elif self.kernel.acronym == 'Matern':
                np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
                                           np.array([.0086179, .6666981, .9421087]), rtol=1e-5)
            elif self.kernel.acronym == 'RQ':
                # NOTE: Here, we get a slightly better optimum than with the default solver of the gpml toolbox, but
                #  the parameter \alpha is quite high. Don't know if a value that high makes much sense.
                # NOTE: The last value is always fluctuating very much on different computers since the sensitivity of
                #  that value w.r.t. to objective value seems to be very low, so we are ignoring it for now.
                np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters][:-1],
                                           np.array([.00852509, .529822, .811455]), rtol=1e-5)
            elif self.kernel.acronym == 'PP':
                if self.degree == 0:
                    # NOTE: Here, we get a better optimum than with the default solver of the gpml toolbox for MATLAB
                    #  (-1.29015 vs -1.78466). Also, if we use the optimal parameter values in the gpml toolbox, no
                    #  optimization is taking place. But the value for the noise variance is super low. I don't know if
                    #  such a low value makes much sense. Maybe we can find a better example or there is some way to
                    #  make sure that the hyperparameters don't get too low.
                    np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
                                               np.array([3.71835e-29, .994594, .427023]), rtol=1e-5)
                elif self.degree == 1:
                    np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
                                               np.array([.0088391, 1.6079331, .6545336]), rtol=1e-5)
                elif self.degree == 2:
                    np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
                                               np.array([.0088244, 2.0883167, .852502]), rtol=1e-5)
                elif self.degree == 3:
                    np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
                                               np.array([.0086766, 2.247363, .7873581]), rtol=1e-5)
            elif self.kernel.acronym == 'Poly':
                np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
                                           np.array([.0980796, 1.3112287, .5083423]), rtol=1e-5)
            elif self.kernel.acronym == 'Lin':
                np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
                                           np.array([.6627861, .008198]), rtol=1e-4)
            elif self.kernel.acronym == 'NN':
                np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
                                           np.array([.0095177, 5.7756069, .1554265]), rtol=1e-5)
            elif self.kernel.acronym == 'Periodic':
                np.testing.assert_allclose([parameter.value for parameter in self.gp.hyperparameters],
                                           np.array([.4975112, .159969, .5905631, .8941061]), rtol=1e-3)

    def test_gp_regression(self) -> None:
        """

        :return:
        """
        gp = GP(['x'], ['y'], kernel=self.kernel, noise_variance=np.exp(-2), solver=self.solver)
        gp.set_training_data(self.x.T, self.y.T)
        gp.setup()
        with self.assertWarns(UserWarning) as context:
            gp.fit_model()
            warnings.warn("Dummy warning!!!")
        self.assertEqual(len(context.warnings), 1)  # this will catch unsuccessful optimizations (among other warnings)

        self.gp = gp


class TestOneFeatureOneLabelConst(TestOneFeatureOneLabel):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        super().setUp()
        self.y += 3.
        self.kernel = Kernel.constant()


@skip("This test is not necessary, since the GP in 'TestOneFeatureOneLabel' already defaults to the squared exponential"
      " kernel")
class TestOneFeatureOneLabelSE(TestOneFeatureOneLabel):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        super().setUp()
        self.kernel = Kernel.squared_exponential()


class TestOneFeatureOneLabelE(TestOneFeatureOneLabel):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        super().setUp()
        self.kernel = Kernel.exponential()


class TestOneFeatureOneLabelM32(TestOneFeatureOneLabel):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        super().setUp()
        self.kernel = Kernel.matern_32()


class TestOneFeatureOneLabelM52(TestOneFeatureOneLabel):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        super().setUp()
        self.kernel = Kernel.matern_52()


class TestOneFeatureOneLabelM72(TestOneFeatureOneLabel):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        from hilo_mpc import MaternKernel

        super().setUp()
        self.kernel = MaternKernel(3)


class TestOneFeatureOneLabelRQ(TestOneFeatureOneLabel):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        super().setUp()
        self.kernel = Kernel.rational_quadratic()


class TestOneFeatureOneLabelPP0(TestOneFeatureOneLabel):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        # TODO: Find out what's happening here. We get a super small noise variance.
        super().setUp()
        self.kernel = Kernel.piecewise_polynomial(0)
        self.solver = 'Powell'
        self.degree = 0


class TestOneFeatureOneLabelPP1(TestOneFeatureOneLabel):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        super().setUp()
        self.kernel = Kernel.piecewise_polynomial(1)
        self.solver = 'BFGS'
        self.degree = 1


class TestOneFeatureOneLabelPP2(TestOneFeatureOneLabel):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        super().setUp()
        self.kernel = Kernel.piecewise_polynomial(2)
        self.degree = 2


class TestOneFeatureOneLabelPP3(TestOneFeatureOneLabel):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        super().setUp()
        self.kernel = Kernel.piecewise_polynomial(3)
        self.degree = 3


class TestOneFeatureOneLabelPoly(TestOneFeatureOneLabel):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        super().setUp()
        self.kernel = Kernel.polynomial(3)


class TestOneFeatureOneLabelLin(TestOneFeatureOneLabel):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        super().setUp()
        self.kernel = Kernel.linear()


class TestOneFeatureOneLabelNN(TestOneFeatureOneLabel):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        super().setUp()
        self.kernel = Kernel.neural_network()
        self.solver = 'CG'


class TestOneFeatureOneLabelPeriodic(TestOneFeatureOneLabel):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        super().setUp()
        self.kernel = Kernel.periodic()


class RasmussenSimpleRegression(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        self.kernel = Kernel.matern_32(length_scales=.25)
        self.mean = Mean.linear(coefficient=.5, bounds={'coefficient': 'fixed'}) + Mean.one()

        self.x = randn(0.3, (20, 1))
        K = self.kernel(self.x.T)
        mu = self.mean(self.x.T)
        self.y = np.linalg.cholesky(K) @ randn(.15, (20, 1)) + mu.T + .1 * randn(.2, (20, 1))
        self.x_s = np.linspace(-1.9, 1.9, 101).reshape(1, -1)

    def tearDown(self) -> None:
        """

        :return:
        """
        np.testing.assert_approx_equal(self.lml, -11.9706317)
        np.testing.assert_allclose([parameter.value for parameter in self.gp_1.hyperparameters],
                                   np.array([.0222571, .3703166, 3.9427837]), rtol=1e-6)
        np.testing.assert_allclose([parameter.value for parameter in self.gp_2.hyperparameters],
                                   np.array([.02183, 1.1918832, 1.4624534, .2201449, .4017997]), rtol=1e-6)
        # NOTE: Here, we get a better optimum than with the default solver of the gpml toolbox for MATLAB
        #  (-2.26587 vs -3.36002), but only with the solvers 'Nelder-Mead' and 'Powell', which don't require gradients.
        #  Also, if we use the optimal parameter values in the gpml toolbox, no optimization is taking place.
        #  Maybe we can find a better example.
        np.testing.assert_allclose([parameter.value for parameter in self.gp_3.hyperparameters],
                                   np.array([.0100000, 1.00023, 1., .260644, .719711]), rtol=1e-5)

    def test_gp_regression(self):
        """

        :return:
        """
        gp = GP(['x'], ['y'], mean=self.mean, kernel=self.kernel, noise_variance=.1 ** 2)
        gp.set_training_data(self.x.T, self.y.T)
        gp.setup()
        self.lml = gp.log_marginal_likelihood()

        gp = GP(['x'], ['y'], noise_variance=.1 ** 2)
        gp.set_training_data(self.x.T, self.y.T)
        gp.setup()
        gp.fit_model()
        self.gp_1 = gp

        mean = Mean.linear(coefficient=0.) + Mean.constant(bias=0.)
        gp = GP(['x'], ['y'], mean=mean, noise_variance=.1 ** 2)
        gp.set_training_data(self.x.T, self.y.T)
        gp.setup()
        gp.fit_model()
        self.gp_2 = gp

        mean_1 = Mean.linear(coefficient=0., hyperprior='Gaussian')
        mean_1.coefficient.prior.mean = 1.
        mean_1.coefficient.prior.variance = .01 ** 2
        mean_2 = Mean.constant(bias=0., hyperprior='Laplace')
        mean_2.bias.prior.mean = 1.
        mean_2.bias.prior.variance = .01 ** 2
        mean = mean_1 + mean_2
        gp = GP(['x'], ['y'], mean=mean, noise_variance=.1 ** 2, solver='Powell')
        gp.noise_variance.prior = 'Delta'
        gp.set_training_data(self.x.T, self.y.T)
        gp.setup()
        gp.fit_model()
        self.gp_3 = gp


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
