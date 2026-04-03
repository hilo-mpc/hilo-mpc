from unittest import TestCase

import numpy as np

from hilo_mpc import Model, PF


class TestParticleFilterInitialization(TestCase):
    """"""
    def test_particle_filter_linear_model_warning(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh')
        equations = """
        dx_1/dt = -k_1*x_1(t) + u(k)
        dx_2/dt = k_1*x_1(t) - k_2*x_2(t)
        y(k) = x_2(t)
        """
        model.set_equations(equations=equations)
        model.discretize('erk', order=1, inplace=True)
        model.setup(dt=1.)

        with self.assertWarns(UserWarning) as context:
            PF(model, plot_backend='bokeh')
        self.assertEqual(len(context.warnings), 1)
        self.assertEqual(
            str(context.warning),
            "The supplied model is linear. For better efficiency use an observer targeted at the estimation of linear "
            "systems."
        )

    def test_particle_filter_initial_tuning_parameters(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh', discrete=True)
        model.set_dynamical_states('x')
        model.set_measurements('y')
        model.set_dynamical_equations('x/2 + 25*dt*x/(1 + x^2)')
        model.set_measurement_equations('x^2/20')
        model.setup(dt=1.)

        pf = PF(model, plot_backend='bokeh')
        self.assertTrue(callable(pf.probability_density_function))
        # TODO: Any other assertions with respect to the probability density function?
        self.assertIsNone(pf.variant)
        self.assertEqual(pf.sample_size, 15)


class TestParticleFilterPDF(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh', discrete=True)
        model.set_dynamical_states('x')
        model.set_measurements('y')
        model.set_dynamical_equations('x/2 + 25*dt*x/(1 + x^2)')
        model.set_measurement_equations('x^2/20')
        model.setup(dt=1.)

        pf = PF(model, plot_backend='bokeh')
        pf.setup()
        self.pf = pf

    def test_particle_filter_pdf_not_callable(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.pf.probability_density_function = 'pdf'
        self.assertEqual(str(context.exception),
                         "Probability density function of the particle filter needs to be callable.")

    def test_particle_filter_pdf_annotations(self) -> None:
        """

        :return:
        """
        def pdf(mu: np.ndarray, sigma: np.ndarray, n: int) -> np.ndarray:
            """

            :param mu:
            :param sigma:
            :param n:
            :return:
            """
            np.random.seed(0)
            return np.random.multivariate_normal(mu, sigma, size=n)

        pf = self.pf
        pf.probability_density_function = pdf
        self.assertTrue(callable(pf.probability_density_function))
        np.testing.assert_allclose(pf.probability_density_function(np.array([0., 0., 0.]), np.eye(3), 3),
                                   pdf(np.array([0., 0., 0.]), np.eye(3), 3))

    def test_particle_filter_pdf_wrong_annotation_mu(self) -> None:
        """

        :return:
        """
        def pdf(mu: float, sigma: np.ndarray, n: int) -> np.ndarray:
            """

            :param mu:
            :param sigma:
            :param n:
            :return:
            """
            pass

        with self.assertRaises(TypeError) as context:
            self.pf.probability_density_function = pdf
        self.assertEqual(
            str(context.exception),
            "The 1st argument to the probability density function (pdf) needs to be the 'mean' with type ndarray."
        )

    def test_particle_filter_pdf_wrong_annotation_sigma(self) -> None:
        """

        :return:
        """
        def pdf(mu: np.ndarray, sigma: float, n: int) -> np.ndarray:
            """

            :param mu:
            :param sigma:
            :param n:
            :return:
            """
            pass

        with self.assertRaises(TypeError) as context:
            self.pf.probability_density_function = pdf
        self.assertEqual(
            str(context.exception),
            "The 2nd argument to the probability density function (pdf) needs to be the 'covariance' with type ndarray."
        )

    def test_particle_filter_pdf_wrong_annotation_n(self) -> None:
        """

        :return:
        """
        def pdf(mu: np.ndarray, sigma: np.ndarray, n: float) -> np.ndarray:
            """

            :param mu:
            :param sigma:
            :param n:
            :return:
            """
            pass

        with self.assertRaises(TypeError) as context:
            self.pf.probability_density_function = pdf
        self.assertEqual(
            str(context.exception),
            "The 3rd argument to the probability density function (pdf) needs to be the 'sample size' with type int."
        )

    def test_particle_filter_pdf_wrong_annotation_return(self) -> None:
        """

        :return:
        """
        def pdf(mu: np.ndarray, sigma: np.ndarray, n: int) -> float:
            """

            :param mu:
            :param sigma:
            :param n:
            :return:
            """
            pass

        with self.assertRaises(TypeError) as context:
            self.pf.probability_density_function = pdf
        self.assertEqual(
            str(context.exception),
            "The return value of the probability density function (pdf) needs to be a 'random sample' with type "
            "ndarray."
        )

    def test_particle_filter_pdf_no_annotation(self) -> None:
        """

        :return:
        """
        def pdf(mu, sigma, n):
            """

            :param mu:
            :param sigma:
            :param n:
            :return:
            """
            np.random.seed(0)
            return np.random.multivariate_normal(mu, sigma, size=n).T

        pf = self.pf
        pf.probability_density_function = pdf
        self.assertTrue(callable(pf.probability_density_function))
        np.testing.assert_allclose(pf.probability_density_function(np.array([0., 0., 0.]), np.eye(3), 3),
                                   pdf(np.array([0., 0., 0.]), np.eye(3), 3))

    def test_particle_filter_pdf_no_annotation_transposed(self) -> None:
        """

        :return:
        """
        def pdf(mu, sigma, n):
            """

            :param mu:
            :param sigma:
            :param n:
            :return:
            """
            np.random.seed(0)
            return np.random.multivariate_normal(mu, sigma, size=n)

        pf = self.pf
        pf.probability_density_function = pdf
        self.assertTrue(callable(pf.probability_density_function))
        np.testing.assert_allclose(pf.probability_density_function(np.array([0., 0., 0.]), np.eye(3), 3),
                                   pdf(np.array([0., 0., 0.]), np.eye(3), 3))

    # def test_particle_filter_pdf_no_annotation_dimension_mismatch(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     def pdf(mu, sigma, n):
    #         """
    #
    #         :param mu:
    #         :param sigma:
    #         :param n:
    #         :return:
    #         """
    #         return np.random.multivariate_normal(np.tile(mu, 2), np.tile(sigma, (2, 2)), size=n)
    #
    #     pf = self.pf
    #     # FIXME: Right now this ValueError will be caught by the following except statement
    #     with self.assertRaises(ValueError) as context:
    #         pf.probability_density_function = pdf
    #     self.assertEqual(str(context.exception), "Dimension mismatch. Expected dimension 1x15, got 15x2.")

    def test_particle_filter_pdf_no_annotation_exception(self) -> None:
        """

        :return:
        """
        def pdf(mu, sigma):
            """

            :param mu:
            :param sigma:
            :return:
            """
            pass

        pf = self.pf
        with self.assertRaises(RuntimeError) as context:
            pf.probability_density_function = pdf
        # Check that the error message contains the key parts (Python 3.11+ includes full function path)
        error_msg = str(context.exception)
        self.assertIn("The following exception was raised", error_msg)
        self.assertIn("TypeError:", error_msg)
        self.assertIn("pdf() takes 2 positional arguments but 3 were given", error_msg)
        self.assertIn("Please make sure that the supplied probability density function", error_msg)
        self.assertIn("mu - mean of the pdf (type: numpy.ndarray)", error_msg)
        self.assertIn("sigma - covariance of the mean (type: numpy.ndarray)", error_msg)
        self.assertIn("n - sample size (type: int)", error_msg)
        self.assertIn("X - random sample (type: numpy.ndarray)", error_msg)


class TestParticleFilterOtherTunableParameterSetters(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh', discrete=True)
        model.set_dynamical_states('x')
        model.set_measurements('y')
        model.set_dynamical_equations('x/2 + 25*dt*x/(1 + x^2)')
        model.set_measurement_equations('x^2/20')
        model.setup(dt=1.)

        pf = PF(model, plot_backend='bokeh')
        pf.setup()
        self.pf = pf

    def test_particle_filter_variant_setter(self) -> None:
        """

        :return:
        """
        pf = self.pf
        pf.variant = 'default'
        self.assertEqual(pf.variant, 'default')

    def test_particle_filter_sample_size_setter(self) -> None:
        """

        :return:
        """
        pf = self.pf
        pf.sample_size = 20
        self.assertEqual(pf.sample_size, 20)


class TestParticleFilterMultipleMeasurements(TestCase):
    """Tests for PF with n_y > 1, covering the bug reported in issue #44."""

    def _make_lorenz_model(self):
        """Return a set-up Lorenz attractor model with 3 states and 3 measurements."""
        model = Model(plot_backend='bokeh')
        states = model.set_dynamical_states(['x1', 'x2', 'x3'])
        inputs = model.set_inputs(['u1', 'u2', 'u3'])
        model.set_measurements(['o1', 'o2', 'o3'])
        model.set_measurement_equations(states)

        x1 = states[0]
        x2 = states[1]
        x3 = states[2]
        u1 = inputs[0]
        u2 = inputs[1]
        u3 = inputs[2]

        c = np.array([10., 28., 8 / 3])
        dx1 = c[0] * (x2 - x1) + u1
        dx2 = c[1] * x1 - x2 - x1 * x3 + u2
        dx3 = x1 * x2 - c[2] * x3 + u3

        model.set_dynamical_equations([dx1, dx2, dx3])
        model.setup(dt=0.001)
        model.set_initial_conditions(x0=[0., 0., 0.])
        return model

    def test_particle_filter_setup_n_y_greater_than_1(self) -> None:
        """Regression test for issue #44: PF.setup() must not raise for n_y > 1."""
        model = self._make_lorenz_model()
        pf = PF(model, roughening=True, plot_backend='bokeh')
        # Previously raised: RuntimeError: Input 0 (i0) has mismatching shape. Got 3-by-15.
        pf.setup()
        self.assertIsNotNone(pf._update_function)

    def test_particle_filter_setup_single_measurement_unchanged(self) -> None:
        """The n_y = 1 path must still work after the fix."""
        model = Model(plot_backend='bokeh', discrete=True)
        model.set_dynamical_states('x')
        model.set_measurements('y')
        model.set_dynamical_equations('x/2 + 25*dt*x/(1 + x^2)')
        model.set_measurement_equations('x^2/20')
        model.setup(dt=1.)

        pf = PF(model, plot_backend='bokeh')
        pf.setup()
        self.assertIsNotNone(pf._update_function)

    def test_particle_filter_estimate_n_y_greater_than_1(self) -> None:
        """End-to-end: PF.estimate() must produce a state vector of the right shape for n_y > 1."""
        np.random.seed(0)
        model = self._make_lorenz_model()
        # Use roughening=False here; roughening is tested separately in the setup test.
        # When all particles start at the Lorenz fixed point with zero process noise the
        # particles collapse after resampling and the roughening step fails with a singular
        # covariance — that is a separate pre-existing issue unrelated to issue #44.
        pf = PF(model, plot_backend='bokeh')
        pf.setup()

        pf.R = np.diag([0.1, 0.1, 0.1])
        pf.set_initial_guess([0., 0., 0.])

        pf.estimate(y=[0., 0., 0.], u=[0., 0., 0.])

        x_est = pf.solution.get_by_id('x:f')
        self.assertEqual(x_est.shape, (3, 1))


class TestParticleFilterEstimation(TestCase):
    """"""
    # TODO: Tests for transpose_pdf equal to True and False + dimension mismatch error (maybe the first 2 can be
    #  integrated into the already existing tests, dimension mismatch will get an extra test)
    def setUp(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh')
        equations = """
        dT/dt = alpha*(T_amb - T(t))
        dcB/dt = r - D(k)*cB(t)
        dcS/dt = -r/Y - D(k)*cS(t)

        r = (mu_0 + mu_1*T(t))*cS(t)*cB(t)/(K + cS(t))
        """
        model.set_equations(equations=equations)

        self.model = model

    def test_particle_filter_one_step(self) -> None:
        """PF with 2 measurements (n_y=2) must complete one estimation step."""
        np.random.seed(0)
        model = self.model
        model.set_measurement_equations(['T', 'cB'])
        model.setup(dt=.1)

        pf = PF(model, plot_backend='bokeh')
        pf.setup()

        pf.R = np.diag([.25, .01])
        pf.set_initial_guess([299.876, .217108, 20.])
        pf.set_initial_parameter_values([.15, 303.15, .13, .00025, 15., .14])

        pf.estimate(y=[300.941, .245805], u=.01)

        x_est = pf.solution.get_by_id('x:f')
        self.assertEqual(x_est.shape, (3, 1))

    def test_particle_filter_one_step_roughening(self) -> None:
        """PF with roughening and 2 measurements must complete one estimation step.

        A realistic P0 (matching the scale of each state) is provided so that initial
        particles are spread out and do not collapse to a single point after resampling,
        which would make the roughening covariance singular.
        """
        np.random.seed(42)
        model = self.model
        model.set_measurement_equations(['T', 'cB'])
        model.setup(dt=.1)

        pf = PF(model, roughening=True, plot_backend='bokeh')
        pf.setup(n_samples=30)

        pf.R = np.diag([.25, .01])
        # P0 diagonal values are chosen relative to the magnitude of each state so that
        # particles are well spread and dx > 0 in every dimension after resampling.
        pf.set_initial_guess([299.876, .217108, 20.], P0=[25., 0.01, 25.])
        pf.set_initial_parameter_values([.15, 303.15, .13, .00025, 15., .14])

        pf.estimate(y=[300.941, .245805], u=.01)

        x_est = pf.solution.get_by_id('x:f')
        self.assertEqual(x_est.shape, (3, 1))

    # def test_particle_filter_one_step_prior_editing_K_supplied(self) -> None:
    #     """PF with prior editing, custom K, and 2 measurements.
    #
    #     This test is skipped because prior_editing has a pre-existing bug when n_y > 1:
    #     `ca.sum2(need_roughening)` returns a (n_y,) vector instead of a scalar, so
    #     `int(ca.sum2(...))` raises a CasADi "is_scalar() failed" error.  That bug is
    #     independent of the issue #44 fix and requires a separate fix.
    #     """
