import io
from unittest import TestCase, mock

import casadi as ca
import numpy as np

from hilo_mpc import Model


class TestLinearizationGeneral(TestCase):
    """"""
    @mock.patch('sys.stdout', new_callable=io.StringIO)
    def assertPrints(self, expected_string, mock_stdout) -> None:
        """
        See https://stackoverflow.com/a/46307456

        :param expected_string:
        :param mock_stdout:
        :return:
        """
        if hasattr(self, 'model'):
            self.model.linearize()
            self.assertEqual(mock_stdout.getvalue().rstrip(), expected_string)  # .rstrip() because print appends '\n'

    def setUp(self) -> None:
        """

        :return:
        """
        # Initialize empty model
        model = Model(plot_backend='bokeh')

        self.model = model

    def test_empty_model(self) -> None:
        """

        :return:
        """
        self.assertPrints("Model is empty. Nothing to be done.")

    def test_already_linear_1(self) -> None:
        """

        :return:
        """
        # Model equations
        equations = """
        dx_1(t)/dt = 2*x_1(t) + 2*x_2(t)
        dx_2(t)/dt = 2*x_1(t) - 2*x_2(t)
        """
        self.model.set_equations(equations=equations)

        self.assertPrints("Model is already linear. Linearization is not necessary. Nothing to be done.")

    def test_already_linear_2(self) -> None:
        """

        :return:
        """
        self.model.A = np.array([[2., 2.], [2., -2.]])

        self.assertPrints("Model is already linear. Linearization is not necessary. Nothing to be done.")

    def test_already_linearized(self) -> None:
        """

        :return:
        """
        # Model equations
        equations = """
        dx_1(t)/dt = x_1(t)^2 + x_2(t)^2 - 2
        dx_2(t)/dt = x_1(t)^2 - x_2(t)^2
        """
        self.model.set_equations(equations=equations)

        self.model = self.model.linearize()
        self.assertPrints("Model is already linearized. Nothing to be done.")


class TestLinearizationEquilibriumPointODE(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        # Initialize empty model
        model = Model(plot_backend='bokeh')

        # Model equations
        equations = """
        dx_1(t)/dt = x_2(t)
        dx_2(t)/dt = -2*x_1(t) - sin(x_1(t)) + u(k)
        y(k) = x_1
        """
        model.set_equations(equations=equations)

        self.model = model
        self.model.setup(dt=.01)
        self.case = None

    def tearDown(self) -> None:
        """

        :return:
        """
        if self.case is not None:
            if self.case == 'equilibrium_point':
                self.model.set_initial_conditions([np.pi / 2., 0.])
                self.lin_model.set_initial_conditions([0., 0.])

                pulse = ca.repmat(ca.horzcat(.1 * ca.DM.ones(1, 100), ca.DM.zeros(1, 100)), 1, 5)
                u = 1. + ca.pi + pulse

                self.model.simulate(u=u)
                self.lin_model.simulate(u=pulse)

                y = self.model.solution['y'].full()
                dy = self.lin_model.solution.to_dict('y')['y']
                y_lin = dy + np.pi / 2.

                np.testing.assert_allclose(y_lin, y, atol=1e-2)
                np.testing.assert_raises(AssertionError, np.testing.assert_allclose, y_lin, y, atol=1e-3)
            elif self.case == 'discretize_then_linearize':
                self.model.set_initial_conditions([np.pi / 2., 0.])
                self.disc_model.set_initial_conditions([np.pi / 2., 0.])
                self.lin_model.set_initial_conditions([0., 0.])

                pulse = ca.repmat(ca.horzcat(.1 * ca.DM.ones(1, 100), ca.DM.zeros(1, 100)), 1, 5)
                u = 1. + ca.pi + pulse

                self.model.simulate(u=u)
                self.disc_model.simulate(u=u)
                self.lin_model.simulate(u=pulse)

                y = self.model.solution['y'].full()
                y_disc = self.disc_model.solution['y'].full()
                dy = self.lin_model.solution.to_dict('y')['y']
                y_lin = dy + np.pi / 2.

                np.testing.assert_allclose(y_lin, y_disc, atol=1e-2)
                np.testing.assert_raises(AssertionError, np.testing.assert_allclose, y_lin, y_disc, atol=1e-3)
                np.testing.assert_allclose(y_lin, y, atol=1e-2)
                np.testing.assert_raises(AssertionError, np.testing.assert_allclose, y_lin, y, atol=1e-3)
                np.testing.assert_allclose(y_disc, y, atol=1e-2)
                np.testing.assert_raises(AssertionError, np.testing.assert_allclose, y_disc, y, atol=1e-3)
            elif self.case == 'linearize_then_discretize':
                self.model.set_initial_conditions([np.pi / 2., 0.])
                self.lin_model.set_initial_conditions([0., 0.])
                self.disc_model.set_initial_conditions([0., 0.])

                pulse = ca.repmat(ca.horzcat(.1 * ca.DM.ones(1, 100), ca.DM.zeros(1, 100)), 1, 5)
                u = 1. + ca.pi + pulse

                self.model.simulate(u=u)
                self.lin_model.simulate(u=pulse)
                self.disc_model.simulate(u=pulse)

                y = self.model.solution['y'].full()
                dy = self.lin_model.solution.to_dict('y')['y']
                y_lin = dy + np.pi / 2.
                dy = self.disc_model.solution.to_dict('y')['y']
                y_disc = dy + np.pi / 2.

                np.testing.assert_allclose(y_disc, y_lin, atol=1e-2)
                np.testing.assert_raises(AssertionError, np.testing.assert_allclose, y_disc, y_lin, atol=1e-3)
                np.testing.assert_allclose(y_disc, y, atol=1e-2)
                np.testing.assert_raises(AssertionError, np.testing.assert_allclose, y_disc, y, atol=1e-3)
            elif self.case == 'equilibrium_trajectory':
                self.model.set_initial_conditions([0., 1.])
                self.lin_model.set_initial_conditions([0., 0.])

                pulse = ca.repmat(ca.horzcat(.05 * ca.DM.ones(1, 100), ca.DM.zeros(1, 100)), 1, 5)
                t_span = ca.linspace(0., 10., 1001).T
                u = 2. - 3. * ca.exp(-t_span[:, :-1]) + ca.sin(1. - ca.exp(-t_span[:, :-1])) + pulse

                self.model.simulate(u=u)
                self.lin_model.simulate(u=pulse)

                # ext_y = self.lin_model.solution.to_dict('y', subplots=True)['y']
                # ext_y['label'] = 'y_lin'
                # ext_y['data'] += 1. - np.exp(-t_span.full())
                # self.model.solution.plot(y_data=[ext_y])

                y = self.model.solution['y'].full()
                dy = self.lin_model.solution.to_dict('y')['y']
                y_lin = dy + 1. - np.exp(-t_span.full())

                np.testing.assert_allclose(y_lin, y, atol=1e-2)
                np.testing.assert_raises(AssertionError, np.testing.assert_allclose, y_lin, y, atol=1e-3)

    def test_equilibrium_point_missing_x(self) -> None:
        """

        :return:
        """
        self.lin_model = self.model.linearize()
        self.lin_model.setup(dt=.01)
        with self.assertRaises(ValueError) as context:
            self.lin_model.set_equilibrium_point(u_eq=[0.])
        self.assertEqual("Dynamical state information is missing from the equilibrium point.", str(context.exception))

    def test_equilibrium_point_missing_u(self) -> None:
        """

        :return:
        """
        self.lin_model = self.model.linearize()
        self.lin_model.setup(dt=.01)
        with self.assertRaises(ValueError) as context:
            self.lin_model.set_equilibrium_point(x_eq=[0., 0.])
        self.assertEqual("Input information is missing from the equilibrium point.", str(context.exception))

    def test_wrong_dimensions_in_x(self) -> None:
        """

        :return:
        """
        self.lin_model = self.model.linearize()
        self.lin_model.setup(dt=.01)
        with self.assertRaises(ValueError) as context:
            self.lin_model.set_equilibrium_point(x_eq=[0., 0., 0.], u_eq=[0.])
        self.assertEqual("Dimension mismatch for the dynamical state information of the equilibrium point. Got 3, "
                         "expected 2.", str(context.exception))

    def test_wrong_dimensions_in_u(self) -> None:
        """

        :return:
        """
        self.lin_model = self.model.linearize()
        self.lin_model.setup(dt=.01)
        with self.assertRaises(ValueError) as context:
            self.lin_model.set_equilibrium_point(x_eq=[0., 0.], u_eq=[0., 0.])
        self.assertEqual("Dimension mismatch for the input information of the equilibrium point. Got 2, expected 1.",
                         str(context.exception))

    def test_wrong_equilibrium_point_given(self) -> None:
        """

        :return:
        """
        self.lin_model = self.model.linearize()
        self.lin_model.setup(dt=.01)
        with self.assertRaises(ValueError) as context:
            self.lin_model.set_equilibrium_point(x_eq=[np.pi / 2., 0.], u_eq=[0.])
        self.assertEqual("Supplied values are not an equilibrium point. Maximum error: 4.14159", str(context.exception))

    def test_no_equilibrium_point_given(self) -> None:
        """

        :return:
        """
        self.lin_model = self.model.linearize()
        self.lin_model.setup(dt=.01)
        self.lin_model.set_initial_conditions([0., 0.])
        with self.assertRaises(RuntimeError) as context:
            self.lin_model.simulate(u=.1)
        self.assertEqual(
            "Model is linearized, but no equilibrium point was set. Please set equilibrium point before simulating the "
            "model!",
            str(context.exception)
        )

    def test_equilibrium_point_given(self) -> None:
        """

        :return:
        """
        self.lin_model = self.model.linearize()
        self.lin_model.setup(dt=.01)
        self.lin_model.set_equilibrium_point(x_eq=[np.pi / 2., 0.], u_eq=[1. + np.pi])
        self.case = 'equilibrium_point'

    def test_discretize_then_linearize(self) -> None:
        """

        :return:
        """
        self.disc_model = self.model.discretize('erk')
        self.disc_model.setup(dt=.01)
        self.lin_model = self.disc_model.linearize()
        self.lin_model.setup(dt=.01)
        self.lin_model.set_equilibrium_point(x_eq=[np.pi / 2., 0.], u_eq=[1. + np.pi])
        self.case = 'discretize_then_linearize'

    def test_linearize_then_discretize(self) -> None:
        """

        :return:
        """
        self.lin_model = self.model.linearize()
        self.disc_model = self.lin_model.discretize('erk')
        self.lin_model.setup(dt=.01)
        self.disc_model.setup(dt=.01)
        self.lin_model.set_equilibrium_point(x_eq=[np.pi / 2., 0.], u_eq=[1. + np.pi])
        self.disc_model.set_equilibrium_point(x_eq=[np.pi / 2., 0.], u_eq=[1. + np.pi])
        self.case = 'linearize_then_discretize'

    def test_equilibrium_trajectory_given(self) -> None:
        """

        :return:
        """
        trajectory = {
            'x': ['1 - exp(-t)', 'exp(-t)'],
            'u': '2 - 3*exp(-t) + sin(1 - exp(-t))'
        }
        self.assertFalse(self.model.is_time_variant())
        self.lin_model = self.model.linearize(trajectory=trajectory)
        self.lin_model.setup(dt=.01)
        self.assertTrue(self.lin_model.is_time_variant())
        self.case = 'equilibrium_trajectory'


class TestLinearizationEquilibriumPointDAE(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        # NOTE: Maybe we can find a better model here. The 2 obvious equilibrium points of the pendulum are standing
        #  upright and hanging down. Both equilibrium points turn out to be unstable (at least one positive eigenvalue
        #  for the corresponding linearized state matrices), but for the equilibrium point where the pendulum just
        #  hangs down, the equilibrium point itself cancels out all the algebraic equations (so there is no change in
        #  the linearized algebraic states) and the model effectively becomes an ODE system which is marginally stable
        #  (hence the oscillations).
        # Initialize empty model
        model = Model(plot_backend='bokeh', solver='idas')

        # Dynamical states
        model.set_dynamical_states('x', 'u')

        # Algebraic states
        model.set_algebraic_states('y', 'v', 'l')

        # Set equations
        model.set_dynamical_equations(['u', 'l * x'])
        model.set_algebraic_equations(['x^2 + y^2 - L^2', 'u * x + v * y', 'u^2 - 9.81 * y + v^2 + L^2 * l'])

        self.model = model
        self.model.setup(dt=.01)
        self.case = None

    def tearDown(self) -> None:
        """

        :return:
        """
        if self.case is not None:
            if self.case == 'equilibrium_point':
                x0 = .05
                u0 = 0.
                y0 = -np.sqrt(.2 ** 2 - x0 ** 2)
                v0 = 0.
                l0 = 9.81 * y0 / .2 ** 2

                self.model.set_initial_conditions([x0, u0], z0=[y0, v0, l0])
                self.model.set_initial_parameter_values(.2)

                self.lin_model.set_initial_conditions([x0, u0], z0=[y0 + .2, v0, l0 + 9.81 / .2])

                self.model.simulate(steps=1000)
                self.lin_model.simulate(steps=1000)

                x = self.model.solution.get_by_id('x').full()
                dx = self.lin_model.solution.to_dict('x')
                x_lin = np.append(dx['dx'], dx['du'], axis=0)

                np.testing.assert_allclose(x_lin, x, atol=1e-1)
                np.testing.assert_raises(AssertionError, np.testing.assert_allclose, x_lin, x, atol=1e-2)

    def test_equilibrium_point_missing_z(self) -> None:
        """

        :return:
        """
        self.lin_model = self.model.linearize()
        self.lin_model.setup(dt=.01)
        with self.assertRaises(ValueError) as context:
            self.lin_model.set_equilibrium_point(x_eq=[0., 0.])
        self.assertEqual("Algebraic state information is missing from the equilibrium point.", str(context.exception))

    def test_equilibrium_point_missing_p(self) -> None:
        """

        :return:
        """
        self.lin_model = self.model.linearize()
        self.lin_model.setup(dt=.01)
        with self.assertRaises(RuntimeError) as context:
            self.lin_model.set_equilibrium_point(x_eq=[0., 0.], z_eq=[0., 0., 0.])
        self.assertEqual(
            "Please set the values for the parameters by executing the 'set_initial_parameter_values' method before "
            "setting the equilibrium point.",
            str(context.exception)
        )

    def test_wrong_dimensions_in_z(self) -> None:
        """

        :return:
        """
        self.lin_model = self.model.linearize()
        self.lin_model.setup(dt=.01)
        with self.assertRaises(ValueError) as context:
            self.lin_model.set_equilibrium_point(x_eq=[0., 0.], z_eq=[0.])
        self.assertEqual(
            "Dimension mismatch for the algebraic state information of the equilibrium point. Got 1, expected 3.",
            str(context.exception)
        )

    def test_wrong_equilibrium_point_given(self) -> None:
        """

        :return:
        """
        self.lin_model = self.model.linearize()
        self.lin_model.setup(dt=.01)
        self.lin_model.set_initial_parameter_values(.2)
        with self.assertRaises(ValueError) as context:
            self.lin_model.set_equilibrium_point(x_eq=[0., 0.], z_eq=[0., 0., 0])
        self.assertEqual("Supplied values are not an equilibrium point. Maximum error: 0.04000", str(context.exception))

    def test_equilibrium_point_given(self) -> None:
        """

        :return:
        """
        self.lin_model = self.model.linearize()
        self.lin_model.setup(dt=.01)
        self.lin_model.set_initial_parameter_values(.2)
        self.lin_model.set_equilibrium_point(x_eq=[0., 0.], z_eq=[-.2, 0., -9.81 / .2])
        self.case = 'equilibrium_point'
