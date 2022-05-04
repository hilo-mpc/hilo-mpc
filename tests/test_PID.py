from unittest import TestCase

import numpy as np

from hilo_mpc import PID


class TestPIDInitialization(TestCase):
    """"""
    def test_initial_tuning_parameters_single_set_point(self) -> None:
        """

        :return:
        """
        pid = PID(plot_backend='bokeh')

        np.testing.assert_equal(pid.k_p, np.ones((1, 1)))
        np.testing.assert_equal(pid.t_i, np.inf * np.ones((1, 1)))
        np.testing.assert_equal(pid.k_i, np.zeros((1, 1)))
        np.testing.assert_equal(pid.t_d, np.zeros((1, 1)))
        np.testing.assert_equal(pid.k_d, np.zeros((1, 1)))

    def test_initial_set_point_single_set_point(self) -> None:
        """

        :return:
        """
        pid = PID(plot_backend='bokeh')

        np.testing.assert_allclose(pid.set_point, np.array([[0.]]))

    def test_initial_tuning_parameters_multiple_set_points(self) -> None:
        """

        :return:
        """
        pid = PID(n_set_points=2, plot_backend='bokeh')

        np.testing.assert_equal(pid.k_p, np.eye(2))
        np.testing.assert_equal(pid.t_i, np.diag(np.inf * np.ones(2)))
        np.testing.assert_equal(pid.k_i, np.zeros((2, 2)))
        np.testing.assert_equal(pid.t_d, np.zeros((2, 2)))
        np.testing.assert_equal(pid.k_d, np.zeros((2, 2)))

    def test_initial_set_point_multiple_set_points(self) -> None:
        """

        :return:
        """
        pid = PID(n_set_points=2, plot_backend='bokeh')

        np.testing.assert_allclose(pid.set_point, np.array([[0.], [0.]]))


class TestPIDSetup(TestCase):
    """"""
    def test_is_set_up(self) -> None:
        """

        :return:
        """
        pid = PID(plot_backend='bokeh')
        pid.setup(dt=.01)

        self.assertTrue(pid.is_setup())


class TestPIDTunableParameterSetter(TestCase):
    """"""
    # TODO: Wrong dimension test once convert function has been updated to return error messages
    def test_set_k_p_single_set_point(self) -> None:
        """

        :return:
        """
        pid = PID(plot_backend='bokeh')
        pid.k_p = 2
        np.testing.assert_equal(pid.k_p, np.array([[2.]]))

    def test_set_t_i_single_set_point(self) -> None:
        """

        :return:
        """
        pid = PID(plot_backend='bokeh')
        pid.t_i = .1
        np.testing.assert_equal(pid.t_i, np.array([[.1]]))
        np.testing.assert_equal(pid.k_i, np.array([[10.]]))

    def test_set_t_d_single_set_point(self) -> None:
        """

        :return:
        """
        pid = PID(plot_backend='bokeh')
        pid.t_d = 10.
        np.testing.assert_equal(pid.t_d, np.array([[10.]]))
        np.testing.assert_equal(pid.k_d, np.array([[10.]]))

    def test_set_k_p_multiple_set_points_coupled(self) -> None:
        """

        :return:
        """
        pid = PID(n_set_points=2, plot_backend='bokeh')
        with self.assertRaises(ValueError) as context:
            pid.k_p = 2.
        self.assertTrue(str(context.exception) == "The number of set points is greater than 1, but the supplied matrix"
                                                  " for K_P is not a diagonal matrix. Coupled multi-variable control is"
                                                  " not supported at the moment.")

    def test_set_t_i_multiple_set_points_coupled(self) -> None:
        """

        :return:
        """
        pid = PID(n_set_points=2, plot_backend='bokeh')
        with self.assertRaises(ValueError) as context:
            pid.t_i = .1
        self.assertTrue(str(context.exception) == "The number of set points is greater than 1, but the supplied matrix"
                                                  " for T_I is not a diagonal matrix. Coupled multi-variable control is"
                                                  " not supported at the moment.")

    def test_set_t_d_multiple_set_points_coupled(self) -> None:
        """

        :return:
        """
        pid = PID(n_set_points=2, plot_backend='bokeh')
        with self.assertRaises(ValueError) as context:
            pid.t_d = 2.
        self.assertTrue(str(context.exception) == "The number of set points is greater than 1, but the supplied matrix"
                                                  " for T_D is not a diagonal matrix. Coupled multi-variable control is"
                                                  " not supported at the moment.")

    def test_set_k_p_multiple_set_points(self) -> None:
        """

        :return:
        """
        pid = PID(n_set_points=2, plot_backend='bokeh')
        pid.k_p = [2., 2.]
        np.testing.assert_equal(pid.k_p, 2. * np.eye(2))
        pid.k_p = 2. * np.eye(2)
        np.testing.assert_equal(pid.k_p, 2. * np.eye(2))

    def test_set_t_i_multiple_set_points(self) -> None:
        """

        :return:
        """
        pid = PID(n_set_points=2, plot_backend='bokeh')
        pid.t_i = [.1, .1]
        np.testing.assert_equal(pid.t_i, .1 * np.eye(2))
        pid.t_i = .1 * np.eye(2)
        np.testing.assert_equal(pid.t_i, .1 * np.eye(2))

    def test_set_t_d_multiple_set_points(self) -> None:
        """

        :return:
        """
        pid = PID(n_set_points=2, plot_backend='bokeh')
        pid.t_d = [10., 10.]
        np.testing.assert_equal(pid.t_d, 10. * np.eye(2))
        pid.t_d = 10. * np.eye(2)
        np.testing.assert_equal(pid.t_d, 10. * np.eye(2))


class TestPIDSetPointSetter(TestCase):
    """"""
    def test_single_set_point(self) -> None:
        """

        :return:
        """
        pid = PID(plot_backend='bokeh')
        pid.setup(dt=.01)
        pid.set_point = 1.

        np.testing.assert_allclose(pid.set_point, np.array([[1.]]))

    def test_multiple_set_points_single_set_point_set(self) -> None:
        """

        :return:
        """
        pid = PID(n_set_points=3, plot_backend='bokeh')
        pid.setup(dt=.01)
        pid.set_point = 1.

        np.testing.assert_allclose(pid.set_point, np.array([[1.], [1.], [1.]]))

    def test_multiple_set_points_multiple_set_points_set(self) -> None:
        """

        :return:
        """
        pid = PID(n_set_points=2, plot_backend='bokeh')
        pid.setup(dt=.01)
        pid.set_point = [1., 1.]

        np.testing.assert_allclose(pid.set_point, np.array([[1.], [1.]]))

    def test_wrong_dimensions(self) -> None:
        """

        :return:
        """
        pid = PID(n_set_points=4, plot_backend='bokeh')
        pid.setup(dt=.01)
        with self.assertRaises(ValueError) as context:
            pid.set_point = [1., 1., 1.]
        self.assertTrue(str(context.exception) == "Dimension mismatch. Supplied dimension for the set point is 3x1, but"
                                                  " required dimension is 4x1.")


class TestPIDCall(TestCase):
    """"""
    def test_p_not_set_up(self) -> None:
        """

        :return:
        """
        pid = PID(plot_backend='bokeh')
        with self.assertRaises(RuntimeError) as context:
            pid.call()
        self.assertTrue(str(context.exception) == "P controller is not set up. Run PID.setup() before calling the P "
                                                  "controller")

    def test_pi_not_set_up(self) -> None:
        """

        :return:
        """
        pid = PID(t_i=.1, plot_backend='bokeh')
        with self.assertRaises(RuntimeError) as context:
            pid.call()
        self.assertTrue(str(context.exception) == "PI controller is not set up. Run PID.setup() before calling the PI "
                                                  "controller")

    def test_pd_not_set_up(self) -> None:
        """

        :return:
        """
        pid = PID(t_d=10., plot_backend='bokeh')
        with self.assertRaises(RuntimeError) as context:
            pid.call()
        self.assertTrue(str(context.exception) == "PD controller is not set up. Run PID.setup() before calling the PD "
                                                  "controller")

    def test_pid_not_set_up(self) -> None:
        """

        :return:
        """
        pid = PID(t_i=.1, t_d=10., plot_backend='bokeh')
        with self.assertRaises(RuntimeError) as context:
            pid.call()
        self.assertTrue(str(context.exception) == "PID controller is not set up. Run PID.setup() before calling the PID"
                                                  " controller")

    def test_single_set_point(self) -> None:
        """

        :return:
        """
        pid = PID(k_p=8., t_i=1., plot_backend='bokeh')
        pid.setup(dt=.01)
        pid.set_point = 1.

        u = pid.call()
        np.testing.assert_allclose(u, np.array([[8.08]]))
        u = pid.call(pv=.1)
        np.testing.assert_allclose(u, np.array([[7.352]]))

    def test_multiple_set_points(self) -> None:
        """

        :return:
        """
        pid = PID(n_set_points=2, k_p=[8., 8.], t_i=[1., 1.], plot_backend='bokeh')
        pid.setup(dt=.01)
        pid.set_point = 1.

        u = pid.call()
        np.testing.assert_allclose(u, np.array([[8.08], [8.08]]))
        u = pid.call(pv=.1)
        np.testing.assert_allclose(u, np.array([[7.352], [7.352]]))
