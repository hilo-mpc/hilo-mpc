import unittest

import numpy as np

from hilo_mpc.library.models import cstr_schaffner_and_zeitz, cstr_seborg
from hilo_mpc.util.plotting import set_plot_backend

set_plot_backend('bokeh')


class TestCSTRSchaffnerAndZeitz(unittest.TestCase):
    """Tests for the Schaffner & Zeitz CSTR benchmark model"""

    def test_model_creation(self):
        """cstr_schaffner_and_zeitz() returns a Model without error"""
        model = cstr_schaffner_and_zeitz()
        self.assertIsNotNone(model)

    def test_model_has_dynamical_states(self):
        """CSTR model has 2 dynamical states (x_1, x_2)"""
        model = cstr_schaffner_and_zeitz()
        self.assertEqual(model.n_x, 2)

    def test_model_has_inputs(self):
        """CSTR model has at least 1 input"""
        model = cstr_schaffner_and_zeitz()
        self.assertGreater(model.n_u, 0)

    def test_model_has_measurements(self):
        """CSTR model has measurement equations"""
        model = cstr_schaffner_and_zeitz()
        self.assertGreater(model.n_y, 0)

    def test_model_name(self):
        """CSTR model is named 'CSTR'"""
        model = cstr_schaffner_and_zeitz()
        self.assertEqual(model.name, 'CSTR')

    def test_model_setup(self):
        """CSTR model sets up (discretizes) without error"""
        model = cstr_schaffner_and_zeitz()
        model.setup(dt=0.1)
        self.assertTrue(model.is_setup())

    def test_model_simulation(self):
        """CSTR model can be simulated for a few steps with parameters"""
        model = cstr_schaffner_and_zeitz()
        model.setup(dt=0.1)
        model.set_initial_conditions(x0=[0.2, 0.1])
        # Parameters: a_1, b_1, E, a_2, b_2, g (Schaffner & Zeitz values)
        p = [1.0, 1.0, 20.0, 1.0, 1.0, 1.0]
        model.set_initial_parameter_values(p=p)
        for _ in range(3):
            model.simulate(u=[0.], p=p)
        # If we got here without exception, simulation worked
        self.assertTrue(True)


class TestCSTRSeborg(unittest.TestCase):
    """Tests for the Seborg CSTR benchmark model"""

    def test_model_creation(self):
        """cstr_seborg() returns a Model without error"""
        model = cstr_seborg()
        self.assertIsNotNone(model)

    def test_model_has_dynamical_states(self):
        """Seborg CSTR model has 3 dynamical states (C_A, T, T_c)"""
        model = cstr_seborg()
        self.assertEqual(model.n_x, 3)

    def test_model_has_inputs(self):
        """Seborg CSTR model has at least 1 input"""
        model = cstr_seborg()
        self.assertGreater(model.n_u, 0)

    def test_model_has_measurements(self):
        """Seborg CSTR model has measurement equations"""
        model = cstr_seborg()
        self.assertGreater(model.n_y, 0)

    def test_model_name(self):
        """Seborg CSTR model is named 'CSTR'"""
        model = cstr_seborg()
        self.assertEqual(model.name, 'CSTR')

    def test_model_setup(self):
        """Seborg CSTR model sets up without error"""
        model = cstr_seborg()
        model.setup(dt=0.1)
        self.assertTrue(model.is_setup())


if __name__ == '__main__':
    unittest.main()
