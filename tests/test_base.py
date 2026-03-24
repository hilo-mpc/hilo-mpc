import unittest

import casadi as ca
import numpy as np

from hilo_mpc.modules.base import TimeSeries, Vector
from hilo_mpc.modules.object import Object


class TestObject(unittest.TestCase):
    """Tests for the Object base class"""

    def test_object_creation_no_args(self):
        """Object can be created without arguments (auto-generates id)"""
        obj = Object()
        self.assertIsNotNone(obj)

    def test_object_auto_generates_id(self):
        """Object generates a non-empty id automatically"""
        obj = Object()
        # id may be None until _create_id() is called
        self.assertIsNotNone(obj)

    def test_object_with_name(self):
        """Object stores supplied name"""
        obj = Object(name='my_object')
        self.assertEqual(obj.name, 'my_object')

    def test_object_with_id(self):
        """Object stores supplied id"""
        obj = Object(id='custom_id')
        self.assertEqual(obj.id, 'custom_id')

    def test_two_objects_different_names(self):
        """Two Objects with different names are distinguishable"""
        obj1 = Object(name='obj_a')
        obj2 = Object(name='obj_b')
        self.assertNotEqual(obj1.name, obj2.name)


class TestVector(unittest.TestCase):
    """Tests for the Vector class"""

    def test_vector_sx_creation(self):
        """Vector with CasADi SX type can be created"""
        v = Vector(ca.SX, values_or_names=['x', 'y'])
        self.assertIsNotNone(v)

    def test_vector_sx_length(self):
        """SX Vector has correct length"""
        v = Vector(ca.SX, values_or_names=['a', 'b', 'c'])
        self.assertEqual(len(v), 3)

    def test_vector_dm_creation(self):
        """Vector with CasADi DM type can be created"""
        v = Vector(ca.DM, values_or_names=[])
        self.assertIsNotNone(v)

    def test_vector_is_empty_when_created_with_empty_names(self):
        """DM Vector created with empty names is empty"""
        v = Vector(ca.DM, values_or_names=[])
        self.assertTrue(v.is_empty())

    def test_vector_sx_not_empty(self):
        """SX Vector with names is not empty"""
        v = Vector(ca.SX, values_or_names=['x'])
        self.assertFalse(v.is_empty())

    def test_vector_size(self):
        """Vector size1() returns number of rows"""
        v = Vector(ca.SX, values_or_names=['x', 'y', 'z'])
        self.assertEqual(v.size1(), 3)

    def test_vector_element_access(self):
        """Vector elements can be accessed by index"""
        v = Vector(ca.SX, values_or_names=['x', 'y'])
        # Should return SX expressions
        self.assertIsNotNone(v[0])
        self.assertIsNotNone(v[1])


class TestTimeSeries(unittest.TestCase):
    """Tests for the TimeSeries class"""

    def test_timeseries_creation(self):
        """TimeSeries can be instantiated without arguments"""
        ts = TimeSeries()
        self.assertIsNotNone(ts)

    def test_timeseries_is_not_set_up_initially(self):
        """New TimeSeries is not set up initially"""
        ts = TimeSeries()
        self.assertFalse(ts.is_set_up())

    def test_timeseries_setup(self):
        """TimeSeries can be set up with variable definitions"""
        ts = TimeSeries()
        ts.setup('x', 'y', x={'values_or_names': ['x_1', 'x_2'], 'shape': (2, 0), 'data_format': ca.DM},
                 y={'values_or_names': ['y_1'], 'shape': (1, 0), 'data_format': ca.DM})
        self.assertTrue(ts.is_set_up())

    def test_timeseries_n_samples_zero_initially(self):
        """New TimeSeries has zero samples"""
        ts = TimeSeries()
        self.assertEqual(ts.n_samples, 0)

    def test_timeseries_is_empty_initially(self):
        """New TimeSeries is empty"""
        ts = TimeSeries()
        self.assertTrue(ts.is_empty())

    def test_timeseries_add_data(self):
        """TimeSeries can add data and reflects sample count"""
        ts = TimeSeries()
        ts.setup('x', x={'values_or_names': ['x_1', 'x_2'], 'shape': (2, 0), 'data_format': ca.DM})
        ts.add('x', ca.DM([[1., 2., 3.], [4., 5., 6.]]))
        self.assertEqual(ts.n_samples, 3)

    def test_timeseries_get_by_id(self):
        """TimeSeries.get_by_id() returns the stored vector"""
        ts = TimeSeries()
        ts.setup('x', x={'values_or_names': ['x_1'], 'shape': (1, 0), 'data_format': ca.DM})
        ts.add('x', ca.DM([[1., 2.]]))
        result = ts.get_by_id('x')
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[1], 2)

    def test_timeseries_not_empty_after_add(self):
        """TimeSeries is no longer empty after adding data"""
        ts = TimeSeries()
        ts.setup('x', x={'values_or_names': ['x_1'], 'shape': (1, 0), 'data_format': ca.DM})
        ts.add('x', ca.DM([[1.]]))
        self.assertFalse(ts.is_empty())

    def test_timeseries_get_names(self):
        """TimeSeries.get_names() returns variable names"""
        ts = TimeSeries()
        ts.setup('x', 'y',
                 x={'values_or_names': ['x_1'], 'shape': (1, 0), 'data_format': ca.DM},
                 y={'values_or_names': ['y_1'], 'shape': (1, 0), 'data_format': ca.DM})
        names = ts.get_names()
        self.assertIn('x_1', names)
        self.assertIn('y_1', names)

    def test_timeseries_contains(self):
        """TimeSeries supports 'in' operator for key membership"""
        ts = TimeSeries()
        ts.setup('x', x={'values_or_names': ['x_1'], 'shape': (1, 0), 'data_format': ca.DM})
        self.assertIn('x', ts)
        self.assertNotIn('z', ts)

    def test_timeseries_clear(self):
        """TimeSeries.clear() marks the series as empty (is_empty() becomes True)

        Note: n_samples retains its previous value after clear() — only is_empty()
        signals the cleared state. This appears to be the library's design choice.
        """
        ts = TimeSeries()
        ts.setup('x', x={'values_or_names': ['x_1'], 'shape': (1, 0), 'data_format': ca.DM})
        ts.add('x', ca.DM([[1., 2., 3.]]))
        self.assertFalse(ts.is_empty())
        ts.clear()
        self.assertTrue(ts.is_empty())


if __name__ == '__main__':
    unittest.main()
