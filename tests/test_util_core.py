import unittest

import numpy as np

from hilo_mpc.util.util import check_and_wrap_to_list, scale_vector


class MyTestCase(unittest.TestCase):
    def test_check_and_wrap_to_list(self):
        # 1-d np array
        a = np.array([1, 2, 3])

        a_new = check_and_wrap_to_list(a)

        assert isinstance(a_new, list)
        print(a_new)

        # 2-d np array
        a = np.array([[1, 2, 3]])

        a_new = check_and_wrap_to_list(a)

        assert isinstance(a_new, list)
        print(a_new)

        # 3-d np array
        a = np.array([[[1, 2, 3]]])

        a_new = check_and_wrap_to_list(a)

        assert isinstance(a_new, list)
        print(a_new)

        a = [1, 2, 3]
        a_new = check_and_wrap_to_list(a)

        assert isinstance(a_new, list)
        print(a_new)

        a = (1, 2, 3)
        self.assertRaises(TypeError, check_and_wrap_to_list, a)

    def test_scaler_1(self):
        a = np.array([10, 10, 10])
        scaler = [10, 10, 10]
        s_a = scale_vector(a, scaler)
        self.assertTrue(all([i == 1.0 for i in s_a]), 'something is wrong.')

    def test_scaler_2(self):
        a = np.array([10, 10, 10])
        scaler = [10, 10, 10, 10]
        self.assertRaises(ValueError, scale_vector, a, scaler)

    if __name__ == '__main__':
        unittest.main()
