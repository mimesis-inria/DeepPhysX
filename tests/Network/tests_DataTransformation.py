import unittest
import numpy as np

from DeepPhysX.Core.Network.BaseNetworkConfig import *


class TestDataTransformation(unittest.TestCase):

    def setUp(self):
        self.transform = BaseNetworkConfig(network_class=BaseNetwork,
                                           optimization_class=BaseOptimization,
                                           data_transformation_class=DataTransformation).create_data_transformation()

    def test_init(self):
        # Default values
        self.assertEqual(self.transform.data_type, any)

    def test_transform_before_prediction(self):
        # Identity
        data = np.random.random((10, 3))
        self.assertTrue(np.equal(self.transform.transform_before_prediction(data), data).all())

    def test_transform_before_loss(self):
        # Identity
        data = np.random.random((10, 3))
        self.assertTrue(np.equal(self.transform.transform_before_loss(data)[0], data).all())

    def test_transform_before_apply(self):
        # Identity
        data = np.random.random((10, 3))
        self.assertTrue(np.equal(self.transform.transform_before_apply(data), data).all())

    def test_check_type(self):
        check_type = DataTransformation.check_type(self.transform.transform_before_prediction)
        data = np.random.random((10, 3))
        # With list
        self.transform.data_type = list
        self.assertEqual(check_type(data.tolist()), data.tolist())
        with self.assertRaises(TypeError):
            check_type(self.transform, data)
        # With array
        self.transform.data_type = np.ndarray
        self.assertTrue(np.equal(check_type(data), data).all())
        with self.assertRaises(TypeError):
            check_type(self.transform, data.tolist())
