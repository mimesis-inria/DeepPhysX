from unittest import TestCase
import os

from DeepPhysX.Core.Network.BaseNetworkConfig import BaseNetworkConfig, BaseNetwork, BaseOptimization, DataTransformation


class TestNetworkConfig(TestCase):

    def setUp(self):
        pass

    def test_init(self):
        # TypeError
        with self.assertRaises(TypeError):
            BaseNetworkConfig(network_dir=0)
            BaseNetworkConfig(network_name=0)
            BaseNetworkConfig(which_network="0")
            BaseNetworkConfig(save_each_epoch=-1)
        # ValueError
        with self.assertRaises(ValueError):
            BaseNetworkConfig(network_dir=os.path.join(os.getcwd(), 'network'))
            BaseNetworkConfig(which_network=-1)
        # Default values
        network_config = BaseNetworkConfig()
        self.assertEqual(network_config.network_class, BaseNetwork)
        self.assertEqual(network_config.optimization_class, BaseOptimization)
        self.assertEqual(network_config.data_transformation_class, DataTransformation)
        self.assertEqual(network_config.network_dir, None)
        self.assertEqual(network_config.training_stuff, False)
        self.assertEqual(network_config.which_network, 0)
        self.assertEqual(network_config.save_each_epoch, False)
        # Network config
        self.assertTrue('network_name' in network_config.network_config._fields)
        self.assertEqual(network_config.network_config.network_name, 'Network')
        self.assertTrue('network_type' in network_config.network_config._fields)
        self.assertEqual(network_config.network_config.network_type, 'BaseNetwork')
        # Optimization config
        self.assertTrue('loss' in network_config.optimization_config._fields)
        self.assertEqual(network_config.optimization_config.loss, None)
        self.assertTrue('lr' in network_config.optimization_config._fields)
        self.assertEqual(network_config.optimization_config.lr, None)
        self.assertTrue('optimizer' in network_config.optimization_config._fields)
        self.assertEqual(network_config.optimization_config.optimizer, None)

    def test_create_network(self):
        # ValueError
        self.assertRaises(ValueError, BaseNetworkConfig(network_class=Test1).create_network)
        # TypeError
        self.assertRaises(TypeError, BaseNetworkConfig(network_class=Test2).create_network)
        # No error
        self.assertIsInstance(BaseNetworkConfig().create_network(), BaseNetwork)

    def test_create_optimization(self):
        # ValueError
        self.assertRaises(ValueError, BaseNetworkConfig(optimization_class=Test1).create_optimization)
        # TypeError
        self.assertRaises(TypeError, BaseNetworkConfig(optimization_class=Test2).create_optimization)
        # No error
        self.assertIsInstance(BaseNetworkConfig().create_optimization(), BaseOptimization)

    def test_create_data_transformation(self):
        # ValueError
        self.assertRaises(ValueError, BaseNetworkConfig(data_transformation_class=Test1).create_data_transformation)
        # TypeError
        self.assertRaises(TypeError, BaseNetworkConfig(data_transformation_class=Test2).create_data_transformation)
        # No error
        self.assertIsInstance(BaseNetworkConfig().create_data_transformation(), DataTransformation)


class Test1:
    def __init__(self):
        pass


class Test2:
    def __init__(self, config):
        pass
