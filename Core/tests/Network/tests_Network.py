import unittest

from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.network = BaseNetworkConfig().create_network()

    def test_init(self):
        # Default values
        self.assertEqual(self.network.device, None)
        self.assertEqual(self.network.config.network_name, 'Network')
        self.assertEqual(self.network.config.network_type, 'BaseNetwork')

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.network.predict(None)
            self.network.forward(None)
            self.network.set_train()
            self.network.set_eval()
            self.network.set_device()
            self.network.load_parameters(None)
            self.network.get_parameters()
            self.network.save_parameters()
            self.network.transform_from_numpy(None)
            self.network.transform_to_numpy(None)

