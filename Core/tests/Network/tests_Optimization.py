import unittest

from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig


class TestOptimization(unittest.TestCase):

    def setUp(self):
        self.optimization = BaseNetworkConfig().create_optimization()

    def test_init(self):
        # Default values
        self.assertEqual(self.optimization.manager, None)
        self.assertEqual(self.optimization.loss_class, None)
        self.assertEqual(self.optimization.loss_value, 0.)
        self.assertEqual(self.optimization.loss, None)
        self.assertEqual(self.optimization.optimizer_class, None)
        self.assertEqual(self.optimization.optimizer, None)
        self.assertEqual(self.optimization.lr, None)

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.optimization.set_loss()
            self.optimization.compute_loss(None, None, None)
            self.optimization.transform_loss(None)
            self.optimization.set_optimizer(None)
            self.optimization.optimize()
