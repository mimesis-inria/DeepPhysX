from unittest import TestCase
import shutil
from numpy import array, arange, load
from numpy.random import shuffle
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from DeepPhysX_Core.Manager.NetworkManager import NetworkManager

from TestNetwork import NumpyNetwork, NumpyOptimisation, NumpyDataTransformation, NumpyNetworkConfig


class TestNetworkManager(TestCase):

    def setUp(self):
        self.dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'network/')
        self.net_config = NumpyNetworkConfig(network_class=NumpyNetwork,
                                             optimization_class=NumpyOptimisation,
                                             data_transformation_class=NumpyDataTransformation,
                                             network_name='InterpolationNetwork',
                                             network_type='Interpolation',
                                             save_each_epoch=True,
                                             lr=0.1)
        self.manager = None

    def tearDown(self):
        if self.manager:
            self.manager.close()
        for folder in [f for f in os.listdir(os.getcwd()) if f.__contains__('network')]:
            if os.path.isdir(folder):
                shutil.rmtree(folder)

    def test_init(self):
        self.manager = NetworkManager(network_config=self.net_config,
                                      session_dir=os.getcwd())
        # Check default values
        self.assertEqual(self.manager.session_dir, os.getcwd())
        self.assertEqual(self.manager.new_session, True)
        self.assertEqual(self.manager.network_dir, self.dir)
        self.assertEqual(self.manager.network_template_name, 'default_network_{}')
        self.assertEqual(self.manager.training, True)
        self.assertEqual(self.manager.save_each_epoch, True)
        self.assertEqual(self.manager.saved_counter, 0)

    def test_set_network(self):
        self.manager = NetworkManager(network_config=self.net_config,
                                      session_dir=os.getcwd())
        # Check default values
        self.assertIsInstance(self.manager.network, NumpyNetwork)
        self.assertIsInstance(self.manager.optimization, NumpyOptimisation)
        self.assertIsInstance(self.manager.data_transformation, NumpyDataTransformation)

    def test_load_network(self):
        # Close and save parameters
        self.manager = NetworkManager(network_config=self.net_config,
                                      session_dir=os.getcwd())
        params = self.manager.network.get_parameters()
        self.manager.close()
        # Create from existing network
        self.net_config.network_dir = self.dir
        self.manager = NetworkManager(network_config=self.net_config,
                                      session_dir=os.getcwd())
        self.assertTrue((self.manager.network.get_parameters() == params).all())
        self.net_config.network_dir = None

    def test_compute_prediction_and_loss(self):
        self.manager = NetworkManager(network_config=self.net_config,
                                      session_dir=os.getcwd())
        self.manager.network.p = array([1., 2., 0.])
        # No optimization
        batch = {'input': array([1.]),
                 'output': array([1.])}
        prediction, loss = self.manager.compute_prediction_and_loss(batch, optimize=False)
        self.assertEqual(prediction, 3.)
        self.assertEqual(loss, 2. * 2.)
        self.assertTrue((self.manager.network.get_parameters() == array([1., 2., 0.])).all())
        # With optimization
        inputs = array([[float(i / 2)] for i in range(-100, 101)]) * 1e-2
        outputs = array([[0.5 + x - 1.5 * (x ** 2)] for x in inputs])
        index = arange(len(inputs))
        for _ in range(100):
            shuffle(index)
            for i in index:
                batch = {'input': inputs[i], 'output': outputs[i]}
                self.manager.compute_prediction_and_loss(batch, optimize=True)
        for state, target in zip(self.manager.network.get_parameters(), [0.5, 1, -1.5]):
            self.assertTrue(abs(0.98 * target) < abs(state) < abs(1.02 * target))

    def test_compute_online_prediction(self):
        self.manager = NetworkManager(network_config=self.net_config,
                                      session_dir=os.getcwd())
        self.manager.network.p = array([1., 2., -1.])
        # Online predictions
        inputs = array([[float(i / 2)] for i in range(-10, 11)]) * 1e-2
        outputs = array([[1 + 2 * x - 1 * (x ** 2)] for x in inputs])
        for i, o in zip(inputs, outputs):
            prediction = self.manager.compute_online_prediction(i)
            self.assertEqual(prediction, o)

    def test_save_network(self):
        self.manager = NetworkManager(network_config=self.net_config,
                                      session_dir=os.getcwd())
        params = self.manager.network.get_parameters()
        # Multiple saves
        for _ in range(3):
            self.manager.save_network()
        self.manager.save_network(last_save=True)
        # Check names
        saved_networks = sorted(os.listdir(self.manager.network_dir))
        self.assertEqual(saved_networks, [f'default_network_{i}.npy' for i in (0, 1, 2)] + ['network.npy'])
        # Check parameters
        for net_params in saved_networks:
            self.assertTrue((load(os.path.join(self.manager.network_dir, net_params)) == params).all())

    def test_close(self):
        self.manager = NetworkManager(network_config=self.net_config,
                                      session_dir=os.getcwd())
        self.manager.close()
        self.assertEqual(os.listdir(self.manager.network_dir), ['network.npy'])
        self.assertRaises(AttributeError, self.manager.close)
        self.manager = None
