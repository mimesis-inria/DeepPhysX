from unittest import TestCase
from numpy import array, equal, sort
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from DeepPhysX_Core.Manager.EnvironmentManager import EnvironmentManager
from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.AsyncSocket.TcpIpServer import TcpIpServer

from .TestEnvironment import TestEnvironment


class TestEnvironmentManager(TestCase):

    def setUp(self):
        self.env_config_single = BaseEnvironmentConfig(environment_class=TestEnvironment,
                                                       as_tcp_ip_client=False,
                                                       param_dict={'interpolation': [0, 1, 0, 0]})
        self.env_config_tcp_ip = BaseEnvironmentConfig(environment_class=TestEnvironment,
                                                       param_dict={'interpolation': [0, 1, 0, 0]},
                                                       number_of_thread=4)
        self.manager = None

    def tearDown(self):
        if self.manager:
            self.manager.close()

    def test_init_single(self):
        self.manager = EnvironmentManager(environment_config=self.env_config_single)
        # Check default values
        self.assertEqual(self.manager.data_manager, None)
        self.assertEqual(self.manager.number_of_thread, 1)
        self.assertEqual(self.manager.server, None)
        self.assertIsInstance(self.manager.environment, TestEnvironment)
        self.assertEqual(self.manager.batch_size, 1)
        self.assertEqual(self.manager.train, True)
        self.assertEqual(self.manager.visualizer_manager, None)

    def test_init_tcp_ip(self):
        self.manager = EnvironmentManager(environment_config=self.env_config_tcp_ip)
        # Check default values
        self.assertEqual(self.manager.data_manager, None)
        self.assertEqual(self.manager.number_of_thread, 4)
        self.assertIsInstance(self.manager.server, TcpIpServer)
        self.assertEqual(self.manager.environment, None)
        self.assertEqual(self.manager.batch_size, 1)
        self.assertEqual(self.manager.train, True)
        self.assertEqual(self.manager.visualizer_manager, None)

    def test_get_data_single(self):
        self.manager = EnvironmentManager(environment_config=self.env_config_single, batch_size=5)
        # Get a batch size of 5
        data = self.manager.get_data()
        pair = array([2 * i for i in range(1, 6)])
        # A batch is produced only on pair steps
        self.assertEqual(self.manager.environment.idx_step, 10)
        # Check input / output
        self.assertTrue('input' in data)
        self.assertTrue('output' in data)
        self.assertTrue(equal(data['input'], data['output']).all())
        # Check loss
        self.assertTrue('loss' in data)
        self.assertTrue(equal(data['loss'].squeeze(), pair).all())
        # Check additional fields
        self.assertTrue('additional_fields' in data)
        self.assertTrue('step' in data['additional_fields'] and
                        equal(data['additional_fields']['step'].squeeze(), pair).all())
        self.assertTrue('step_1' in data['additional_fields'] and
                        equal(data['additional_fields']['step_1'].squeeze(), pair).all())

    def test_get_data_tcp_ip(self):
        self.manager = EnvironmentManager(environment_config=self.env_config_tcp_ip, batch_size=5)
        # Get a batch size of 5
        data = self.manager.get_data()
        # A batch is produced only on pair steps with 4 Clients
        pair = array([2, 2, 2, 2, 4])
        # Check input / output
        self.assertTrue('input' in data)
        self.assertTrue('output' in data)
        self.assertTrue(equal(data['input'], data['output']).all())
        # Check loss
        self.assertTrue('loss' in data)
        self.assertTrue(equal(data['loss'].squeeze(), pair).all())
        # Check additional fields
        self.assertTrue('additional_fields' in data)
        self.assertTrue('step' in data['additional_fields'] and
                        equal(data['additional_fields']['step'].squeeze(), pair).all())
        self.assertTrue('step_1' in data['additional_fields'] and
                        equal(data['additional_fields']['step_1'].squeeze(), pair).all())

    def test_dispatch_batch_single(self):
        self.manager = EnvironmentManager(environment_config=self.env_config_single, batch_size=5)
        # Get a batch_size of 5
        data = self.manager.get_data()
        data_ = self.manager.dispatch_batch(data)
        # Check input / output
        self.assertTrue(equal(2 * data['input'], data_['input']).all())
        self.assertTrue(equal(2 * data['output'], data_['output']).all())
        # Check additional fields received
        self.assertEqual(list(data_['additional_fields'].keys()), ['full'])
        self.assertFalse(False in data_['additional_fields']['full'])

    def test_dispatch_batch_tcp_ip(self):
        self.manager = EnvironmentManager(environment_config=self.env_config_tcp_ip, batch_size=5)
        # Get a batch of 5
        data = self.manager.get_data()
        data_ = self.manager.dispatch_batch(data)
        # Check input / output
        self.assertTrue(equal(2 * sort(data['input'].squeeze()), sort(data_['input'].squeeze())).all())
        # Check additional fields received
        self.assertFalse(False in data_['additional_fields']['full'])
