from unittest import TestCase
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DeepPhysX.Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
import TestEnvironment as Env


class TestBaseEnvironmentConfig(TestCase):

    def setUp(self):
        pass

    def test_init(self):
        # TypeError
        with self.assertRaises(TypeError):
            BaseEnvironmentConfig(environment_class=Env.TestEnvironment, simulations_per_step="0")
            BaseEnvironmentConfig(environment_class=Env.TestEnvironment, max_wrong_samples_per_step="0")
            BaseEnvironmentConfig(environment_class=Env.TestEnvironment, always_create_data="0")
            BaseEnvironmentConfig(environment_class=Env.TestEnvironment, number_of_thread="0")
        # ValueError
        with self.assertRaises(ValueError):
            BaseEnvironmentConfig(environment_class=Env.TestEnvironment, simulations_per_step=0)
            BaseEnvironmentConfig(environment_class=Env.TestEnvironment, max_wrong_samples_per_step=0)
            BaseEnvironmentConfig(environment_class=Env.TestEnvironment, number_of_thread=-1)
        # Default values
        environment_config = BaseEnvironmentConfig(environment_class=Env.TestEnvironment)
        # TcpIpClient values
        self.assertEqual(environment_config.environment_class, Env.TestEnvironment)
        self.assertEqual(environment_config.environment_file, Env.__file__)
        self.assertEqual(environment_config.param_dict, {})
        self.assertEqual(environment_config.as_tcp_ip_client, True)
        # TcpIpServer values
        self.assertEqual(environment_config.ip_address, 'localhost')
        self.assertEqual(environment_config.port, 10000)
        self.assertEqual(environment_config.number_of_thread, 1)
        self.assertEqual(environment_config.max_client_connections, 1000)
        # EnvironmentManager values
        self.assertEqual(environment_config.always_create_data, False)
        self.assertEqual(environment_config.record_wrong_samples, False)
        self.assertEqual(environment_config.screenshot_sample_rate, 0)
        self.assertEqual(environment_config.use_dataset_in_environment, False)
        self.assertEqual(environment_config.simulations_per_step, 1)
        self.assertEqual(environment_config.max_wrong_samples_per_step, 10)
        self.assertEqual(environment_config.visualizer, None)

    def test_create_environment(self):
        # ValueError
        with self.assertRaises(ValueError):
            BaseEnvironmentConfig(environment_class=Test1).create_environment(environment_manager=None)
        # TypeError
        with self.assertRaises(TypeError):
            BaseEnvironmentConfig(environment_class=Test2).create_environment(environment_manager=None)
        # No error
        environment_config = BaseEnvironmentConfig(environment_class=Env.TestEnvironment)
        environment = environment_config.create_environment(environment_manager=None)
        # Check the instance of environment created
        self.assertIsInstance(environment, Env.TestEnvironment)
        # Check the automatic call of init functions
        self.assertTrue(environment.call_create)
        self.assertTrue(environment.call_init)

    def test_create_server(self):
        parameters = {'multiply_by_2': 10,
                      'multiply_by_4': 10}
        environment_config = BaseEnvironmentConfig(environment_class=Env.TestEnvironment,
                                                   param_dict=parameters)
        server = environment_config.create_server(environment_manager=None)
        # Check client and server are connected
        self.assertTrue(environment_config.server_is_ready)
        # Check the exchange of parameters : send a dict, receive a modified dict of parameters
        self.assertEqual(len(environment_config.received_parameters.keys()), 1)
        self.assertEqual(parameters.keys(), environment_config.received_parameters[0].keys())
        for key, res in zip(parameters.keys(), [20, 40]):
            self.assertEqual(environment_config.received_parameters[0][key], res)
        # Do not forget to close client
        server.close()


class Test1:
    def __init__(self):
        pass


class Test2:
    def __init__(self, environment_manager, as_tcp_ip_client):
        pass
