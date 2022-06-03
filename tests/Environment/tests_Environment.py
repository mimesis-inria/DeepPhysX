from unittest import TestCase
import numpy as np

from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment


class TestBaseEnvironment(TestCase):

    def setUp(self):
        self.env = BaseEnvironment(as_tcp_ip_client=False)

    def test_init(self):
        # Default values at init
        for attribute in [self.env.input, self.env.output]:
            self.assertTrue(type(attribute), np.ndarray)
        for attribute in [self.env.sample_in, self.env.sample_out]:
            self.assertEqual(attribute, None)
        for attribute in [self.env.loss_data, self.env.environment_manager]:
            self.assertEqual(attribute, None)
        for attribute in [self.env.additional_fields]:
            self.assertEqual(attribute, {})

    def test_not_implemented(self):
        # Check not implemented functions
        with self.assertRaises(NotImplementedError):
            self.env.create()
            self.env.step()

    # def test_set_dataset_sample(self):
    #     # Create random samples
    #     sample_in, sample_out = np.random.random((3, 1)), np.random.random((3, 3))
    #     additional_in, additional_out = np.random.random((3, 1)), np.random.random((3, 3))
    #     # Set as dataset samples
    #     self.env.set_dataset_sample(sample_in, sample_out, {'additional_data': additional_in},
    #                                 {'additional_data': additional_out})
    #     # Check samples are well-defined in environment
    #     for data, env_data in zip([sample_in, sample_out],
    #                               [self.env.sample_in, self.env.sample_out]):
    #         self.assertTrue(np.equal(data, env_data).all())
    #     # Check additional samples are well defined too
    #     for data, env_data in zip([additional_in, additional_out],
    #                               [self.env.additional_inputs, self.env.additional_outputs]):
    #         self.assertTrue('additional_data' in env_data and (data == env_data['additional_data']).all())

    def test_set_training_data(self):
        # Create random samples
        sample_in, sample_out = np.random.random((3, 1)), np.random.random((3, 3))
        # Set as training data
        self.env.set_training_data(sample_in, sample_out)
        # Check training data is well-defined in environment
        for data, env_data in zip([sample_in, sample_out], [self.env.input, self.env.output]):
            self.assertTrue(np.equal(data, env_data).all())

    def test_set_loss_data(self):
        # Create random sample
        loss = np.random.random((1,))
        # Set as loss data
        self.env.set_loss_data(loss)
        # Check loss data is well-defined in environment
        self.assertTrue(np.equal(self.env.loss_data, loss).all())

    def test_additional_dataset(self):
        # Check initial state of the additional fields
        for attribute in [self.env.additional_fields]:
            self.assertEqual(attribute, {})
        # Create random samples
        additional_field = np.random.random((3, 1))
        # Set as additional inputs and outputs
        self.env.set_additional_dataset('field', additional_field)
        # Check additional data are well-defined in environment
        self.assertTrue('field' in self.env.additional_fields)
        self.assertTrue(np.equal(self.env.additional_fields['field'], additional_field).all())
