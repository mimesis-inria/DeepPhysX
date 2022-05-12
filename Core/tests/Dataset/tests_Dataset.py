from unittest import TestCase
import numpy as np
from numpy import ndarray
from os import remove

from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig


class TestBaseDataset(TestCase):

    def setUp(self):
        dataset_config = BaseDatasetConfig()
        self.dataset = dataset_config.create_dataset()

    def test_init(self):
        # Default init values
        self.assertEqual(self.dataset.data_type, ndarray)
        self.assertEqual(self.dataset.current_sample, 0)
        self.assertEqual(self.dataset.nb_samples, 0)
        self.assertEqual(self.dataset.max_size, 1e9)
        self.assertEqual(self.dataset.is_empty(), True)

    def test_check_data(self):
        with self.assertRaises(TypeError):
            for data in [[], (), {}, None]:
                self.dataset.check_data(None, data)
        try:
            self.dataset.check_data(None, np.array([]))
        except TypeError:
            self.fail()

    def test_add(self):
        # TypeError
        with self.assertRaises(TypeError):
            for data in [[], (), {}]:
                self.dataset.add('in', data, None)
        # Adding first batch
        data_in = np.zeros((10, 2, 1))
        data_out = np.ones((10, 2, 1))
        data_new = np.empty((10, 2, 1))
        for field, data in zip(['input', 'output', 'new'], [data_in, data_out, data_new]):
            self.dataset.add(field, data)
            self.assertTrue((self.dataset.data[field] == data).all())
        self.assertEqual(self.dataset.is_empty(), True)
        self.assertEqual(self.dataset.current_sample, 10)
        self.assertEqual(self.dataset.fields, ['input', 'output', 'new'])
        for field in ['input', 'output', 'new']:
            self.assertEqual(self.dataset.batch_per_field[field], 1)
            self.assertEqual(self.dataset.shape[field], (2, 1))
        # Adding following batches
        for field, data in zip(['input', 'output', 'new'], [data_in, data_out, data_new]):
            self.dataset.add(field, data)
            self.assertTrue((self.dataset.data[field] == np.concatenate((data, data))).all())
        self.assertEqual(self.dataset.is_empty(), False)
        self.assertEqual(self.dataset.current_sample, 20)
        for field in ['input', 'output', 'new']:
            self.assertEqual(self.dataset.batch_per_field[field], 2)
        # Adding new field too late
        with self.assertRaises(ValueError):
            self.dataset.add('field', data_new)

    def test_set(self):
        # TypeError
        with self.assertRaises(TypeError):
            for data in [[], (), {}]:
                self.dataset.add('in', data, None)
        # Adding first batch
        data_in = np.zeros((20, 2, 1))
        data_out = np.ones((20, 2, 1))
        data_new = np.empty((20, 2, 1))
        for field, data in zip(['input', 'output', 'new'], [data_in, data_out, data_new]):
            self.dataset.set(field, data)
            self.assertTrue((self.dataset.data[field] == data).all())
        self.assertEqual(self.dataset.is_empty(), False)
        self.assertEqual(self.dataset.current_sample, 0)
        self.assertEqual(self.dataset.fields, ['input', 'output', 'new'])
        for field in ['input', 'output', 'new']:
            self.assertEqual(self.dataset.shape[field], (2, 1))

    def test_get(self):
        data = np.empty((10, 2, 1))
        self.dataset.set('input', data)
        # Access without shuffle
        self.assertTrue((self.dataset.get('input', 0, 5) == data[:5]).all())
        self.assertTrue((self.dataset.get('input', 5, 10) == data[5:]).all())
        # Access with shuffle
        self.dataset.shuffle()
        self.assertTrue((self.dataset.get('input', 0, 5) == data[self.dataset.shuffle_pattern[:5]]).all())
        self.assertTrue((self.dataset.get('input', 5, 10) == data[self.dataset.shuffle_pattern[5:]]).all())

    def test_save(self):
        file = 'test.npy'
        data = np.empty((10, 2, 1))
        # Manually
        self.dataset.set('input', data)
        self.dataset.save('input', file)
        self.assertTrue((np.load(file) == data).all())
        remove(file)
        # Automatically
        self.dataset.add('input', data, file)
        self.assertTrue((np.load(file) == np.concatenate((data, data))).all())
        remove(file)

    def test_empty(self):
        data = np.ones((10, 2, 1))
        self.dataset.add('input', data)
        self.dataset.add('input', data)
        self.assertFalse(self.dataset.is_empty())
        self.assertEqual(self.dataset.current_sample, 20)
        self.assertEqual(self.dataset.batch_per_field['input'], 2)
        # Empty
        self.dataset.empty()
        self.assertTrue(self.dataset.is_empty())
        self.assertEqual(self.dataset.current_sample, 0)
        self.assertEqual(self.dataset.batch_per_field['input'], 0)
        self.assertEqual(len(self.dataset.data['input']), 0)
        self.assertEqual(self.dataset.memory_size(), 0)

    def test_shuffle(self):
        # Adding enough data to minimize the probability to get an identity shuffle
        data = np.arange(100)
        self.dataset.set('input', data)
        # Shuffle
        self.dataset.shuffle()
        self.assertFalse((self.dataset.get('input', 0, 100) == data).all())

    def test_memory_size(self):
        data = np.ones((100, 10, 1))
        self.dataset.set('input', data)
        self.dataset.set('output', np.array(-1. * data, dtype=np.int))
        self.dataset.set('IN_new', np.array(data, dtype=np.bool))
        self.dataset.set('OUT_new', np.array(data, dtype=np.float32))
        for field, size in zip(['input', 'output', 'IN_new', 'OUT_new', None],
                               [8000, 8000, 1000, 4000, 21000]):
            self.assertEqual(self.dataset.memory_size(field), size)

    def test_is_empty(self):
        data = np.empty((10, 2, 1))
        self.dataset.add('input', data)
        self.assertTrue(self.dataset.is_empty())
        self.dataset.add('output', data)
        self.assertTrue(self.dataset.is_empty())
        self.dataset.add('input', data)
        self.assertFalse(self.dataset.is_empty())

    def test_nb_samples(self):
        data = np.empty((10, 2, 1))
        self.dataset.add('input', data)
        self.assertEqual(self.dataset.nb_samples, 10)
        self.dataset.add('output', data)
        self.assertEqual(self.dataset.nb_samples, 10)
        self.dataset.add('input', data)
        self.assertEqual(self.dataset.nb_samples, 20)

    def test_init_data_size(self):
        data_in = np.empty((10, 23, 76))
        data_out = np.empty((10, 45, 98))
        self.dataset.init_data_size('input', data_in[0].shape)
        self.dataset.init_data_size('output', data_out[0].shape)
        self.assertEqual(self.dataset.get_data_shape('input'), (23, 76))
        self.assertEqual(self.dataset.get_data_shape('output'), (45, 98))

    def test_init_additional_field(self):
        data = np.empty((10, 2, 1))
        self.dataset.init_additional_field('field', data[0].shape)
        self.assertEqual(self.dataset.fields, ['input', 'output', 'field'])
        self.assertTrue('field' in self.dataset.batch_per_field)
        self.assertTrue('field' in self.dataset.data)


