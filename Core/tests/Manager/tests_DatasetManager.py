from unittest import TestCase
import os
import shutil
from numpy import array, arange

from DeepPhysX_Core.Manager.DatasetManager import DatasetManager
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig, BaseDataset


class TestDatasetManager(TestCase):

    def setUp(self):
        self.data_config = BaseDatasetConfig(shuffle_dataset=False)
        self.manager = None

    def tearDown(self):
        for folder in [f for f in os.listdir(os.getcwd()) if f.__contains__('dataset')]:
            if os.path.isdir(folder):
                shutil.rmtree(folder)

    def test_init(self):
        self.manager = DatasetManager(dataset_config=self.data_config,
                                      session_dir=os.getcwd())
        # Check default values
        self.assertIsInstance(self.manager.dataset, BaseDataset)
        self.assertEqual(self.manager.max_size, 1e9)
        self.assertEqual(self.manager.shuffle_dataset, False)
        self.assertFalse(False in self.manager.record_data.values())
        self.assertEqual(self.manager.mode, 0)
        self.assertEqual(len(self.manager.partitions_templates), 3)
        self.assertFalse(False in [partitions == [[], [], []] for partitions in self.manager.list_partitions.values()])
        self.assertFalse(False in [idx == 0 for idx in self.manager.idx_partitions])
        self.assertFalse(False in [current is None for current in self.manager.current_partition_path.values()])
        for attribute in [self.manager.mul_part_idx, self.manager.mul_part_slices, self.manager.mul_part_list_path]:
            self.assertEqual(attribute, None)
        self.assertEqual(self.manager.dataset_dir, os.path.join(os.getcwd(), 'dataset/'))
        self.assertEqual(self.manager.new_session, True)
        # Check repository creation
        self.assertTrue(os.path.isdir(self.manager.dataset_dir))

    def test_add_data(self):
        self.manager = DatasetManager(dataset_config=self.data_config, session_dir=os.getcwd())
        # Add a batch
        data = {'input': array([[i] for i in range(10)]),
                'output': array([[2 * i] for i in range(10)])}
        self.manager.add_data(data)
        # Check dataset
        for field, value in data.items():
            self.assertTrue((self.manager.dataset.data[field] == value).all())
        # Check repository
        self.assertFalse(self.manager.first_add)
        self.assertFalse(False in [len(partitions[self.manager.mode]) == 1 for partitions in
                                   self.manager.list_partitions.values()])
        self.assertFalse(False in [current is not None for current in self.manager.current_partition_path.values()])
        self.assertEqual(self.manager.idx_partitions[self.manager.mode], 1)

    def test_get_data(self):
        self.manager = DatasetManager(dataset_config=self.data_config, session_dir=os.getcwd())
        # Add a batch
        data = {'input': array([[i] for i in range(10)]),
                'output': array([[2 * i] for i in range(10)])}
        self.manager.add_data(data)
        # Get a batch
        batch = self.manager.get_data(True, True, 5)
        for field, value in data.items():
            self.assertTrue((batch[field] == value[:5]).all())
        batch = self.manager.get_data(True, True, 5)
        for field, value in data.items():
            self.assertTrue((batch[field] == value[5:]).all())
        batch = self.manager.get_data(True, True, 8)
        for field, value in data.items():
            self.assertTrue((batch[field] == value[:8]).all())
        batch = self.manager.get_data(True, True, 8)
        for field, value in data.items():
            self.assertEqual(batch[field].tolist(), value[-2:].tolist() + value[:6].tolist())

    def test_register_new_fields(self):
        self.manager = DatasetManager(dataset_config=self.data_config, session_dir=os.getcwd())
        fields = ['IN_field1', 'OUT_field1', 'OUT_field2']
        self.manager.register_new_fields(fields)
        # Check fields
        self.assertEqual(len(self.manager.fields), 2 + len(fields))
        self.assertFalse(False in [field in self.manager.list_partitions for field in fields])
        self.assertFalse(False in [self.manager.list_partitions[field] == [[], [], []] for field in fields])
        self.assertFalse(False in [self.manager.record_data[field] for field in fields])

    def test_load_multiple_dataset(self):
        # Produce a Dataset with several fields and small partition size
        dataset_config = BaseDatasetConfig(partition_size=1e-6)
        manager = DatasetManager(dataset_config=dataset_config, session_dir=os.getcwd())
        input_array, duplicated_in = arange(0, 100), arange(0, 100)
        output_array, duplicated_out = 2 * input_array, 2 * duplicated_in
        for i in range(100):
            data = {'input': input_array[10 * i: 10 * (i + 1)],
                    'output': output_array[10 * i: 10 * (i + 1)],
                    'additional_fields': {'duplicated_in': duplicated_in[10 * i: 10 * (i + 1)],
                                          'duplicated_out': duplicated_out[10 * i: 10 * (i + 1)]}}
            manager.add_data(data)
        manager.close()
        # Load the multiple partitions
        dataset_config = BaseDatasetConfig(dataset_dir=os.getcwd(), shuffle_dataset=False)
        self.manager = DatasetManager(dataset_config=dataset_config)
        input_get, duplicated_in_get = [], []
        output_get, duplicated_out_get = [], []
        for _ in range(5):
            data = self.manager.get_data(get_inputs=True, get_outputs=True, batch_size=10)
            input_get += data['input'].tolist()
            output_get += data['output'].tolist()
            duplicated_in_get += data['additional_fields']['duplicated_in'].tolist()
            duplicated_out_get += data['additional_fields']['duplicated_out'].tolist()
        # Check the concatenated batches received
        val = lambda x, y, s: arange(x, y, s).tolist()
        input_expected = val(0, 14, 1) + val(40, 54, 1) + val(80, 87, 1) + val(14, 28, 1) + [54]
        output_expected = val(0, 28, 2) + val(80, 108, 2) + val(160, 174, 2) + val(28, 56, 2) + [108]
        self.assertEqual(input_get, input_expected)
        self.assertEqual(duplicated_in_get, input_expected)
        self.assertEqual(output_get, output_expected)
        self.assertEqual(duplicated_out_get, output_expected)
