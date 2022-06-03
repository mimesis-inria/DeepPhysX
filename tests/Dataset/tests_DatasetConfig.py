from unittest import TestCase
import os


from DeepPhysX.Core.Dataset.BaseDataset import BaseDataset
from DeepPhysX.Core.Dataset.BaseDatasetConfig import BaseDatasetConfig


class TestBaseDatasetConfig(TestCase):

    def setUp(self):
        pass

    def test_init(self):
        # ValueError
        with self.assertRaises(ValueError):
            BaseDatasetConfig(dataset_dir=os.path.join(os.getcwd(), 'dataset'))
            BaseDatasetConfig(partition_size=0)
        # Default values
        dataset_config = BaseDatasetConfig()
        self.assertEqual(dataset_config.dataset_class, BaseDataset)
        self.assertEqual(dataset_config.dataset_dir, None)
        self.assertEqual(dataset_config.shuffle_dataset, True)
        # Config
        self.assertTrue('max_size' in dataset_config.dataset_config._fields)
        self.assertEqual(dataset_config.dataset_config.max_size, 1e9)

    def test_createDataset(self):
        # ValueError
        self.assertRaises(ValueError, BaseDatasetConfig(dataset_class=Test1).create_dataset)
        # TypeError
        self.assertRaises(TypeError, BaseDatasetConfig(dataset_class=Test2).create_dataset)
        # No error
        self.assertIsInstance(BaseDatasetConfig().create_dataset(), BaseDataset)


class Test1:
    def __init__(self):
        pass


class Test2:
    def __init__(self, config):
        pass
