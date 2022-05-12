import unittest
from os import devnull
from sys import stdout

from tests_DatasetConfig import TestBaseDatasetConfig
from tests_Dataset import TestBaseDataset


if __name__ == '__main__':
    stdout = open(devnull, 'w')
    unittest.main()
