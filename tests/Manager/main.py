import unittest
from os import devnull
from sys import stdout

from tests_EnvironmentManager import TestEnvironmentManager
from tests_NetworkManager import TestNetworkManager
from tests_DatasetManager import TestDatasetManager


if __name__ == '__main__':
    stdout = open(devnull, 'w')
    unittest.main()
