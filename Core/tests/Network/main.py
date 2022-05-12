import unittest
from os import devnull
from sys import stdout

from tests_NetworkConfig import TestNetworkConfig
from tests_Network import TestNetwork
from tests_Optimization import TestOptimization
from tests_DataTransformation import TestDataTransformation


if __name__ == '__main__':
    stdout = open(devnull, 'w')
    unittest.main()
