import unittest
from os import devnull
from sys import stdout

from tests_EnvironmentConfig import TestBaseEnvironmentConfig
from tests_Environment import TestBaseEnvironment


if __name__ == '__main__':
    stdout = open(devnull, 'w')
    unittest.main()
