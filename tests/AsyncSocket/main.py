import unittest
from os import devnull
from sys import stdout

from tests_BytesConverter import TestBytesConverter
from tests_TcpIpObject import TestTcpIpObjects


if __name__ == '__main__':
    stdout = open(devnull, 'w')
    unittest.main()
