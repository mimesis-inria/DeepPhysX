from unittest import TestCase
from numpy import array, ndarray

from DeepPhysX.Core.AsyncSocket.BytesConverter import BytesConverter


class TestBytesConverter(TestCase):

    def setUp(self):
        self.converter = BytesConverter()
        self.types = {type(None): {'nb_fields': 2, 'equality': lambda a, b: a == b},
                      bytes:      {'nb_fields': 2, 'equality': lambda a, b: a == b},
                      str:        {'nb_fields': 2, 'equality': lambda a, b: a == b},
                      bool:       {'nb_fields': 2, 'equality': lambda a, b: a == b},
                      int:        {'nb_fields': 2, 'equality': lambda a, b: a == b},
                      float:      {'nb_fields': 2, 'equality': lambda a, b: a == b},
                      list:       {'nb_fields': 4, 'equality': lambda a, b: a == b},
                      ndarray:    {'nb_fields': 4, 'equality': lambda a, b: (a == b).all()}}

    def test_conversions(self):
        # Check conversions for all types
        for data in [None, b'test', 'test', True, False, 1, -1, 1., -1., [0.1, 0.1], [[-1, 0], [0, 1]],
                     array([0.1, 0.1], dtype=float), array([[-1, 0], [0, 1]], dtype=int)]:
            # Convert data to bytes fields
            bytes_fields = self.converter.data_to_bytes(data, as_list=True)
            # Get the number of bytes fields used to convert data
            size = self.converter.size_from_bytes(bytes_fields.pop(0))
            self.assertEqual(size, self.types[type(data)]['nb_fields'])
            # Convert bytes fields into data (previous fields are bytes size to write / read on socket)
            recovered_data = self.converter.bytes_to_data(bytes_fields[size:])
            self.assertEqual(type(data), type(recovered_data))
            # Finally, check equality
            self.assertTrue(self.types[type(data)]['equality'](data, recovered_data))
