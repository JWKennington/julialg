"""Unittests for julialg.array module"""

import unittest

import numpy
from julialg import formatting


class ArrayTests(unittest.TestCase):
    def setUp(self):
        self.a = numpy.arange(1, 2**4 + 1).reshape((2,2,2,2))

    def test_format_float(self):
        self.assertEqual(formatting.format_float.pyfunc(0.123456789, 8, 4), 
                         '  0.1235')
