"""Unittests for julialg.array module"""

import unittest

import numpy
from julialg.array import JulArray


class StringLookLikeMixin:
    def assert_looks_like(self, x, looks_like: str):
        x = '\n'.join([line.strip() for line in str(x).split('\n') if line.strip()])
        ll_str = '\n'.join([line.strip() for line in looks_like.split('\n') if line.strip()])
        self.assertEqual(x, ll_str, '\nExpected:\n{}\n\nGot:\n{}\n\n'.format(ll_str, x))


class ArrayTests(unittest.TestCase, StringLookLikeMixin):
    def setUp(self):
        self.a = numpy.arange(1, 2**4 + 1).reshape((2,2,2,2))
        self.j = JulArray(self.a)
        self.j2 = JulArray(self.a.astype(float))

    def test_1_index(self):
        a_slice = self.a[:,0:1,0,0]
        j_slice = self.j[:,1:2,1,1]
        self.assert_looks_like(a_slice, """
        [[1]
         [9]]
        """)

        self.assert_looks_like(j_slice.array, """
        [[1]
         [9]]
        """)

    def test_repr_int(self):
        self.assert_looks_like(self.j, """
        2x2x2x2 Array{int64,4}
        [:, :, 0, 0] =
        1   9
        5   13
        [:, :, 0, 1] =
        2   10
        6   14
        [:, :, 1, 0] =
        3   11
        7   15
        [:, :, 1, 1] =
        4   12
        8   16
        """)

    def test_repr_float(self):
        self.assert_looks_like(self.j2, """
        2x2x2x2 Array{float64,4}
        [:, :, 0, 0] =
        1.0000     9.0000
        5.0000    13.0000
        [:, :, 0, 1] =
        2.0000    10.0000
        6.0000    14.0000
        [:, :, 1, 0] =
        3.0000    11.0000
        7.0000    15.0000
        [:, :, 1, 1] =
        4.0000    12.0000
        8.0000    16.0000
        """)
