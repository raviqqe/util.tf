import unittest

import tensorflow as tf
import numpy as np

from . import *



class TestFunctions(unittest.TestCase):
  T0 = tf.constant(1, tf.int32)
  T1 = tf.constant([True, False, True], tf.bool)
  T2 = tf.constant([[1, 2], [3, 4.2]], tf.float32)
  Ts = [T0, T1, T2]

  def test_static_shape(self):
    self.assertEqual(static_shape(self.T0), [])
    self.assertEqual(static_shape(self.T1), [3])
    self.assertEqual(static_shape(self.T2), [2, 2])

  def test_static_rank(self):
    for i, T in enumerate(self.Ts):
      self.assertEqual(static_rank(T), i)

  def test_static_shapes(self):
    self.assertEqual(static_shapes(*self.Ts), [[], [3], [2, 2]])

  def test_dtypes(self):
    self.assertEqual(dtypes(*self.Ts), [tf.int32, tf.bool, tf.float32])

  def test_func_scope(self):
    @func_scope()
    def foo(): pass

    @func_scope("baz")
    def bar(): pass

  def test_on_device(self):
    @on_device("cpu:0")
    def foo(): pass

  def test_dimension_indices(self):
    for i, T in enumerate(self.Ts):
      self.assertEqual(dimension_indices(T), [*range(i)])

    self.assertEqual(dimension_indices(self.T2, 1), [1])


if __name__ == "__main__":
  unittest.main()
