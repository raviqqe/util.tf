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

  def test_dimension_indices(self):
    for i, T in enumerate(self.Ts):
      self.assertEqual(dimension_indices(T), [*range(i)])


if __name__ == "__main__":
  unittest.main()
