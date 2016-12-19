import unittest

import tensorflow as tf
import numpy as np

from . import *



class TestFunctions(unittest.TestCase):
  T0 = tf.constant(1, tf.int32)
  T1 = tf.constant([True, False, True], tf.bool)
  T2 = tf.constant([[1, 2], [3, 4.2]], tf.float32)
  Ts = [T0, T1, T2]
  FLOAT_TYPES = [tf.float32, tf.float64]

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

  def test_dtype_min(self):
    for float_type in self.FLOAT_TYPES:
      min = dtype_min(float_type)
      self._assert_bool_tensor(min > -np.inf)
      self._assert_bool_tensor(min < -1e8)

  def test_dtype_epsilon(self):
    for float_type in self.FLOAT_TYPES:
      epsilon = dtype_epsilon(float_type)
      self._assert_bool_tensor(epsilon > 0)
      self._assert_bool_tensor(epsilon < 1e-6)

  def test_flatten(self):
    for T in self.Ts[1:]:
      self.assertEqual(static_rank(flatten(T)), 1)
      self.assertEqual(flatten(T).get_shape().num_elements(),
                       T.get_shape().num_elements())

  def test_rename(self):
    name = "foo"
    self.assertEqual(rename(tf.constant(42), name).name.split(':')[0], name)

  def _assert_bool_tensor(self, tensor):
    self.assertTrue(run(tensor))


def run(tensor):
  with tf.Session() as session:
    return session.run(tensor)


if __name__ == "__main__":
  unittest.main()
