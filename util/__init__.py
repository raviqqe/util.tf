import functools

import numpy
import tensorflow as tf



def static_shape(tensor):
  """Get a static shape of a Tensor.

  Args:
    tensor: Tensor object.

  Return:
    List of int.
  """
  return tf.convert_to_tensor(tensor).get_shape().as_list()


def static_shapes(*tensors):
  """Get static shapes of Tensors.

  Args:
    tensors: Tensor objects.

  Returns:
    List of list of int.
  """
  return [static_shape(tensor) for tensor in tensors]


def static_rank(tensor):
  """Get a static rank of a Tensor.

  Args:
    tensor: Tensor object.

  Returns:
    int.
  """
  return len(static_shape(tf.convert_to_tensor(tensor)))


def dtypes(*tensors):
  """Get data types of Tensors.

  Args:
    tensors: Tensor objects.

  Returns:
    DTypes of the Tensor objects.
  """
  return [tensor.dtype for tensor in tensors]


def func_scope(name=None, initializer=None):
  """A function decorator to wrap a function in a variable scope.

  Args:
    name: Name of a variable scope, defaults to a name of a wrapped function.
    initializer: Initializer for a variable scope.

  Returns:
    A decorated function.
  """
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      with tf.variable_scope(name or func.__name__, initializer=initializer):
        return func(*args, **kwargs)
    return wrapper

  return decorator


def on_device(device_name):
  """A function decorator to run everything in a function on a device.

  Args:
    device_name: Device name where every operations and variables in a function
                 are run.

  Returns:
    A decorated function.
  """
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      with tf.device(device_name):
        return func(*args, **kwargs)
    return wrapper
  return decorator


def dimension_indices(tensor, start=0):
  """Get dimension indices of a Tensor object.

  An example below is hopefully comprehensive.

  >>> dimension_indices(tf.constant([[1, 2], [3, 4]]))
  [0, 1]

  Args:
    tensor: Tensor object.
    start: The first index of output indices.

  Returns:
    List of dimension indices. For example, when a Tensor `x` is of rank 5,
    `dimension_indices(x, 2) == [2, 3, 4]`.
  """
  return [*range(static_rank(tensor))][start:]


@func_scope()
def dtype_min(dtype):
  """Get a minimum for a TensorFlow data type.

  Args:
    dtype: TensorFlow data type.

  Returns:
    A scalar minimum of the data type.
  """
  return tf.constant(numpy.finfo(dtype.as_numpy_dtype).min)


@func_scope()
def dtype_epsilon(dtype):
  """Get a machine epsilon for a TensorFlow data type.

  Args:
    dtype: TensorFlow data type.

  Returns:
    A scalar machine epsilon of the data type.
  """
  return tf.constant(_numpy_epsilon(dtype.as_numpy_dtype))


def _numpy_epsilon(dtype):
  return numpy.finfo(dtype).eps


def flatten(tensor):
  """Flatten a multi-dimensional Tensor object.

  Args:
    tensor: Tensor object.

  Returns:
    Flattened Tensor object of a vector.
  """
  return tf.reshape(tensor, [-1])


def rename(tensor, name):
  """Rename a Tensor.

  Args:
    tensor: Tensor object.
    name: New name of the Tensor object.

  Returns:
    Renamed Tensor object.
  """
  return tf.identity(tensor, name)
