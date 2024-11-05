import numpy as np
import tensorflow as tf


class ModelLoader:
  """Tensorflow Convolution Neural Network Forecasting.
  """
  def __init__(self, path_to_pb: str):
    """Load tensorflow from saved model.

    Args:
      path_to_pb (str): path where to find the saved model.
    """
    if not tf.config.list_logical_devices('GPU'):
      raise EnvironmentError('No GPU is available')
    
    self.gpu = tf.config.list_logical_devices('GPU')[0]
    self.net = tf.keras.models.load_model(path_to_pb)
    self.W, self.H, self.C = (224, 224, 3)
    
  def __call__(self, frames: np.ndarray) -> np.array:
    """Call for forecasting.
    The frames should be fload32 otherwise it may not work.

		Args:
			frames (np.ndarray): frames used for forecasting.
				that frames should be a shape (batch, input_width, 1), where input_width is fixed and equal to 50 (last 50 samples).

		Returns:
			np.array: return the 10 future values.
		"""
    # for the computation to be done on gpu.
    with tf.device(self.gpu.name):
      normalized = tf.image.resize(frames, (self.W, self.H))
      return self.net.predict(normalized)