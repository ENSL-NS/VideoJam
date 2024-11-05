import numpy as np
import cv2

def fourier_forecast(x: np.ndarray, n_predict, n_harm=10) -> np.ndarray:
  """Fourier forecasting method for time series data.

  Args:
    x (np.ndarray): signal on which to apply fourier transform.
    n_predict (_type_): number of time step the future to forecast.
    n_harm (int, optional): number of harmonies in the spectrum to use for forecast. Defaults to 5.
      
  Source:
    Algorithm follow as described in "Fourier Analysis for Demand Forecasting in a Fashion Company".
    Base code was taken from https://gist.github.com/tartakynov/83f3cd8f44208a1856ce
  Returns:
    np.ndarray: return the forecasted data.
  """
  
  N = x.size
  t = np.arange(0, N)
  # 3. calculate a linear trend of the series (i.e., y = a * x + b).
  a, b, c = np.polyfit(t, x, 2)               # finding linear trend in x.
  x_detrended = x - (a * t**2 + b * t + c)    # removes a linear trend in x
  # 4. calculate the fft on the detrended data.
  x_fft = np.fft.fft(x_detrended)             # detrended x in frequency domain
  # 5. create the frequency spectrum (f0, f1, ..., fN).
  freq = np.fft.fftfreq(N)
  # 6. Except for f0, order the f1... fN components in a decreasing order of amplitude.
  args = np.zeros_like(freq, dtype=np.uint32)
  args = np.argsort(np.absolute(freq))
  # 7. Perform the inverse Fourier transform.
  t = np.arange(N, N + n_predict)
  forcasted_sig = np.zeros(t.size)
  for i in args[:1 + n_harm * 2]:
    ampli = np.absolute(x_fft[i]) / N   # amplitude
    phase = np.angle(x_fft[i])          # phase
    forcasted_sig += ampli * np.cos(2 * np.pi * freq[i] * t + phase)
      
  return forcasted_sig + (a * t**2 + b * t + c)


def simple_exp_smooth(data, extra_periods=1, alpha=0.4):  
	cols = data.size  # Historical period length    
	data = np.append(data, [np.nan] * extra_periods)  # Append np.nan into the demand array to cover future periods    
	forecast = np.full(cols+extra_periods,np.nan)  # Forecast array    
	forecast[1] = data[0]  # initialization of first forecast    
	# Create all the t+1 forecasts until end of historical period
	for t in range(2,cols+1):
		forecast[t] = alpha * data[t-1] + (1 - alpha) * forecast[t-1]  
	forecast[cols+1:] = forecast[t]  # Forecast for all extra periods
	return forecast
    

def regression_forecast(Y: np.ndarray, n_predict=10):
  result = np.empty(shape=(len(Y), n_predict))
  for i, y in enumerate(Y):
    x = np.arange(1, len(y) + 1)
    x_pred = x[-n_predict:] + n_predict # for next windows
    a, b, c = np.polyfit(x, y, deg=2)
    result[i, :] = a * x_pred**2 + b * x_pred + c
  return result


class Identity:
  def __init__(self, n_predict=10):
    self.n_predict = n_predict
  
  def __call__(self, data: np.ndarray) -> np.array:
    return data[:, -self.n_predict:]


class Model:
  def __init__(self, path_to_pb: str, gpu=False):
    self.net = cv2.dnn.readNetFromTensorflow(path_to_pb)
    if gpu:
      self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
      self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
      self.net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
      self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
  def __call__(self, data: np.ndarray) -> np.array:
    """Call for forecasting.
    The data should be fload32 otherwise it may not work.

		Args:
			data (np.ndarray): data used for forecasting.
				that data should be a shape (batch, input_width, 1), where input_width is fixed and equal to 50 (last 50 samples).

		Returns:
			np.array: return the 10 future values.
		"""
  
    # sample = np.float32(sample)
    blob = cv2.dnn.blobFromImages(data, scalefactor=1.0, mean=[0], swapRB=False)
    self.net.setInput(blob)
    return self.net.forward()