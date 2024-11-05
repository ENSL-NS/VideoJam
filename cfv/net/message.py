import time
import pickle
import numpy as np
from enum import Enum

from cfv.utils.memory_sizeof import total_size
from cfv.utils.videojam import Table

class Priority(Enum):
  INFO = 1
  REPLY = 1
  OFFLOAD = 1
  CONGESTION = 0
  DEFAULT = 5
  
  
class Trace:
  def __init__(self) -> None:
    self.net_propagtion_time = 0
    self.net_out_queue_time = 0
    self.proc_queue_time = 0
    self.count_trip = 0
    
  # NETWORK
  def network_send(self):
    self._network_send_at = time.time()
  def network_recv(self):
    self.count_trip += 1
    self.net_propagtion_time += time.time() - self._network_send_at
    self._network_send_at = None
    
  # NET QUEUE
  def net_queue_put(self):
    self._net_queued_at = time.time()
  def net_queue_get(self):
    self.net_out_queue_time += time.time() - self._net_queued_at
    self._net_queued_at = None
    
  # PROC QUEUE
  def proc_queue_put(self):
    self._proc_queued_at = time.time()
  def proc_queue_get(self):
    self.proc_queue_time += time.time() - self._proc_queued_at
    self._proc_queued_at = None
    

class Message(Trace):
  def __init__(self, at=None, data=None, arguments={}, priority=Priority.DEFAULT):
    '''Message wrapper.

    Args:
      data (optional): data to send. Defaults to None.
      arguments (dict, optional): metadata. Defaults to {}.
      priority (int, optional): sends or retrieves message in priority order (lowest first). Defaults to 5.
    '''
    Trace.__init__(self)
    self.at = at
    self.data = data
    self.arguments = arguments
    self.priority = priority # when priority queue is used only.

  def reset(self):
    self.at = None
    self.data = None
    self.arguments = {}

  def set_data(self, data):
    self.data = data

  def get_data(self):
    return self.data

  def set_argument(self, key, value):
    self.arguments[key] = value

  def set_arguments(self, arguments):
    for key in arguments:
      self.arguments[key] = arguments[key]

  def get_argument(self, key):
    if key not in self.arguments.keys():
      return None
    else:
      return self.arguments[key]

  def get_arguments(self):
    return self.arguments

  def marshal(self) -> bytes:
    obj = {}
    for property in dir(self):
      if property.startswith('__') or callable(getattr(self, property)):
        continue
      value = getattr(self, property)
      if isinstance(value, np.ndarray):
        obj[property] = value
      else:
        obj[property] = value
    pkl = pickle.dumps(obj)
    return pkl

  def unmarshal(self, pkl) -> None:
    obj = pickle.loads(pkl)
    properties = obj.keys()
    for property in properties:
      setattr(self, property, obj[property])
    
  def _memory(self, value) -> int:
    """memory used by the given value in bytes.

    Args:
      value (_type_): object for which to compute the occupancy in bytes.

    Returns:
      int: occupancy in bytes.
    """
    if isinstance(value, np.ndarray):
      return value.nbytes
    elif isinstance(value, Table):
      return value.nbytes
    elif isinstance(value, list):
      return sum([self._memory(item) for item in value])
    elif isinstance(value, dict):
      return sum([self._memory(item) for item in value.values()])
    elif isinstance(value, Message):
      return self._memory(value.arguments) + self._memory(value.data)
    else:
      return total_size(value)
    
  def get_memory_size(self) -> int:
    """Compute the memory that object uses. 

    Returns:
      int: memory used in bytes.
    """
    return self._memory(self.arguments) + self._memory(self.data)
    
  def total_size(self, verbose=False) -> int:
    '''
    Compute the approximate memory footprint of the object.
    '''
    def handler(obj: Message):
      yield obj.data
      yield obj.arguments
    return total_size(self, handlers={Message: handler}, verbose=verbose)
  
  def __lt__(self, other: object) -> bool:
    """Comparison between two instance of `Message`.

    Args:
      other (object): instance to compare with the current instance.

    Returns:
      bool: true if the current instance is older or has higher priority than the given instance.
    """
    if not isinstance(other, Message):
      raise NotImplementedError()
    if self.priority.value == other.priority.value:
      return self.at < other.at # is older than the given instance.
    return self.priority.value < other.priority.value # has higher priority than the given instance.
    
    