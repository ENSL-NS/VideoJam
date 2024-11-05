import bisect
import time
import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Callable
from collections import defaultdict
from enum import Enum

from cfv.utils.memory_sizeof import total_size




class Signal(Enum):
  RESET = 0
  INFO = 1
  REPLY = 2
  CONGESTION = 3
  OFFLOAD = 4
  
  
class TimerQueue:
  def __init__(self, timeout: float=None, callback: Callable[[str], None]=None):
    """Priority queue with timeout.

    Args:
      timeout (float, optional): maximum delay before considering a item out-of-date. Defaults to None.
      callback (_type_, optional): callable function called when item is out-of-date. Defaults to None.
    """
    self._priority_queue = asyncio.PriorityQueue()
    self._timeout = timeout
    self._callback = callback if callback else logging.debug
  
  def qsize(self) -> int:
    return self._priority_queue.qsize()
    
  async def put(self, item):
    if (time.time() - item.at) > self._timeout:
      self._callback('Item has reached its timeout')
    else:
      await self._priority_queue.put(item)

  def put_nowait(self, item):
    if (time.time() - item.at) > self._timeout:
      self._callback('Item has reached its timeout')
    else:
      self._priority_queue.put_nowait(item)

  async def get(self) -> Any:
    """Retrieve data at given position by wait if not available. pos=-1 means the last item. 

    Returns:
      Any: item to retrieve.
    """
    while True:
      item = await self._priority_queue.get()
      if (time.time() - item.at) > self._timeout:
        self._callback('Item has reached its timeout')
        continue
      else:
        return item

  def get_nowait(self) -> Any:
    """Retrieve and return the oldest data from the queue without blocking. 

    Returns:
      Any: item retrieved.
    """
    while True: # we exit either when queue is empty or when a item is found.
      item = self._priority_queue.get_nowait()
      if (time.time() - item.at) > self._timeout:
        self._callback('Item has reached its timeout')
        continue
      else:
        return item
  
  def __repr__(self) -> str:
    return """TimerQueue('timeout': {}, 'qsize': {},)""".format(self._timeout, self._priority_queue.qsize(),)


class IQueue:
  def __init__(self, size: int, match_batch_size=False) -> None:
    self._size = size
    self._match_batch_size = match_batch_size
    self.get_batch = self._get_batch if match_batch_size else self._get_minimum_batch
    
  def qsize(self) -> int:
    raise NotImplemented()
    
  async def put(self, item):
    raise NotImplemented()

  def put_nowait(self, item):
    raise NotImplemented()

  async def get(self) -> Any:
    raise NotImplemented()

  def get_nowait(self) -> Any:
    raise NotImplemented()
  
  async def _get_batch(self) -> List[Any]:
    raise NotImplemented()

  async def _get_minimum_batch(self) -> List[Any]:
    raise NotImplemented()
  
  async def _get_batch(self) -> List[Any]:
    batch =  []
    # as long as the batch size has not been reached and there is an element in the queue.
    while len(batch) < self._size:
      # make sure to wait for the first message, otherwise the loop may crash the execution.
      item = await self.get()
      batch.append(item)
    return batch

  async def _get_minimum_batch(self) -> List[Any]:
    # make sure to wait for the first message, otherwise the loop may crash the execution.
    item = await self.get()
    batch = [item] # append the first element.
    # as long as the batch size has not been reached and there is an element in the queue.
    try:
      while len(batch) < self._size:
        item = self.get_nowait()
        batch.append(item)
    except asyncio.QueueEmpty: pass
    return batch
  
  def __repr__(self) -> str:
    return """Queue('size': {}, 'match_batch_size': {}, 'qsize': {},)""".format(self._size, self._match_batch_size, self.qsize(),)


class Queue(IQueue):
  def __init__(self, maxsize: int=0, size: int=1, match_batch_size=False, callback: Callable[[str], None]=None) -> None:
    """Priority queue with maxsize.

    Args:
      maxsize (float, optional): maximum delay before considering a item out-of-date. Defaults to None.
      callback (_type_, optional): callable function called when item is out-of-date. Defaults to None.
    """
    super().__init__(size=size, match_batch_size=match_batch_size)
    self._priority_queue = asyncio.PriorityQueue(maxsize=maxsize)
    self._maxsize = maxsize
    self._callback = callback if callback else logging.debug
  
  def qsize(self) -> int:
    return self._priority_queue.qsize()
    
  async def put(self, item):
    item.proc_queue_put()
    await self._priority_queue.put(item)

  def put_nowait(self, item):
    if self._maxsize > 0 and self.qsize() >= self._maxsize:
      self._callback('Queue is full %i'%self.qsize())
    else:
      item.proc_queue_put()
      self._priority_queue.put_nowait(item)

  async def get(self) -> Any:
    """Retrieve data at given position by wait if not available. pos=-1 means the last item. 

    Returns:
      Any: item to retrieve.
    """
    item = await self._priority_queue.get()
    item.proc_queue_get()
    return item

  def get_nowait(self) -> Any:
    """Retrieve and return the oldest data from the queue without blocking. 

    Returns:
      Any: item retrieved.
    """
    item = self._priority_queue.get_nowait()
    item.proc_queue_get()
    return item



class JamQueue(IQueue):
  def __init__(self, maxsize: int = 0, size: int = 1, match_batch_size=False, callback: Callable[[str], None] = None) -> None:
    """Priority queue with maxsize.

    Args:
      maxsize (float, optional): maximum delay before considering a item out-of-date. Defaults to None.
      callback (_type_, optional): callable function called when item is out-of-date. Defaults to None.
    """
    super().__init__(size, match_batch_size)
    self._added = [ [], [], ] # first is normal queue and second is queue for data from neighbors.
    self._maxsize = maxsize
    self.mimiqueue = asyncio.Queue()
    self._callback = callback if callback else logging.debug
  
  def qsize(self) -> int:
    return self.mimiqueue.qsize()
    
  async def put(self, item, from_offloading=False):
    item.proc_queue_put()
    await self.mimiqueue.put(None)
    i = 1 if from_offloading else 0 # anything from neighbors get to the second queue first.
    bisect.insort(self._added[i], item)

  def put_nowait(self, item, from_offloading=False):
    if self._maxsize > 0 and self.qsize() >= self._maxsize:
      self._callback('Queue is full %i'%self.qsize())
      return
    item.proc_queue_put()
    self.mimiqueue.put_nowait(None)
    i = 1 if from_offloading else 0 # anything from neighbors get to the second queue first.
    bisect.insort(self._added[i], item)

  async def get(self, for_offloading=False) -> Any:
    """Retrieve data at given position by wait if not available. pos=-1 means the last item. 

    Returns:
      Any: item to retrieve.
    """
    await self.mimiqueue.get()
    if for_offloading:
      i = 0 if self._added[0] else 1 # the first queue has priority when offloading
      pos = -1 # newest first
    else:
      i = 1 if self._added[1] else 0 # the second queue (data from offload) has priority when processing
      pos = np.random.choice([0, -1], p=[.2, .8])
    item = self._added[i].pop(pos)
    item.proc_queue_get()
    return item

  def get_nowait(self, for_offloading=False) -> Any:
    """Retrieve data at given position by wait if not available. pos=-1 means the last item. 

    Returns:
      Any: item to retrieve.
    """
    self.mimiqueue.get_nowait()
    if for_offloading:
      i = 0 if self._added[0] else 1 # the first queue has priority when offloading
      pos = -1 # newest first
    else:
      i = 1 if self._added[1] else 0 # the second queue (data from offload) has priority when processing
      pos = np.random.choice([0, -1], p=[.2, .8])
    item = self._added[i].pop(pos)
    item.proc_queue_get()
    return item
  

  def __repr__(self) -> str:
    return """JamQueue('maxsize': {}, 'qsize': {},)""".format(self._maxsize, self.qsize(),)



class Row:
  def __init__(self,
    id="",
    at=0.0,
    receive_at=0.0,
    num_replicas=0,
    load=0,
    incoming_rate:float=None,
    processing_rate=0.0,
    policy: dict={},
    maximum_round=0,
    ) -> None:
    """_summary_

    Args:
      id (str, optional): _description_. Defaults to "".
      num_replicas (int, optional): _description_. Defaults to 0.
      load (int, optional): _description_. Defaults to 0.
      incoming_rate (List[int], optional): from monitor per w. Defaults to 0.
      processing_rate (int, optional): _description_. Defaults to 0.
      policy (tuple, optional): result from computation of offload by the dispatcher. Defaults to [].
    """
    
    self.at = at
    self.id = id                  # neighbor identification.
    self.receive_at = receive_at  # time when this information has been received. 
    self.load = load              # actual/forecasted load of the neighbor for the next or current cycle.
    self.num_replicas = num_replicas        # number of workers that are with the neighbor.
    self.incoming_rate = np.array([]) if incoming_rate is None else incoming_rate # get from the monitor.
    self.processing_rate = processing_rate  # get from the monitor.
    self.policy = policy  # (from, value)
    # this is the maximum time limit for considering information that is still valid.
    # beyond that delay, info signal should be sent.
    self.maximum_round = maximum_round
    self.is_available = True
    self.policy_key = None
    self.maximum_processed_per_window = None
    
    self.nbytes: int = sum([
      self._nbytes(self.at),
      self._nbytes(self.id),
      self._nbytes(self.receive_at),
      self._nbytes(self.load),
      self._nbytes(self.num_replicas),
      self._nbytes(self.incoming_rate),
      self._nbytes(self.processing_rate),
      self._nbytes(self.policy),
      self._nbytes(self.maximum_round),
      self._nbytes(self.is_available),
      self._nbytes(self.policy_key),
      self._nbytes(self.maximum_processed_per_window),
      ])


  def update_incoming_rate(self, value):
    """Update last window by freeing the oldest.

    Args:
      value (np.ndarray): array of all intervals of the last window.
    """
    self.incoming_rate[:-1] = self.incoming_rate[1:]
    self.incoming_rate[-1] = value


  def is_overloaded(self, w: float, threshold:float=.2) -> bool:
    _in = self.load + self.get_input_load(w=w)
    _out = self.get_maximum_processed_per_window(w=w)
    return ((_in - _out) / _out > threshold) if (_out > 0) else (_in > 0)


  def get_offload(self, id) -> int:
    if id in self.policy.keys():
      return self.policy[id]
    return 0


  def get_offloaded(self) -> int:
    return np.sum(list(self.policy.values()))


  def get_input_load(self, w: float) -> float:
    """Retrun the load received during the last monitoring window.

    Args:
      w (float): elapsed time interval.
    """
    return np.sum(self.get_latest_incoming_rate() * w)
  
  
  def get_maximum_processed_per_window(self, w: float) -> float:
    """Compute the maximum frame processed per window.

    Args:
        w (float): elapsed time interval.
    Returns:
        float: the maximum number of frames processed in the window.
    """
    _round = self.incoming_rate.shape[-1] # (HISTORY, WINDOW)
    return np.sum(self.processing_rate * _round * w)
  
  
  def set_latest_incoming_rate(self, value: np.ndarray):
    """Update last window by the given value.

    Args:
      value (np.ndarray): array of all intervals of the last window.
    """
    self.incoming_rate[-1] = value
  
  
  def get_latest_incoming_rate(self) -> np.ndarray:
    """The incoming rate is in shape of (HISTORY, WINDOW) where [0, :] is the oldest one and [-1, :] the latest.
    """
    return self.incoming_rate[-1]
  

  def __repr__(self) -> str:
    return """row(
      'at': {},
      'id': {},
      'receive_at': {},
      'load': {},
      'num_replicas': {},
      'incoming_rate': [{}... {}],
      'processing_rate': {},
      'policy': {},
      'maximum_round': {},
      'is_available': {},
    )""".format(
      self.at,
      self.id,
      self.receive_at,
      self.load,
      self.num_replicas,
      self.incoming_rate[0], self.get_latest_incoming_rate(),
      self.processing_rate,
      self.policy,
      self.maximum_round,
      self.is_available,
    )


  def is_unknwon(self) -> bool:
    return (self.maximum_round <= 0) or (self.at <= 0) or (self.processing_rate <= 0)

  def _nbytes(self, value):
    if isinstance(value, np.ndarray):
      return value.nbytes
    elif isinstance(value, list):
      return sum([self._nbytes(item) for item in value])
    elif isinstance(value, dict):
      return sum([self._nbytes(item) for item in value.values()])
    return total_size(value)
    


class Table:
  def __init__(self) -> None:
    self.table: Dict[int, Row] = defaultdict(Row)
    self.nbytes = 0
    
    
  def add(self, key: any, row: Row):
    self.table[key] = row
    self.nbytes = sum([row.nbytes for row in self.rows()])

  def get(self, key) -> Row:
    if key in self.table.keys():
      return self.table[key]
    return None
  
  def compute_workload(self, row: Row, w: float) -> int:
    """Estimate the worload of the given row by tacking into account the following:
      - the previous load
      - plus the prior forecasting
      - minus the processing rate
      - and more or less the offload.

    Args:
    """
    # load = qsize + sum(input_rate - proc_rate) * w + offload
    load = row.load # current load
    load += np.sum(row.get_latest_incoming_rate() * w) # near futur incoming load.
    load -= row.get_maximum_processed_per_window(w) # maximum workload process per window (i.e., into interval [w1, w2, ..., wn]).
    load -= row.get_offloaded() # workload offloaded to neighbors.
    load += np.sum([
      neighbor.policy[row.id]
      for neighbor in self.table.values()
      if row.id != neighbor.id and row.id in neighbor.policy.keys()
      ]) # offload from a neighbors.
    return max([0, load])
  
  def get_maximum_discharge_workload(self, src, dest) -> int:
    """Maximum amount of workload offloadable to neighbor.

    Args:
      src (_type_): the sender
      dest (_type_): the receiver.
    Returns:
      int: maximum workload to offload.
    """
    if dest in self.table[src].policy.keys():
      return self.table[src].policy[dest]
    self.table[src].policy[dest] = 0
    return 0
  
  def update_policy(self, key, policy: dict) -> None:
    self.table[key].policy.update(policy) # update to new values.
  
  def reset_policy(self, key) -> None:
    if key in self.table.keys():
      self.table[key].policy = dict.fromkeys(self.table[key].policy, 0) # reset all existing keys to 0.
    return None
    
  def set_availability(self, key, value) -> None:
    self.table[key].is_available = value
    
  def remove(self, key) -> Row:
    if key in self.table.keys():
      row =  self.table.pop(key)
      self.nbytes = sum([row.nbytes for row in self.rows()])
      return row
    raise KeyError("Key {} not found".format(key))
    
  def update(self, row: Row) -> bool:
    # the current row is the latest
    if self.table[row.id].at < row.at:
      self.table[row.id] = row
      elapsed = self.table[row.id].receive_at - self.table[row.id].at
      self.nbytes = sum([row.nbytes for row in self.rows()])
      logging.debug("[Scheduler] - Propagation delay {:.2f}s".format(elapsed))
      return True
    return False
  
  def keys(self) -> List[int]:
    """Retrieve keys

    Returns:
      List[int]: _description_
    """
    return [k for k, v in self.table.items() if v.is_available]

  def rows(self) -> List[Row]:
    """Retrieve rows

    Returns:
      List[Row]: _description_
    """
    return [row for row in self.table.values() if row.is_available]
  
  def __repr__(self) -> str:
    return "Table\n" + "".join(["{}\n".format(row) for row in self.table.values()])

  def __iter__(self):
    """Iter only over the available neighbors.

    Yields:
        Row: row that holds neighbor's information.
    """
    for key, row in self.table.items():
      if not row.is_available:
        continue
      yield key, row
      
  def __len__(self):
    return len(self.table)