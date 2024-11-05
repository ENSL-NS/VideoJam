import time
import logging
from typing import List

import numpy as np

from cfv.net.message import Message
from cfv.functions.function import Function
from cfv.functions.functions_directory import get_class_by_name
from cfv.utils.videojam import Queue




class Wrapper(Function):
  def __init__(self):
    """Wrapper for multiprocessing.
    """
    
    Function.__init__(self)
    self.processing_rate = 0.0

    
  def configure(self, config: dict):
    if 'component_name' not in config.keys():
      raise ValueError('Missing component_name parameter')
    self.component_name = config['component_name']
    if 'config' not in config.keys():
      raise ValueError('Missing config parameter')
    if 'batch_size' not in config.keys():
      raise ValueError('Missing batch_size parameter')
    self.batch_size = config['batch_size']
    if 'match_batch_size' not in config.keys():
      raise ValueError('Missing match_batch_size parameter')
    maxsize = config['maxsize'] if ('maxsize' in config.keys()) else 0 # seconds
    self.queue = Queue(maxsize=maxsize, size=self.batch_size, match_batch_size=config['match_batch_size'], callback=self.on_maxsize)
    self.component = get_class_by_name(self.component_name)() # instantiate the actual class and configure it.
    self.component.configure(config=config['config'])
    logging.info('[wrapper({})]-wrapper initialized with batch size {}, match batch size {} and maxsize is {}s'.format(self.component_name, self.batch_size, config['match_batch_size'], maxsize))
    
  
  def on_maxsize(self, msg):
    """A callback method called whenever a message reaches its time out.

    Args:
      msg (_type_): any message to show for more information.
    """
    logging.warning('[wrapper(%s)]-%s'%(self.component_name, msg))
  
   
  async def push(self, id, msg: Message):
    """Add received message to the queue if not full discard otherwise.

    Args:
      msg (Message): message to add.
    """
    if 'wrapper' in msg.arguments.keys():
      data: List[Message] = msg.get_data()
      for item in data:
        item.net_propagtion_time = msg.net_propagtion_time
        item.net_out_queue_time = msg.net_out_queue_time
    else:
      data = [msg]
    for msg in data:
      self.queue.put_nowait(msg)

 
  async def run(self):
    '''

    :param port:
    :return:
    '''
    self.recovery_id = int(time.time())
    throughputs = np.full(shape=(10,), fill_value=np.nan, dtype=np.float32)
    max_batch = 0
    while True:
      batch: List[Message] = await self.queue.get_batch()
      N = len(batch)
      elapsed = time.time()
      result: List[Message] = await self.component.push(0, batch) # compute wrapped function.
      elapsed = time.time() - elapsed
      if N >= max_batch:
        max_batch = N
        throughputs[:-1] = throughputs[1:]
        throughputs[-1] = max_batch / elapsed
        self.processing_rate = max_batch / elapsed
        logging.info('[component(%s)]-Throughput: %.2f | Queue size: %i'%(self.component_name, self.processing_rate, self.queue.qsize()))
      elapsed = time.time() - elapsed
      await self.next(data=result) # send data to the next in the pipeline if exists.
  

  def get_async_tasks(self):
    '''

    :return:
    '''
    return Function.get_async_tasks(self) + [self.run()]