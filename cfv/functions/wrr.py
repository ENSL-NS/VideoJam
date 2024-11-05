import time
import asyncio
import logging
import numpy as np
from typing import List, Dict

from cfv.functions.function import Function
from cfv.net.message import Message
from cfv.net.local_port import LocalOutPort
from cfv.net.remote_port import RemoteOutPort
from cfv.utils.videojam import Row, Signal, Table, IQueue, Queue
from cfv.utils.schedulers import WRRScheduling, Server
from cfv.functions.functions_directory import get_class_by_name




class StaticWRR(Function):
  def __init__(self):
    """Decentralized Load Balancing.
    Load balancer based on queue size, arrival rate and the processing rate for distributing its traffic to its neighbors.
    """
    
    Function.__init__(self)
    self.id = None # scheduler's id (from incoming[0].id)
    self.w = 1 # (seconds)
    self.round = 10 # how many window to gether for a round for forecasting.
    self.queue: IQueue = None # outgoing tasks for consumers.
    # table that contains all information about scheduler row of performance.
    self.table: Table = Table()
    # schedulers which are assumed to be remote.
    self.neighbors: Dict[str, RemoteOutPort] = {}
    # This variable is used to prevent two unloading windows from crossing. Only the one with right key will keep offloading.
    self.scheduling = WRRScheduling()

    
  def configure(self, config):
    """There is no configuration
    """

    if 'w' not in config.keys():
      raise ValueError('Missing w parameter')
    self.w = config['w']
    if 'round' not in config.keys():
      raise ValueError('Missing round parameter')
    self.round = config['round']
    if 'component_name' not in config.keys():
      raise ValueError('Missing component_name parameter')
    self.component_name = config['component_name']
    if 'config' not in config.keys():
      raise ValueError('Missing config parameter')
    self.batch_size = config['batch_size'] if ('batch_size' in config.keys()) else 1
    maxsize = config['maxsize'] if ('maxsize' in config.keys()) else 0 # seconds
    self.queue = Queue(maxsize=maxsize, size=self.batch_size, match_batch_size=config['match_batch_size'], callback=self.on_maxsize)
    self.component = get_class_by_name(self.component_name)() # instantiate the actual class and configure it.
    self.component.configure(config=config['config'])
    self.warmingup = True
    logging.info('[Scheduler]-window: {}s - round: {} and maxsize: {}'.format(self.w, self.round, maxsize))


  async def compute_policy(self):
    """This is a method of determining the appropriate load balance based on the current state of the table. No predictions are made, but simply the round-robin weighted distribution is calculated and adjusted.

    """
    # while True:
    await asyncio.sleep(self.round * self.w)
    processing_rates = []
    neighbor_ids = []
    for id, row in self.table:
      if not row.is_available:
        continue
      processing_rates.append(row.processing_rate)
      neighbor_ids.append(id)
    processing_rates = np.array(processing_rates)
    weights = np.ceil(processing_rates / (processing_rates.min() + 1e-12)).astype(np.int32) + 1
    logging.info('\u27F3 {} Proc: {} | Weights: {}'.format(self.component_name, processing_rates, weights))
    for id, weight in zip(neighbor_ids, weights):
      logging.info('Updating server server=(k:{}, weight: {})'.format(id, weight))
      self.scheduling.update(key=id, weight=weight)
       

  async def push(self, id, msg: Message):
    """Receiving data from previous steps or neighbors.

    Args:
      id (_type_): _description_
      msg (Message): received message.
    """
    if 'type' in msg.arguments.keys():
      if msg.arguments['type'] == Signal.RESET:
        self.warmingup = True
        self.iii = 0
        for key, _ in self.table:
          self.scheduling.update(key=key, weight=1)
        logging.warning('\u26A0 A resset policy was requested'.format(self.component_name))
      elif self.table.update(row=msg.arguments['row']):
        if msg.arguments['type'] == Signal.INFO:
          notif = Message(at=time.time(), arguments={ 'type': Signal.REPLY, 'row': self.table.get(self.id), })
          await self.offload2neighbor(msg=notif, id=msg.arguments['row'].id)
          asyncio.create_task(self.compute_policy())
          # logging.warning('\u26D4 warning up will start again.')
      if msg.get_data() is not None:
        msg.arguments.pop('type', None) # delete key if exists
        msg.arguments.pop('row', None)  # delete key if exists
        self.add2queue(msg)
    else:
      # call the next neighbor to handle the incoming traffic.
      next = self.scheduling.get_next()
      if next.key == self.id:
        self.add2queue(msg)
      else:
        msg.arguments['type'] = Signal.OFFLOAD
        msg.arguments['row'] = self.table.get(self.id)
        await self.offload2neighbor(msg=msg, id=next.key) # redirect to the outport for it to be sent to the selected neighbor.
      

  def add2queue(self, msg: Message):
    """Add message to the queue if queue not full yet. Drop it otherwise.

    Args:
      msg (Message): message to add into the queue.
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

  
  async def offload2neighbor(self, msg: Message, id: int):
    """Send message to remote neighbor(s).
    """
    for _ in range(5):
      # update availability for all unreachable neighbors.
      if not self.neighbors[id].is_ready():
        self.table.set_availability(id, False)
        logging.warning('[dispatcher]-removing neighbor[{}] from local table'.format(id))
        await asyncio.sleep(.1)
      else:
        self.table.set_availability(id, True) # set availability back.
        await self.neighbors[id].push(msg)
        return
        
  
  async def consumer(self):
    """A consumer from the queue of tasks.

    Args:
      outport (LocalOutPort): the actual operator.
    """
    # here the scheduler push data as fast as the consumer (next instance)
    for key, _ in self.table:
      if key == self.id: continue
      await self.offload2neighbor(msg=Message(at=time.time(), arguments={ 'type': Signal.RESET, }), id=key)
    max_batch = 0
    throughputs = np.full(shape=(self.round,), fill_value=np.nan, dtype=np.float32)
    self.iii = 0
    while True:
      batch: List[Message] = await self.queue.get_batch()
      N = len(batch)
      for msg in batch:
        msg.arguments.pop('type', None) # delete key if exists
        msg.arguments.pop('row', None)# delete key if exists
      logging.debug('[wrr(%s)]-read %i messages from queue'%(self.component_name, len(batch)))
      elapsed = time.time()
      result = await self.component.push(0, batch)
      elapsed = time.time() - elapsed
      if N >= max_batch:
        max_batch = N
        throughputs[:-1] = throughputs[1:]
        throughputs[-1] = max_batch / elapsed
        self.table.get(self.id).processing_rate = np.nanmedian(throughputs)
        self.table.get(self.id).at = time.time()
        logging.info('[wrr(%s)]-Throughput: %.2f | Queue size: %i'%(self.component_name, self.table.get(self.id).processing_rate, self.queue.qsize()))
        if self.iii > 10 and self.warmingup:
          self.warmingup = False
          asyncio.create_task(self.compute_policy())
        elif self.iii < 2:
          for key, _ in self.table:
            if key == self.id: continue
            notif = Message(at=time.time(), arguments={ 'type': Signal.INFO, 'row': self.table.get(self.id), })
            await self.offload2neighbor(msg=notif, id=key)
        self.iii += 1
      await self.next(data=result) # send data to the next in the pipeline if exists.


  async def run(self):
    """Settle all the stuff such as the monitor, the consumers and offloaders and so on.
    Some assumption have been made:
      - all local outport are considered being the consumers;
      - all remote outport are considered to be neighbors;
      - each scheduler should have a unique value (outport.id) that match with each other on the configuration file.
    Raises:
      ValueError: if id for outgoing port is missed.
      ValueError: if an unknown OutPort is given.
    """
    self.recovery_id = int(time.time())
    self.id = self.incoming[0].id
    if self.id is None:
      raise Exception('Please give an id for scheduler port')
    for outport in self.outgoing:
      if isinstance(outport, LocalOutPort): pass # nothing to do
      elif isinstance(outport, RemoteOutPort):
        if outport.id is None or outport.id == 0: # the next port
          pass
        else: # the neighbours
          self.neighbors[outport.id] = outport
          self.scheduling.schedule(Server(key=outport.id, value=outport.push), 1)
          # init all neighbors with weight to 0.
          row = Row(
            id=outport.id,
            processing_rate=-1,
            )
          self.table.add(key=outport.id, row=row)
      else:
        raise ValueError('Unknown outport type {}'.format(type(outport)))
    self.outgoing = [outport for outport in self.outgoing if outport.id == 0]
    self.scheduling.schedule(Server(key=self.id, value=None), weight=1)
    # init table row.
    row = Row(
      id=self.id,
      processing_rate=1,
      )
    self.table.add(key=self.id, row=row)
    
    # add the component
    tasks = [self.consumer()]
    await asyncio.gather(*[asyncio.get_event_loop().create_task(task) for task in tasks])


  def get_async_tasks(self):
    '''

    :return:
    '''
    return Function.get_async_tasks(self) + [self.run()]