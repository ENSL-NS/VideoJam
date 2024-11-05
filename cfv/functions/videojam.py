import glob
import time
import asyncio
import logging
import numpy as np
from typing import List, Dict

from cfv.functions.function import Function
from cfv.net.message import Message, Priority
from cfv.net.local_port import LocalOutPort
from cfv.net.remote_port import RemoteOutPort
from cfv.utils.videojam import IQueue, Row, Signal, Table, JamQueue
from cfv.utils.short_term_forecast_opencv import Model, Identity
from cfv.functions.functions_directory import get_class_by_name




class VideoJam(Function):
  def __init__(self):
    """Decentralized Load Balancing.
    Load balancer based on queue size, arrival rate and the processing rate for distributing its traffic to neighbors.
    """
    
    Function.__init__(self)
    self.id = None # scheduler's id (from incoming[0].id)
    self.w = 1 # (seconds)
    self.round = 10 # how many window to gether for a round for forecasting.
    self.threshold = .2
    self.num_received = 0
    self.maximum_round = 3 # default 4
    self.history_length = 5
    self.queue: IQueue = None # outgoing tasks for consumers.
    # table that contains all information about scheduler row of performance.
    self.table: Table = Table()
    # schedulers which are assumed to be remote.
    self.neighbors: Dict[str, RemoteOutPort] = {}
    # tasks+signals for neighbors (should be a priority queue instead)
    self.neighbor_queues: Dict[int, asyncio.PriorityQueue] = {}
    # This variable is used to prevent two unloading windows from crossing. Only the one with right key will keep offloading.
    self.offloading_key = None
    
    
  def configure(self, config):
    """There is no configuration
    """

    if 'w' not in config.keys():
      raise ValueError('Missing `w` parameter')
    self.w = config['w']
    if 'round' not in config.keys():
      raise ValueError('Missing `round` parameter')
    self.round = config['round']
    self.history_length = 50 // self.round
    paths = []
    if 'forecasting_model' in config.keys():
      paths = glob.glob('{}/*.pb'.format(config['forecasting_model']))
    if paths:
      self.full_pb_path = paths.pop()
      self.model = Model(self.full_pb_path)
    else:
      self.full_pb_path = 'indentity'
      self.model = Identity()
    if 'component_name' not in config.keys():
      raise ValueError('Missing component_name parameter')
    self.component_name = config['component_name']
    if 'config' not in config.keys():
      raise ValueError('Missing config parameter')
    self.processing_rate = config['processing_rate'] if ('processing_rate' in config.keys()) else 0
    self.batch_size = config['batch_size'] if ('batch_size' in config.keys()) else 1
    maxsize = config['maxsize'] if ('maxsize' in config.keys()) else 0 # seconds
    self.queue = JamQueue(maxsize=maxsize, size=self.batch_size, match_batch_size=config['match_batch_size'], callback=self.on_maxsize)
    self.component = get_class_by_name(self.component_name)() # instantiate the actual class and configure it.
    self.component.configure(config=config['config'])
    logging.info('[Scheduler]-window: {}s - round: {} - model: {} and maxsize: {}s'.format(self.w, self.round, self.full_pb_path, maxsize))


  async def update_table(self, table: Table) -> bool:
    """Responsible of receiving signals from neighbor schedulers.

    Args:
      msg (Message): signal that should be treated

    Returns:
      bool: True if changes have been observed.
    """
    if not table:
      return False
    result = False
    # discard any update from itself.
    rows = [row for id, row in table if id != self.id]
    result = any([self.table.update(row) for row in rows])
    if not self.table.get(self.id).processing_rate or (self.num_received < self.batch_size):
      self.table.get(self.id).processing_rate = sum([row.processing_rate for row in rows]) / len(rows)
      
    return result
  
  
  async def monitor(self):
    """It's responsability is to monitor the performance of the workers or replicas in term
    of processing rate, queue size, etc.
    It can also trigger the dispatcher if it happens to detect an overload.
    """
    
    logging.info('[monitor]-monitoring has been initiated at {} for a window of {}seconds'.format(time.time(), self.w))
    input_window = np.empty(shape=(self.round), dtype=np.float32)
    while True:
      # initializations
      w = 0
      input_window[:] = 0
      while w < self.round:
        input_window[w] = self.num_received
        await asyncio.sleep(self.w) # wait for collecting data.
        input_window[w] = self.num_received - input_window[w]
        # self.table.get(self.id).incoming_rate[-1, :self.round-w] = self.table.get(self.id).incoming_rate[-1, w:]
        # self.table.get(self.id).incoming_rate[-1, -w:] = input_window[:w]
        # if np.all(self.table.get(self.id).get_latest_incoming_rate() >= 0):
        #   error = (input_window[:(w+1)]-self.table.get(self.id).get_latest_incoming_rate()[:(w+1)]) / (input_window[:(w+1)] + 1e-10)
        #   logging.debug('[monitor]-load variation error is {}'.format(error))
        #   if np.sum(error > .5) >= self.round // 2:
        #     # call the dispatch (not the dispatcher) method for adjusting the scheduling policy.
        #     await self.compute_policy(restart_offloading=True)
        logging.debug('[monitor]-w[{}] -> input_rate={}, qsize={}'.format(w+1, input_window[w], self.queue.qsize()))
        w += 1
      self.table.get(self.id).set_latest_incoming_rate(input_window / self.w)
      self.table.get(self.id).at = time.time()
      asyncio.create_task(self.dispatcher()) # awake the dispatcher.
      logging.debug('[monitor]-a new round has been completed.')
    

  async def dispatcher(self):
    """Dispatcher.
    The dispatcher is responsible for using the data from the monitor and its neighbors to first make a forecast and then calculate the scheduling policy.
    """
    
    logging.info('[dispatcher]-dispatcher has been executed.')
    # update availability for all unreachable neighbors.
    for id, outport in self.neighbors.items():
      if not outport.is_ready():
        self.table.get(self.id).policy[id] = 0
        self.table.set_availability(id, False)
        logging.warning('[dispatcher]-removing neighbor[{}] from local table'.format(id))
      else:
        self.table.set_availability(id, True) # set availability back.
        
    unknow_neighbors = [id for id, row in self.table if id != self.id and row.is_unknwon()]
    for key in unknow_neighbors:
      msg = Message(
        at=time.time(), 
        arguments={ 'id': self.id, 'type': Signal.INFO, 'table': self.table, 'send_at': time.time() },
        priority=Priority.INFO,
        )
      await self.send_signal(msg=msg, keys=[key])
      
    # dispatch
    n = len(self.table) - len(unknow_neighbors)
    length = self.history_length * self.round
    incoming_rates = np.empty(shape=(n, length), dtype=np.float32)
    id_array = []
    # loop over the reachable neighbors
    for id, row in self.table:
      # local scheduler's row
      if id == self.id:
        # 1. update load (i.e., qsize)
        row.load = self.queue.qsize()
        incoming_rates[len(id_array), :] = row.incoming_rate.flatten()
        id_array.append(id)
      # neighbours with up-to-date information
      elif id not in unknow_neighbors:
        # 1. compute neighbor's load. this should take in account the following:
        #   - the previous load
        #   - plus the prior forecasting
        #   - minus the processing rate
        #   - more or less the offload
        row.load = self.table.compute_workload(row=row, w=self.w)
        incoming_rates[len(id_array), :] = row.incoming_rate.flatten()
        id_array.append(id)
        row.maximum_round -= 1 # updates the maximum round
        if row.maximum_round <= 1: # needs update
          msg = Message(
            at=time.time(), 
            arguments={ 'id': self.id, 'type': Signal.INFO, 'table': self.table, 'send_at': time.time() },
            priority=Priority.INFO,
            )
          await self.send_signal(msg=msg, keys=[id])
      # out-of-date information for these neighbours, need to request.
      else:
        pass
        
    incoming_rates[incoming_rates < 0] = 0 # set all negative load to 0.
    # 2. forecasting for all schedulers at once.
    incoming_rates = self.model(incoming_rates)[:,:self.round]
    incoming_rates[incoming_rates < 0] = 0 # set all negative load to 0.
    for i, id in enumerate(id_array):
      self.table.get(id).update_incoming_rate(incoming_rates[i]) # update the history.
      logging.debug('=== [row:{}] forecasting={}'.format(id, incoming_rates[i]))
    # 3. compute offload.
    await self.compute_policy(restart_offloading=True)
    
    
  async def compute_policy(self, restart_offloading):
    """Distribute incoming workload to replicas or neighbors.
    This is a method of determining the appropriate load balance based on the current state of the table. No predictions are made, but simply the round-robin weighted distribution is calculated and adjusted.

    Args:
      restart_offloading (bool): whether or not to eventually start a new offloadding process.
    """
    workloads = []
    processing_rates = []
    neighbor_ids = []
    for id, row in self.table:
      if id == self.id or not row.is_unknwon():
        # future load, i.e., qsize plus the incoming load.
        workloads.append(row.load + row.get_input_load(w=self.w))
        processing_rates.append(row.processing_rate)
        neighbor_ids.append(id)
    workloads = np.array(workloads, dtype=np.int32)
    processing_rates = np.array(processing_rates)
    # computes the offload.
    balanced_workloads = (workloads.sum() * processing_rates / (processing_rates.sum()+1e-9)).astype(np.int32)
    teta = np.round(balanced_workloads - workloads)
    # { 0: { 2: 26 }, 1: { 2: 6 } }
    policy = {
      from_id: { to_id: 0 for to_id in neighbor_ids if from_id != to_id }
      for from_id in neighbor_ids
    }
    while np.any(teta < 0) and np.any(teta > 0):
      from_id = np.argmin(teta) # from_id --> sender
      to_id = np.argmax(teta) # to_id --> receiver
      offload = np.abs(teta[from_id])
      if offload < teta[to_id]:
        policy[neighbor_ids[from_id]][neighbor_ids[to_id]] = offload
        teta[from_id] = 0
        teta[to_id] -= offload
      else:
        policy[neighbor_ids[from_id]][neighbor_ids[to_id]] = teta[to_id]
        teta[from_id] += teta[to_id]
        teta[to_id] = 0
    if restart_offloading:
      debug = []
      for k, v in policy.items():
        v = { k2: v2 for k2, v2 in v.items() if v2 }
        debug += ['%s->%s'%(k, v)]
      logging.info('\u2705 [videojam({})] - {}/{:.0f} | new computed workload: {}'.format(self.component_name, self.queue.qsize(), self.table.get(self.id).processing_rate, debug))
    
    # for id, row in self.table:
    #   if id in policy.keys():
    #     self.table.update_policy(key=id, policy=policy[id])
    #   else:
    #     self.table.reset_policy(key=id) # reset the workload offloaded to neighbors.
    # update all the rows in the table.
    for id in neighbor_ids:
      self.table.update_policy(key=id, policy=policy[id])
          
    # update its maximum offload.
    # if not self.table.get(self.id).is_overloaded(self.w, threshold=.002):
    #   self.table.reset_policy(self.id) # reset the workload offloaded to neighbors.

    # nothing is supposed to be sent to any neighbor.
    # 1. when there is no neighbor.
    # 2. the local scheduler doesn't have any offload from the computation.
    # 3. when it is not overloaded.
    # 4. or in the case it is a rectification.
    # if (not policy) or\
    #   (not policy[self.id]) or\
    #     (not self.table.get(self.id).is_overloaded(self.w)) or\
    #       (not restart_offloading):
    #   return
    if not restart_offloading:
      return
       
    # start offloading traffic to neighbors.
    self.offloading_key = round(time.time()) # this update will block all previous transfers that have not been completed.
    tasks = [
      self.offload2neighbor(to_neighbor_id, self.offloading_key)
      for to_neighbor_id in self.table.get(self.id).policy.keys() # for each destination (neighbor) begins to unload.
      ]
    await asyncio.gather(*tasks)
  

  async def offload2neighbor(self, to_neighbor_id, offloading_key: int):
    """Offload traffic to the queue of the destinated neighbor.

    Args:
      to_neighbor_id (_type_): id of the neighbor.
      key (int): key generated at each time we have to offload. It helps to prevent from two or more windows to overlap.
    """
    workload = self.table.get_maximum_discharge_workload(src=self.id, dest=to_neighbor_id)
    if not workload > 0:
      return
    num = 0
    # the rate the determine how fast data will be offloaded to the neighbor outgoing queue.
    rate = (self.round * self.w) * .8 / workload
    # rate = (self.round * self.w) * .8 / workload
    # We continue to unload until a new key is generated and until we reach the calculated quantity to be unloaded.
    try:
      while (self.offloading_key == offloading_key) and num < self.table.get_maximum_discharge_workload(src=self.id, dest=to_neighbor_id):
        task = asyncio.sleep(rate)
        msg = await self.queue.get(for_offloading=True)
        # msg = self.queue.get_nowait(for_offloading=True)
        await self.neighbor_queues[to_neighbor_id].put(msg)
        num += 1
        await task
    except asyncio.QueueEmpty:
      await task


  async def send_signal(self, msg: Message, keys: List[int]=[]):
    """Send message to remote neighbor(s).

    Args:
      msg (Message): _description_
      neighbors (List[RemoteOutPort], optional): _description_. Defaults to [].

    Raises:
      Exception: _description_
    """
    tasks = []
    # update availability for all unreachable neighbors.
    for id in keys:
      if not self.neighbors[id].is_ready():
        self.table.get(self.id).policy[id] = 0
        self.table.set_availability(id, False)
        logging.warning('[dispatcher]-removing neighbor[{}] from local table'.format(id))
      else:
        self.table.set_availability(id, True) # set availability back.
        tasks += [self.neighbors[id].push(msg)]
    if tasks: 
      await asyncio.gather(*tasks)
    
    
  async def receive_signal(self, msg: Message):
    """Compute the actual amount of load that each neighbor should have.
    It can be call either when a signal is received or when the monitor detect an overload.
    
    Args:
      msg (Message): Signal from a neighbor. 
    """
    sender_id, neighbor_type = msg.arguments['id'], msg.arguments['type']
    neighbor_table: Table = msg.arguments['table']
    has_changed = await self.update_table(neighbor_table) # update table
    logging.debug('[videojam({})]-{} signal has been received from {}'.format(self.component_name, msg.arguments['type'], sender_id))
    if neighbor_type == Signal.CONGESTION:
      logging.warning('[\u2798 CONGESTION] - notification from {}'.format(sender_id))
      # recompute the load balancing policy
      # await self.compute_policy(restart_offloading=False)
      self.table.get(self.id).policy[sender_id] = neighbor_table.get(self.id).policy[sender_id]
    elif neighbor_type == Signal.INFO:
      logging.info('[\u2798 INFO] - a reply signal will be sent to {}.'.format(sender_id))
      new_msg = Message(at=time.time(), priority=Priority.REPLY)
      new_msg.arguments['id'] = self.id # source
      new_msg.arguments['type'] = Signal.REPLY # a reply signal
      new_msg.arguments['send_at'] = time.time()
      self.table.get(self.id).maximum_round = self.maximum_round # update the maximum round before expiration.
      new_msg.arguments['table'] = self.table
      await self.send_signal(msg=new_msg, keys=[sender_id])
      pass
    elif neighbor_type == Signal.REPLY:
      pass # a reply signal will always be discard
    elif neighbor_type == Signal.OFFLOAD:
      # should append to the queue
      self.add2queue(msg, from_neighbor=True)
      logging.debug('offloading signal has been received, table updated: {}'.format(has_changed))
      # When the local scheduler is unloaded, it is not supposed to receive a load from another neighbour.
      if neighbor_table and ((self.table.get(self.id).get_offloaded() > 0) or has_changed):
        logging.info('[\u2798 OFFLOAD] - offloading signal will be handled (offload is: {})'.format(self.table.get(self.id).policy))
        value = await self.accept_offload(neighbor_table.get(self.id), neighbor_table.get(sender_id))
        if not value:
          new_msg = Message(
            at=time.time(), 
            arguments={ 'id': self.id, 'type': Signal.CONGESTION, 'table': self.table, 'send_at': time.time() },
            priority=Priority.CONGESTION,
            )
          # as ECN (Explicite Congestion Notification).
          await self.send_signal(msg=new_msg, keys=[sender_id])
    else:
      logging.error('\u274c Type signal unknown')
      raise ValueError('Unrecognized signal')
    
  
  async def accept_offload(self, neighbor_forecasting: Row, neighbor: Row) -> bool:
    """Offload controller which determines whether or not in incoming offload should whether be accepted or rejected.

    Args:
      neighbor_forecasting (Row): neighbor's forecasting about local state.
      neighbor (Row): neighbor's row info (e.g., its local forecasting)

    Returns:
      bool: true means the offload is accepted, false otherwise.
    """
    await self.compute_policy(restart_offloading=False)
    # after update, if the neighbor is still not in the offloading of its neighbor, then raise congestion.
    # if my neighbor is not offloading to me (i.e., maximum offload is 0) according to the last computation.
    if self.table.get_maximum_discharge_workload(src=neighbor.id, dest=self.id) == 0:
      logging.warning('\u292D [CONGESTION] - congestion local:{} key id not present into neighbor:{}\'s {} receiver after computation'\
        .format(self.id, neighbor.id, list(neighbor.policy.keys())))
      return False
    elif (self.table.get(self.id).get_offloaded() > 0):
      # This means that the scheduler is supposed to receive the data from the neighbour that has been checked and, at the same time, offload it to other neighbours.
      raise ValueError('\u274c Computation error since local offload is id={}->{} while neighbor send also to local id={}->{}'\
        .format(self.id, self.table.get(self.id).policy, neighbor.id, self.table.get(neighbor.id).policy))
    else:
      try:
        # check forecasting error from remote neighbor.
        neighbor_load = self.table.compute_workload(row=neighbor_forecasting, w=self.w)
        load = self.table.compute_workload(row=self.table.get(self.id), w=self.w)
        unbalance = np.abs(neighbor_load - load) / (neighbor_load + 1e-10)
        
        if unbalance > self.threshold:
          logging.warning('[receiver]-miss forecasting unbalance is {:.2f} (pred={:.2f}, truth={:.2f}), will send update CONGESTION signal to {}'\
            .format(unbalance, neighbor_load, load, neighbor.id))
          # send a reply message for update.
          new_msg = Message(
            at=time.time(),
            arguments={ 'id': self.id, 'type': Signal.CONGESTION, 'table': self.table, 'send_at': time.time() },
            priority=Priority.CONGESTION,
            )
          await self.send_signal(msg=new_msg, keys=[neighbor.id])
      except Exception: pass
    return True
  
 
  async def push(self, id, msg: Message):
    """Receiving data from previous steps or neighbors.

    Args:
      id (_type_): _description_
      msg (Message): received message.
    """
    if 'type' in msg.arguments.keys():
      await self.receive_signal(msg)
    else:
      self.num_received += self.add2queue(msg)


  def add2queue(self, msg: Message, from_neighbor=False):
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
      msg.arguments.pop('type', None) # delete key if exists
      msg.arguments.pop('table', None)# delete key if exists
      msg.priority = Priority.DEFAULT
      self.queue.put_nowait(msg, from_offloading=from_neighbor)
    return len(data)
  
  
  def on_maxsize(self, msg):
    """A callback method called whenever a message reaches its time out.

    Args:
      msg (_type_): any message to show for more information.
    """
    logging.warning('[videojam(%s)]-%s'%(self.component_name, msg))
      
  
  async def consumer(self):
    """A consumer from the queue of tasks.

    Args:
      outport (LocalOutPort): the actual operator.
    """
    # here the scheduler push data as fast as the consumer (next instance)
    max_batch = 0
    throughputs = np.full(shape=(self.round,), fill_value=np.nan, dtype=np.float32)
    while True:
      batch: List[Message] = await self.queue.get_batch()
      N = len(batch)
      elapsed = time.time()
      result = await self.component.push(0, batch)
      elapsed = time.time() - elapsed
      if N >= max_batch:
        max_batch = N
        throughputs[:-1] = throughputs[1:]
        throughputs[-1] = max_batch / elapsed
        self.table.get(self.id).processing_rate = np.nanmedian(throughputs)
        logging.info('[videojam(%s)]-Throughput: %.2f | Queue size: %i'%(self.component_name, self.table.get(self.id).processing_rate, self.queue.qsize()))
      await self.next(data=result) # send data to the next in the pipeline if exists.


  async def offloader(self, outport: RemoteOutPort, queue: IQueue):
    """A offloader from its queue of tasks.

    Args:
      outport (RemoteOutPort): a remote scheduler.
    """
    key = None
    while True:
      msg: Message = await queue.get()
      # the table will be sent only once.
      msg.arguments['send_at'] = time.time()
      msg.arguments['type'] = Signal.OFFLOAD
      msg.arguments['id'] = self.id
      if key != self.offloading_key:
        msg.arguments['table'] = self.table
        key = self.offloading_key
      else:
        msg.arguments['table'] = None
      msg.priority = Priority.OFFLOAD
      await outport.push(msg)


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
          self.neighbor_queues[outport.id] = asyncio.PriorityQueue()
          # init all neighbors with weight to 0.
          row = Row(
            id=outport.id,
            at=0,
            incoming_rate=np.full(shape=(self.history_length, self.round), fill_value=-1, dtype=np.float32), # not set yet
            )
          self.table.add(key=outport.id, row=row)
      else:
        raise ValueError('Unknown outport type {}'.format(type(outport)))
    self.outgoing = [outport for outport in self.outgoing if outport.id == 0]
      
    # init table row.
    row = Row(
      id=self.id,
      at=0,
      num_replicas=1,
      incoming_rate=np.full(shape=(self.history_length, self.round), fill_value=-1, dtype=np.float32), # not set yet
      processing_rate=self.processing_rate,
      maximum_round=self.maximum_round,
      )
    self.table.add(key=self.id, row=row)
    # start the monitor
    tasks = [self.monitor()]
    # add the component
    tasks += [self.consumer()]
    # offloaders=remote schedulers
    tasks += [self.offloader(outport, self.neighbor_queues[id]) for id, outport in self.neighbors.items()]
    await asyncio.gather(*[asyncio.get_event_loop().create_task(task) for task in tasks])


  def get_async_tasks(self):
    '''

    :return:
    '''
    return Function.get_async_tasks(self) + [self.run()]