import time
import glob
import asyncio
import logging
import multiprocessing
from typing import List, Dict
from collections import defaultdict
import numpy as np

from cfv.net.message import Message, Priority
from cfv.net.local_port import LocalOutPort
from cfv.net.remote_port import RemoteOutPort
from cfv.utils.videojam import Row, Signal, Table
from cfv.utils.short_term_forecast_opencv import Model
from cfv.utils.track import trace_mark, csv_writer
from cfv.utils.general import get_pipeline_id


# >>>>>>> test purposes
PIPELINE_ID = get_pipeline_id()
# <<<<<<< test purposes


class LB(multiprocessing.Process):
  """Component is multiprocessing component that runs in parallel to the load balancer and communicate through the given queue.
  """
  def __init__(self,
               config: dict,
               queue: multiprocessing.JoinableQueue,
               signal: multiprocessing.JoinableQueue,
               qsize: multiprocessing.Value,
               num_received: multiprocessing.Value,
               ):
    super(LB, self).__init__(daemon=True)
    self.queue = queue
    self.signal = signal
    # compo
    self.id = None # scheduler's id (from incoming[0].id)
    self.w = 1 # (seconds)
    self.round = 10 # how many window to gether for a round for forecasting.
    self.threshold = .2
    self.num_received = num_received
    self.qsize = qsize
    self.num_processed = 0
    self.maximum_round = 5
    self.history_length = 5
    self.mp_processing_rate = multiprocessing.Value('d', 0.0)
    # workers which are assumed to be locally deployed an synchrous.
    self.replicas: List[LocalOutPort] = []
    self.queue = queue # outgoing tasks for consumers.
    # table that contains all information about scheduler row of performance.
    self.table: Table = Table()
    # schedulers which are assumed to be remote.
    self.neighbors: Dict[str, RemoteOutPort] = {}
    # tasks+signal for neighbors (should be a priority queue instead)
    self.neighbor_queues: Dict[int, asyncio.Queue] = {}
    # This variable is used to prevent two unloading windows from crossing. Only the one with right key will keep offloading.
    self.offloading_key = None
    
    self.configure(config=config)
    
    
  def configure(self, config):
    """There is no configuration
    """

    if 'w' not in config.keys():
      raise ValueError('Missing w parameter')
    self.w = config['w']
    if 'round' not in config.keys():
      raise ValueError('Missing round parameter')
    self.round = config['round']
    if 'data' not in config.keys():
      raise ValueError('Missing data parameter')
    self.full_pb_path = glob.glob('{}/*.pb'.format(config['data']))[0]
    self.model = Model(self.full_pb_path)
    if 'processing_rate' not in config.keys():
      self.mp_processing_rate.value = 0
    else:
      self.mp_processing_rate.value = config['processing_rate']
    if 'batch_size' not in config.keys():
      self.batch_size = 1
    else:
      self.batch_size = config['batch_size']
    logging.info('[Scheduler]-window: {}s - round: {} - model: {}'.format(self.w, self.round, self.full_pb_path))
    


  async def update_table(self, table: Table) -> bool:
    """Responsible of receiving signal from neighbor schedulers.

    Args:
      msg (Message): signal that should be treated

    Returns:
      bool: True if changes have been observed.
    """
    
    res = False
    for id, row in table:
      # discard any update from itself.
      if id == self.id:
        pass
      else:
        # the current row is the latest
        if self.table.update(row):
          res = True
        
    return res
  
  
  def get_process_rate(self) -> float:
    return self.mp_processing_rate.value
  
  
  def get_qsize(self) -> int:
    return self.qsize.value
  
  
  def get_num_received(self) -> int:
    return self.num_received.value
  
  def compute_neighbour_load(self, row: Row) -> int:
    """Update the load over the next futur time interval, i.e., the following:
      - the previous load
      - plus the prior forecasting
      - minus the processing rate
      - and more or less the offload.

    Args:
    """
    # load = qsize + sum(input_rate - proc_rate) * w + offload
    load = row.load\
      + np.sum(row.get_latest_incoming_rate() * self.w)\
      - row.get_maximum_processed_per_window(self.w)\
      - row.get_offloaded()\
      + np.sum([
        neighbor.offloading[row.id] # offload from a neighbor to the given row.
        for neighbor in self.table.rows()
        if row.id in neighbor.offloading.keys()
        ])
    return load if (load > 0) else 0
  
  
  async def monitor(self):
    """It's responsability is to monitor the performance of the workers or replicas in term
    of processing rate, queue size, etc.
    It can also trigger the dispatcher if it happens to detect an overload.
    """
    
    logging.info('[monitor]-monitoring has been initiated at {} for a window of {}seconds'.format(time.time(), self.w))
    input_rate_w = np.empty(shape=(self.round))
    while True:
      # initializations
      w = 1
      input_rate_w[:] = 0
      pred_input_rate_w = self.table.get(self.id).get_latest_incoming_rate().copy()
      while w < self.round:
        num_received_w = self.get_num_received()
        await asyncio.sleep(self.w)                # wait for collecting data.
        num_received_w = self.get_num_received() - num_received_w
        input_rate_w[w] = num_received_w / self.w   # incoming rate
        logging.info('[monitor]-w[{}] -> input_rate={}, qsize={}'.format(w+1, input_rate_w[w], self.get_qsize()))
        self.table.get(self.id).incoming_rate[-1, :self.round-w] = self.table.get(self.id).incoming_rate[-1, w:]
        self.table.get(self.id).incoming_rate[-1, -w:] = input_rate_w[:w]
        if np.all(self.table.get(self.id).get_latest_incoming_rate() >= 0):
          error = (input_rate_w[:(w+1)]-self.table.get(self.id).incoming_rate[-1, :(w+1)]) / (input_rate_w[:(w+1)] + 1e-10)
          logging.debug('[monitor]-load variation error is {}'.format(error))
          if np.sum(error > self.threshold) >= self.round // 2:
            # call the dispatch (not the dispatcher) method for adjusting the scheduling policy.
            # self.dispatch() 
            pass
        w += 1
      at = time.time()
      # update the table (time, processing rate)
      self.table.get(self.id).at = at
      self.table.get(self.id).processing_rate = self.get_process_rate()
      asyncio.create_task(self.dispatcher()) # awake the dispatcher.
      logging.info('[monitor]-a new round has been completed.')
      asyncio.create_task(csv_writer(
        dir='cfv_log/{}/decentralized_lb'.format(PIPELINE_ID),
        filename='monitor',
        mode='a',
        data=[{
          'at': at,
          'id': self.id,
          'process_rate': self.table.get(self.id).processing_rate,
          'actual_input': np.sum(input_rate_w * self.w),
          'pred_input': np.sum(pred_input_rate_w * self.w),
        }]
      ))
    

  async def dispatcher(self):
    """Dispatcher.
    The dispatcher is responsible for using the data from the monitor and its neighbors to first make a forecast and then calculate the scheduling policy.
    """
    
    logging.info('[dispatcher]-dispatcher has been executed.')
    # update availability for all unreachable neighbors.
    for id, outport in self.neighbors.items():
      if not outport.is_ready():
        self.table.get(self.id).offloading[id] = 0
        self.table.set_availability(id, False)
        logging.warn('[dispatcher]-removing neighbor[{}] from local table'.format(id))
      else:
        self.table.set_availability(id, True) # set availability back.
        
    unknow_neighbors = [id for id, row in self.table if id != self.id and row.is_unknwon()]
    if unknow_neighbors:
      msg = Message(
        arguments={ 'id': self.id, 'type': Signal.INFO, 'table': self.table },
        priority=Priority.INFO,
        )
      logging.debug('[dispatcher]-requestion information will be sent to neighbors')
      await self.send_signal(msg=msg, keys=unknow_neighbors)
      
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
        row.load = self.get_qsize()
        incoming_rates[len(id_array), :] = row.incoming_rate.flatten()
        id_array.append(id)
      # neighbours with up-to-date information
      elif not row.is_unknwon():
        # 1. compute neighbor's load. this should take in account the following:
        #   - the previous load
        #   - plus the prior forecasting
        #   - minus the processing rate
        #   - more or less the offload
        row.load = self.compute_neighbour_load(row)
        incoming_rates[len(id_array), :] = row.incoming_rate.flatten()
        id_array.append(id)
        self.table.get(id).maximum_round -= 1 # updates the maximum round
      # out-of-date information for these neighbours, need to request.
      else:
        pass
        
    # get the maximum observed load for each scheduler.
    # max_observed_load = incoming_rates.max(axis=1)
    
    incoming_rates[incoming_rates < 0] = 0 # set all negative load to 0.
    # 2. forecasting for all schedulers at once.
    incoming_rates = self.model(incoming_rates)
    incoming_rates[incoming_rates < 0] = 0 # set all negative load to 0.
    for i, id in enumerate(id_array):
      # incoming_rates[i][incoming_rates[i] > max_observed_load[i]] = max_observed_load[i]
      self.table.get(id).update_incoming_rate(incoming_rates[i]) # update the history.
      print('=== [row:{}] forecasting={}'.format(id, incoming_rates[i]))
    
    # 3. compute oafflaod.
    await self.compute_offload()
    
    
  async def compute_offload(self, possible_offloading_to_neighbours=True):
    """Distribute incoming workload to replicas or neighbors.
    This is a method of determining the appropriate load balance based on the current state of the table. No predictions are made, but simply the round-robin weighted distribution is calculated and adjusted.

    Args:
      possible_offloading_to_neighbours (bool, optional): whether or not to eventually start a new offloadding process. Defaults to True.
    """
    
    loads = []
    processing_rates = []
    neighbor_ids = []
    for id, row in self.table:
      if id == self.id or not row.is_unknwon():
        # future load, i.e., qsize plus the incoming load.
        loads.append(row.load + row.get_input_load(w=self.w))
        processing_rates.append(row.processing_rate)
        neighbor_ids.append(id)
    loads = np.array(loads, dtype=np.int32)
    processing_rates = np.array(processing_rates, dtype=np.int32)
    # computes the offload.
    balanced_loads = (loads.sum() * processing_rates / processing_rates.sum()).astype(np.int32)
    offload_per_n = np.round(balanced_loads - loads)
    # { 0: { 2: 26 }, 1: { 2: 6 } }
    max_offload_n = defaultdict(dict)
    while np.any(offload_per_n < 0) and np.any(offload_per_n > 0):
      from_id = np.argmin(offload_per_n) # from_id --> sender
      to_id = np.argmax(offload_per_n) # to_id --> receiver
      offload = np.abs(offload_per_n[from_id])
      if offload < offload_per_n[to_id]:
        max_offload_n[neighbor_ids[from_id]][neighbor_ids[to_id]] = offload
        offload_per_n[from_id] = 0
        offload_per_n[to_id] -= offload
      else:
        max_offload_n[neighbor_ids[from_id]][neighbor_ids[to_id]] = offload_per_n[to_id]
        offload_per_n[from_id] += offload_per_n[to_id]
        offload_per_n[to_id] = 0
    logging.info('[dispatcher]-new computed load {}'.format(max_offload_n))      
    
    # update all the table.
    for id, row in self.table:
      if id in max_offload_n.keys():
        self.table.get(id).offloading = max_offload_n[id]
      else:
        self.table.get(id).offloading = {}
          
    # update its maximum offload.
    if not self.table.get(self.id).is_overloaded(self.w):
      self.table.get(self.id).offloading = {} # since the scheduler is not overloaded.
    
    at = time.time()
    asyncio.create_task(csv_writer(
      dir='cfv_log/{}/decentralized_lb'.format(PIPELINE_ID),
      filename='dispatcher_id={}_pipeline-id={}'.format(self.id, PIPELINE_ID, PIPELINE_ID),
      mode='a',
      data=[{
        'at': at,
        'decentralized_lb_id': self.id,
        'id': row.id,
        'qsize': row.load, # 1. get qsize if local or computed for neighbors.
        'pred_input': np.sum(row.get_latest_incoming_rate() * self.w), # 2. forecasted load.
        'offloaded': row.get_offloaded(), # 3. computed offload.
        'overloaded': row.is_overloaded(self.w), # 3. computed offload.
        'possible_offloading_to_neighbours': possible_offloading_to_neighbours,
        }
      for row in self.table.rows()]
    ))
    
    # nothing is supposed to be sent to any neighbor.
    # 1. when there is no neighbor.
    # 2. the local scheduler doesn't have any offload from the computation.
    # 3. when it is not overloaded.
    # 4. or in the case it is a rectification.
    if (not max_offload_n) or\
      (not max_offload_n[self.id]) or\
        (not self.table.get(self.id).is_overloaded(self.w)) or\
          (not possible_offloading_to_neighbours):
      return

    async def offload(to_neighbor_id, key: int):
      """Offload traffic to the queue of the destinated neighbor.

      Args:
        to_neighbor_id (_type_): id of the neighbor.
        key (int): key generated at each time we have to offload. It helps to prevent from two or more windows to overlap.
      """
      try:
        # the rate the determine how fast data will be offloaded to the neighbor outgoing queue.
        rate = self.round * self.w / self.table.get(self.id).offloading[to_neighbor_id]
        # We continue to unload until a new key is generated and until we reach the calculated quantity to be unloaded.
        i = 0
        while (self.offloading_key == key) and i < self.table.get(self.id).offloading[to_neighbor_id]:
          task = asyncio.sleep(rate)
          msg = self.queue.get()
          self.queue.task_done()
          await self.neighbor_queues[to_neighbor_id].put(msg)
          i += 1
          await task
      except Exception as e:
        logging.error('[dispatcher]-error during transfer of data from queue to neighbor {}, {}'.format(to_neighbor_id, e))
       
    # start offloading traffic to neighbors.
    self.offloading_key = round(at) # this update will block all previous transfers that have not been completed.
    await asyncio.gather(*[asyncio.create_task(offload(to_neighbor_id, self.offloading_key)) for to_neighbor_id in max_offload_n[self.id]])
     
  

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
        self.table.get(self.id).offloading[id] = 0
        self.table.set_availability(id, False)
        logging.warn('[dispatcher]-removing neighbor[{}] from local table'.format(id))
      else:
        self.table.set_availability(id, True) # set availability back.
        tasks += [self.neighbors[id].push(msg)]
    if tasks:
      await asyncio.gather(*tasks)
    
    
    
  @trace_mark(path='cfv_log/{}/decentralized_lb'.format(PIPELINE_ID), mode='a')
  async def receive_signal(self, msg: Message):
    """Compute the actual amount of load that each neighbor should have.
    It can be call either when a signal is received or when the monitor detect an overload.
    
    Args:
      msg (Message): Signal from a neighbor. 
    """
    sender_id, neighbor_type = msg.arguments['id'], msg.arguments['type']
    neighbor_table: Table = msg.arguments['table']
    has_changed = await self.update_table(neighbor_table) # update table
    logging.info('[receiver]-{} signal has been received from {}'.format(msg.arguments['type'], sender_id))
    if neighbor_type == Signal.CONGESTION:
      logging.warning('Congestion notification from {}'.format(sender_id))
      # updating the maximum offloading for that particular neighbors is enough to correct the mistake until the next window.
      try:
        self.table.get(self.id).offloading[sender_id] = neighbor_table.get(self.id).offloading[sender_id]
      except Exception as e:
        self.table.get(self.id).offloading[sender_id] = 0
      process_at = time.time()
      asyncio.create_task(csv_writer(
        dir='cfv_log/{}/decentralized_lb'.format(PIPELINE_ID),
        filename='congestion_signal_id={}_pipeline-id={}'.format(self.id, PIPELINE_ID, PIPELINE_ID),
        mode='a',
        data={
          'id': self.id,
          'sent_at': msg.arguments['send_at'],
          'processed_at': process_at,
          # local scheduler
          'qsize': self.table.get(self.id).load,
          'offloaded': self.table.get(self.id).get_offloaded(),
          'pred_input': (self.table.get(self.id).get_latest_incoming_rate() * self.w).sum(),
          # neighbor's view of the local state.
          'n_local_qsize': neighbor_table.get(self.id).load,
          'n_local_offloaded': neighbor_table.get(self.id).get_offloaded(),
          'n_local_pred_input': (neighbor_table.get(self.id).get_latest_incoming_rate() * self.w).sum(),
          # remote neighbor
          'n_id': neighbor_table.get(sender_id).id,
          'n_pred_input': (neighbor_table.get(sender_id).get_latest_incoming_rate() * self.w).sum(),
          'n_qsize': neighbor_table.get(sender_id).load,
          'n_offloaded': neighbor_table.get(sender_id).get_offloaded(),
          'n_processing_rate': neighbor_table.get(sender_id).processing_rate,
          }
        ))
    elif neighbor_type == Signal.INFO:
      logging.info('[receiver]-REPLY signal will be sent to {}.'.format(sender_id))
      new_msg = Message(priority=Priority.REPLY)
      new_msg.arguments['id'] = self.id # source
      new_msg.arguments['type'] = Signal.REPLY # a reply signal
      self.table.get(self.id).at = time.time()
      self.table.get(self.id).maximum_round = self.maximum_round # update the maximum round before expiration.
      new_msg.arguments['table'] = self.table
      await self.send_signal(msg=new_msg, keys=[sender_id])
    elif neighbor_type == Signal.REPLY:
      pass # a reply signal will always be discard
    elif neighbor_type == Signal.OFFLOAD:
      # should append to the queue
      self.queue.put(msg)
      logging.info('[receiver]-offloading signal has been received, table updated: {}'.format(has_changed))
      # When the local scheduler is unloaded, it is not supposed to receive a load from another neighbour.
      if (self.table.get(self.id).get_offloaded() > 0) or (self.id not in self.table.get(sender_id).offloading.keys()) or has_changed:
        logging.info('[receiver]-offloading signal will be handled (offload is: {})'.format(self.table.get(self.id).offloading))
        value = await self.accept_offload(neighbor_table.get(self.id), neighbor_table.get(sender_id))
        # value = await self.__accept_offload(neighbor_table.get(self.id), neighbor_table.get(sender_id))
        if not value:
          new_msg = Message(
            arguments={ 'id': self.id, 'type': Signal.CONGESTION, 'table': self.table, 'send_at': time.time() },
            priority=Priority.CONGESTION,
            )
          # as ECN (Explicite Congestion Notification).
          await self.send_signal(msg=new_msg, keys=[sender_id])
    else:
      logging.error('Type signal unknown')
    
  
  async def accept_offload(self, neighbor_forecasting: Row, neighbor: Row) -> bool:
    """Offload controller which determines whether or not in incoming offload should whether be accepted or rejected.

    Args:
      neighbor_forecasting (Row): neighbor's forecasting about local state.
      neighbor (Row): neighbor's row info (e.g., its local forecasting)

    Returns:
      bool: true means the offload is accepted, false otherwise.
    """
    await self.compute_offload(possible_offloading_to_neighbours=False)
    # after update, if the neighbor is still not in the offloading of its neighbor, then raise congestion.
    # if I'm not in my neighbor's offload according to the last computation.
    if self.id not in self.table.get(neighbor.id).offloading.keys():
      logging.warning('[receiver]-congestion between local {} and remote {} respectively \n\t{}\n\t{}'\
        .format(self.id, neighbor.id, self.table.get(self.id).offloading.keys(), neighbor.offloading.keys()))
      return False
    elif (self.table.get(self.id).get_offloaded() > 0):
      # This means that the scheduler is supposed to receive the data from the neighbour that has been checked and, at the same time, offload it to other neighbours.
      raise ValueError('Computation error since local offload is id={}:{} while neighbor send also to local id={}:{}'.format(self.id, self.table.get(self.id).offloading, neighbor.id, self.table.get(neighbor.id).offloading))
    else:
      # check forecasting error from remote neighbor.
      error = np.abs((neighbor_forecasting.get_latest_incoming_rate() - self.table.get(self.id).get_latest_incoming_rate()).sum()) / (neighbor_forecasting.get_latest_incoming_rate().sum() + 1e-10)
      
      if error > self.threshold:
        logging.warn('[receiver]-miss forecasting error is {:.2f} (pred={:.2f}, truth={:.2f}), will send update CONGESTION signal to {}'\
          .format(error, neighbor_forecasting.get_latest_incoming_rate().sum(), self.table.get(self.id).get_latest_incoming_rate().sum(), neighbor.id))
        # send a reply message for update.
        new_msg = Message(arguments={ 'id': self.id, 'type': Signal.CONGESTION, 'table': self.table, 'send_at': time.time() }, priority=Priority.CONGESTION)
        await self.send_signal(msg=new_msg, keys=[neighbor.id])
    return True
  
 
  @trace_mark(path='cfv_log/{}/decentralized_lb'.format(PIPELINE_ID), mode='a')
  async def push(self, id, msg: Message):
    """Receiving data from previous steps or neighbors.

    Args:
      id (_type_): _description_
      msg (Message): _description_
    """
      
    if 'type' in msg.arguments.keys():
      await self.receive_signal(msg)
    else:
      msg.arguments['number_trans'] = 0
      self.queue.put(msg)
      self.num_received += 1


  async def offloader(self, outport: RemoteOutPort, queue: asyncio.Queue):
    """A offloader from its queue of tasks.

    Args:
      outport (RemoteOutPort): a remote scheduler.
    """

    while True:
      msg: Message = await queue.get()
      # self.table.get(outport.id).load += 1 # increase the neighbor's qsize
      msg.arguments = { 'id': self.id, 'type': Signal.OFFLOAD, 'table': self.table }
      msg.priority = Priority.OFFLOAD
      await outport.push(msg)


  async def _start_running(self):
    """Settle all the stuff such as the monitor, the consumers and offloaders and so on.
    Some assumption have been made:
      - all local outport are considered being the consumers;
      - all remote outport are considered to be neighbors;
      - each scheduler should have a unique value (outport.id) that match with each other on the configuration file.
    Raises:
      ValueError: if id for outgoing port is missed.
      ValueError: if an unknown OutPort is given.
    """
    self.id = 100
    if self.id is None:
      raise Exception('Please give an id for scheduler port')
    # for outport in self.outgoing:
    #   if isinstance(outport, LocalOutPort):
    #     self.replicas.append(outport)
    #   elif isinstance(outport, RemoteOutPort):
    #     if outport.id is None:
    #       raise ValueError('Please give an id for outgoing port')
    #     self.neighbors[outport.id] = outport
    #     self.neighbor_queues[outport.id] = asyncio.PriorityQueue(maxsize=500)
    #     # init all neighbors with weight to 0.
    #     row = Row(
    #       id=outport.id,
    #       at=0,
    #       incoming_rate=np.full(shape=(self.history_length, self.round), fill_value=-1, dtype=np.float32), # not set yet
    #       )
    #     self.table.add(key=outport.id, row=row)
    #   else:
    #     raise ValueError('Unknown outport type {}'.format(type(outport)))
    
    # init table row.
    row = Row(
      id=self.id,
      at=0,
      num_replicas=len(self.replicas),
      incoming_rate=np.full(shape=(self.history_length, self.round), fill_value=-1, dtype=np.float32), # not set yet
      processing_rate=self.get_process_rate(),
      maximum_round=self.maximum_round,
      )
    self.table.add(key=self.id, row=row)
    
    # start the monitor
    tasks = [self.monitor()]
    # offloaders=remote schedulers
    tasks += [self.offloader(outport, self.neighbor_queues[id]) for id, outport in self.neighbors.items()]
    await asyncio.gather(*[asyncio.get_event_loop().create_task(task) for task in tasks])
    
  def run(self):
    asyncio.run(self._start_running())