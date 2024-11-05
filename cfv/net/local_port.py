import asyncio
from cfv.net.port import InPort, OutPort


class LocalInPort(InPort):
  '''
  Incoming port that uses local queues to handle message passing.
  '''
  def __init__(self, id, callback, asynchronous=True, batch_size=None):
    """Instantiate a new port and set the callback call

    Args:
      id (_type_): Port identifier
      callback (function): Function to call when a new message is received
      asynchronous (bool, optional): Whether to run the port in asynchronous mode. Defaults to True.
      batch_size (_type_, optional): Whether to run batch or not. Defaults to None.
    """
    InPort.__init__(self, id, callback)
    self.asynchronous = asynchronous
    if self.asynchronous:
      self.queue = asyncio.Queue()
    else:
      self.queue = None
    self.canceled = False
    self.batch_size = batch_size

  async def push(self, msg):
    '''
    Pass a message into the port
    :param msg: Message to pass
    :return:
    '''
    if self.asynchronous:
      self.queue.put_nowait(msg)
    else:
      await self.callback(self.id, msg)

  async def run(self):
    '''
    Control loop to continously read messages in queue
    :return:
    '''
    while not self.canceled:
      if (self.batch_size is not None) and (self.batch_size > 1):
        # the component will run on batch of workload.
        while not self.canceled:
          # make sure to wait for the first message, otherwise the loop may crash the execution.
          task = asyncio.get_event_loop().create_task(self.queue.get())
          msg = await task
          batch = [msg] # append the first element.
          # as long as the batch size has not been reached and there is an element in the queue.
          while self.queue.qsize() > 0 and len(batch) < self.batch_size:
            msg = self.queue.get_nowait()
            batch.append(msg)
          await self.callback(self.id, batch)
      else:
        # the component will run on bat
        while not self.canceled:
          task = asyncio.get_event_loop().create_task(self.queue.get())
          msg = await task
          await self.callback(self.id, msg)

  def get_runners(self):
    '''
    Get the tasks to run in asynchronous mode
    :return: self.run if is asynchronous
    '''
    if self.asynchronous:
      return [self.run()]
    else:
      return []


class LocalOutPort(OutPort):
  def __init__(self, id, nextPort, asynchronous=True):
    '''

    :param id: Port identifier
    :param nextPort: InPort to send messages to
    :param asynchronous: Whether to run the port in asynchronous mode
    '''
    OutPort.__init__(self, id, nextPort)
    self.asynchronous = asynchronous


  async def push(self, msg):
    '''
    Pass a message to the next port
    :param msg: Message to pass
    :return:
    '''
    await self.nextPort.push(msg)