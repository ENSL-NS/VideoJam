import asyncio
import aiohttp
import logging

from cfv.net.port import InPort, OutPort
from cfv.net.message import Message
from cfv.net.http import HTTPServer, HTTPClient
from cfv.net.websocket import WebSocketServer, WebSocketClient


class RemoteInPort(InPort):

  def __init__(self, id, callback, host, port, marshalling='pickle', protocol='http'):
    '''

    '''
    InPort.__init__(self, id, callback)
    self.canceled = False
    self.connected = False
    self.marshalling = marshalling.lower()
    self.protocol = protocol.lower()
    self.host = host
    self.port = port
    self._queue = asyncio.PriorityQueue()
    self.server = None

  async def setup(self):
    '''
    Start server listening for incoming connections

    :return:
    '''
    if self.protocol == 'http':
      self.server = HTTPServer(self.host, self.port, self.received)
    elif self.protocol == 'websocket':
      self.server = WebSocketServer(self.host, self.port, self.received)
    else:
      raise TypeError('{} is an invalid connection protocol'.format(self.protocol))

    await self.server.setup()


  def is_ready(self):
    '''
    Check if server is ready
    
    :return:
    '''
    return self.server.is_ready()

  
  async def wait_ready(self):
    '''
    '''
    while not self.is_ready():
      logging.warning('Not ready yet, waiting 0.1s')
      await asyncio.get_event_loop().create_task(asyncio.sleep(0.1))


  def get_runners(self):
    return self.server.get_runners() + [self.run()]


  async def received(self, msg):
    await self._queue.put(msg)


  async def run(self):
    # the component will run on bat
    while not self.canceled:
      task = asyncio.create_task(self._queue.get())
      msg = await task
      logging.debug('Read message from queue')
      await self.callback(self.id, msg)



class RemoteOutPort(OutPort):
  def __init__(self, id, remote_ip, remote_port, marshalling='pickle', protocol='http'):
    '''

    '''
    OutPort.__init__(self, id, None)
    self.canceled = False
    self.connected = False
    self.marshalling = marshalling
    self.remote_ip = remote_ip
    self.remote_port = remote_port
    self.protocol = protocol
    self._queue = asyncio.PriorityQueue()
    

  async def setup(self):
    '''Setup the connection with the remote node.

    Raises:
      TypeError: _description_
    '''
    if self.protocol == 'http':
      self.server = HTTPClient(self.remote_ip, self.remote_port)
    elif self.protocol == 'websocket':
      self.server = WebSocketClient(self.remote_ip, self.remote_port)
    else:
      raise TypeError('{} is an invalid connection protocol'.format(self.protocol))

    await self.server.setup(on_connect=self.on_connect, on_reset=self.on_reset)


  async def push(self, msg: Message):
    '''Pushing message over network.

    Args:
      msg (Message): message to send.
    '''
    if self.is_ready():
      msg.net_queue_put()
      await self._queue.put(msg)
    else:
      pass


  def get_runners(self):
    return self.server.get_runners() + [self.run()]


  async def run(self):
    while not self.canceled:
      msg: Message = await self._queue.get()
      msg.net_queue_get()
      try:
        await self.server.send(msg)
      except (ConnectionResetError, aiohttp.ClientOSError):
        logging.error('Connection has been lost. Will try to reconnect')
        await self.server.connect()


  def marshall_message(self, msg: Message):
    '''

    :param message:
    :return:
    '''
    if self.marshalling == 'json':
      return msg.marshal()
    elif self.marshalling == 'pickle':
      return msg.marshal()
    else:
      return msg.marshal()
    
  def __repr__(self) -> str:
    return """RemoteOutPort(
      'remote_ip': {},
      'remote_port': {},
      'protocol': {},
      'marshalling': {},
      'canceled': {},
      'connected': {},
    )""".format(
      self.remote_ip,
      self.remote_port,
      self.protocol,
      self.marshalling,
      self.canceled,
      self.connected,
      )
