import aiohttp
import logging
import asyncio
from aiohttp import web

from cfv.net.message import Message
from aiohttp.client_exceptions import ClientConnectorError


class WebSocketServer:
  CLIENT_MAX_SIZE = 1024*1024*20

  def __init__(self, host, port, callback):
    self.host = host
    self.port = port
    self.application = None
    self.ready = False
    self.callback = callback

  async def setup(self):
    self.application = web.Application(client_max_size=WebSocketServer.CLIENT_MAX_SIZE)
    self.application.add_routes([web.get('/ws', self.websocket_handler)])
    self.application.on_startup.append(self.set_ready)

  async def websocket_handler(self, request):
    # READ BODY
    logging.debug("Received new socket connection: {}".format(request.headers))

    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)

    async for websocket_message in ws:
      if websocket_message.type == aiohttp.WSMsgType.TEXT:
        if websocket_message.data[0] == 'c':
          await ws.close()
        elif websocket_message.data[0] == 'f':
          msg = Message()
          msg.unmarshal(websocket_message.data[1:])
          await self.callback(msg)
      elif websocket_message.type == aiohttp.WSMsgType.BINARY:
        if websocket_message.data[0] == b'\x01'[0]:
          await ws.close()
        elif websocket_message.data[0] == b'\x00'[0]:
          msg = Message()
          msg.unmarshal(websocket_message.data[1:])
          msg.network_recv()
          await self.callback(msg)
      elif websocket_message.type == aiohttp.WSMsgType.ERROR:
        logging.error('ws connection closed with exception {}'.format(ws.exception()))

    logging.debug('websocket connection closed')
    
    return ws

  def get_runners(self):
    self.connection = web._run_app(self.application, host=self.host, port=self.port, access_log=None, print=logging.debug)
    return [self.connection]

  async def set_ready(self, app):
    logging.debug("Server is ready")
    self.ready = True

  def is_ready(self):
    """Check if server is ready.

    Returns:
      _type_: _description_
    """
    return self.ready
  
  def __del__(self):
    try:
      self.application.shutdown()
      self.connection.close()
    except Exception as e:
      pass


class WebSocketClient:

  def __init__(self, remote_ip, remote_port):
    self.session = None
    self.remote_ip = remote_ip
    self.remote_port = remote_port
    self.url = 'http://{}:{}/ws'.format(remote_ip, remote_port)
    self.connected = False
    self.ws = None

  async def setup(self, on_connect=None, on_reset=None):
    """Setup the connection with the remote node.

    Args:
      on_connect (callable, optional): will be called when connection is set. Defaults to None.
      on_reset (callable, optional): will be called when connection is reset. Defaults to None.
    """
    self.session = aiohttp.ClientSession()
    self.on_connect = on_connect
    self.on_reset = on_reset

  async def connect(self):
    # keep in loop till the server is ready for a connection
    while not self.connected:
      try:
        # to connect to a websocket server aiohttp.ws_connect() or aiohttp.ClientSession.ws_connect()
        # coroutines should be used.
        self.ws = await self.session.ws_connect(self.url)
        self.connected = True # escape the loop
        if self.on_connect: self.on_connect()
      except ClientConnectorError as e:
        await asyncio.sleep(.1) # wait 1s before trying to make a connection again.
        logging.info('Connect call failed ({}, {})'.format(self.remote_ip, self.remote_port))
    logging.info('Port is connected')

  def get_runners(self):
    return [self.connect()]

  async def send(self, msg: Message):
    if not self.connected:
      logging.warning('No connection available')
      return
    msg.network_send()
    data = msg.marshal()
    try:
      await self.ws.send_bytes(b'\x00' + data)
    except Exception as e:
      self.connected = False
      if self.on_reset: self.on_reset()
      raise e