import asyncio
import logging
import numpy as np
import multiprocessing
import aiohttp
from aiohttp import web
import logging
import asyncio

from cfv.net.message import Message
from cfv.utils.general import kill_port
from cfv.utils.general import get_pipeline_id

# >>>>>>> test purposes
PIPELINE_ID = get_pipeline_id()
# <<<<<<< test purposes

class Input(multiprocessing.Process):
  """Socket is multiprocessing component that runs in parallel to the load balancer and communicate through the given queue.
  """
  
  def __init__(self,
               config,
               queue: multiprocessing.JoinableQueue,
               signal: multiprocessing.JoinableQueue,
               qsize: multiprocessing.Value,
               num_received: multiprocessing.Value,
               ):
    """Initialization of new component that runs in parallel with the load balancer.

    Args:
      push (callable): that represent the function that performs the operation of the component (e.g., object classification, detection, etc.)
      lb_port (int): the input port of the parent.
        This is only used because when experiments are run with cfv.sh and stop ctl+c, the port connection opened by the load balancer (inport) is always used and the experiment can no longer be run unless we close them manually. It can be omitted if another solution is found to solve this problem.
      queue (multiprocessing.Queue): the shared queue on which the component take data from.
    """
    super(Input, self).__init__()
    self.queue = queue
    self.signal = signal
    self.qsize = qsize
    self.num_received = num_received
    self.config(config=config)
    

  def config(self, config: dict):
    '''

    '''
    self.canceled = False
    self.connected = False
    self.marshalling = 'pickle'
    self.protocol = 'websocket'
    self.host = config['host']
    self.port = config['port']
    kill_port(self.port) # try to kille the process that is using the port if exist.

  async def websocket_handler(self, request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    async for websocket_message in ws:
      if websocket_message.type == aiohttp.WSMsgType.TEXT:
        if websocket_message.data[0] == 'c':
          await ws.close()
        elif websocket_message.data[0] == 'f':
          msg = Message()
          msg.unmarshal_json(websocket_message.data[1:])
          self.queue.put(msg)
          self.num_received.value += 1
      elif websocket_message.type == aiohttp.WSMsgType.BINARY:
        if websocket_message.data[0] == b'\x01'[0]:
          await ws.close()
        elif websocket_message.data[0] == b'\x00'[0]:
          msg = Message()
          msg.unmarshal_pickle(websocket_message.data[1:])
          self.queue.put(msg)
          self.num_received.value += 1
      elif websocket_message.type == aiohttp.WSMsgType.ERROR:
        print('ws connection closed with exception %s'%ws.exception())
      self.qsize.value = self.queue.qsize()
    logging.debug('websocket connection closed')
    return ws

  
  async def _start_running(self):
    self.app = web.Application()
    self.app.add_routes([web.get('/ws', self.websocket_handler)])
    await web._run_app(self.app, host=self.host, port=self.port, access_log=None, print=logging.info)

    
  def run(self):
    asyncio.run(self._start_running())
    self.queue.join()