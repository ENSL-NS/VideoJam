import aiohttp
import logging
from aiohttp import web

from cfv.net.message import Message

class HTTPServer:
  CLIENT_MAX_SIZE = 1024*1024*20

  def __init__(self, host, port, callback):
    self.host = host
    self.port = port
    self.application = None
    self.ready = False
    self.callback = callback

  async def setup(self):
    self.application = web.Application(client_max_size=HTTPServer.CLIENT_MAX_SIZE)
    self.application.add_routes([web.post('/', self.post)])
    self.application.on_startup.append(self.set_ready)

  async def post(self, request):
    # READ BODY
    logging.debug("Received new http post: {}".format(request.headers))
    json_body = await request.text()
    msg = Message()
    msg.unmarshal_json(json_body)
    await self.callback(msg)
    return web.Response(status=200)

  def get_runners(self):
    return [web._run_app(self.application, host=self.host, port=self.port, access_log=None, print=logging.debug)]

  async def set_ready(self, app):
    logging.debug("Server is ready")
    self.ready = True

  def is_ready(self):
    '''
    Check if server is ready
    
    :return:
    '''
    return self.ready


class HTTPClient:

  def __init__(self, remote_ip, remote_port):
    self.session = None
    self.remote_ip = remote_ip
    self.remote_port = remote_port
    self.url = 'http://{}:{}'.format(remote_ip, remote_port)

  async def setup(self):
    '''
    Setup the connection with the remote node

    :return:
    '''
    self.session = aiohttp.ClientSession()

  def get_runners(self):
    return []

  async def send(self, msg):
    data = msg.marshal_json()
    resp = await self.session.post(self.url, data=data, headers={'content-type': 'application/json', 'CONNECTION': 'keep-alive'})
    logging.debug(resp)

