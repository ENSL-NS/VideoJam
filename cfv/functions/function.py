import logging
from typing import List

from cfv.net.port import OutPort, InPort
from cfv.net.message import Message

class Function:
  def __init__(self):
    '''

    '''
    self.outgoing: List[OutPort] = []
    self.incoming: List[InPort] = []


  def configure(self, config):
    '''

    :param config:
    :return:
    '''
    pass


  def add_outgoing_port(self, port):
    '''

    :param port:
    :return:
    '''
    self.outgoing.append(port)


  def add_incoming_port(self, port):
    '''

    :param port:
    :return:
    '''
    self.incoming.append(port)


  async def next(self, data: List[Message]) -> None:
    """Push data to the first available node. 

    Args:
    -----
      id (_type_): _description_
      data (List[Message]): _description_
    """
    outports = [outport for outport in self.outgoing if outport.is_ready()]
    if not outports:
      return
    for msg in data:
      await outports[0].push(msg)


  async def run(self):
    '''

    :return:
    '''
    logging.warning("Nothing to run for this function")


  def get_async_tasks(self):
    '''

    :return:
    '''
    tasks = []
    for port in self.incoming + self.outgoing:
      tasks += port.get_runners()
    return tasks

