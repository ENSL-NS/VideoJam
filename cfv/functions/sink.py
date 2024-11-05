from cfv.functions.function import Function
import logging
import datetime

class Sink(Function):
  def __init__(self):
    '''

    '''
    Function.__init__(self)
    self._first = True
    self._stime = None
    self._etime = None
    self._received = 0

  async def push(self, id, msg):
    '''

    :param port:
    :return:
    '''
    rec_time = datetime.datetime.now()
    logging.debug("Received frame at time {}. Discarding it".format(rec_time.timestamp()))
    self._received += 1
    if self._first:
      self._stime = rec_time
      self._first = False
    else:
      self._etime = rec_time
      fps = self._received/((self._etime - self._stime).total_seconds())
      logging.debug("Current fps: {}".format(fps))
    
