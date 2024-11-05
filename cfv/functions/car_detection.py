import cv2
import logging
import datetime

from cfv.functions.function import Function
from cfv.net.message import Message


class CarDetection(Function):
  def __init__(self):
    '''

    '''
    Function.__init__(self)


  def configure(self, config):
    if "data" not in config.keys():
      raise ValueError("Missing data parameter")
    self.car_cascade = cv2.CascadeClassifier(config["data"])


  async def push(self, id, msg: Message):
    '''

    :param port:
    :return:
    '''
    logging.debug("Received frame at time {} from port {}".format(datetime.datetime.now().timestamp(), id))
    cars = self.car_cascade.detectMultiScale(msg.get_data(), 1.1, 1)

    logging.debug("Applied car detection at time {}".format(datetime.datetime.now().timestamp()))

    for (x, y, w, h) in cars:
      cv2.rectangle(msg.get_data(), (x, y), (x + w, y + h), (0, 0, 255), 2)
    await self.outgoing[0].push(msg)
