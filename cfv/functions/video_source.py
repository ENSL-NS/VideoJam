import cv2
import time
import asyncio
import logging
import datetime
from typing import Any, Dict

from cfv.functions.function import Function
from cfv.net.message import Message
from cfv.utils.general import FrameSampler


class VideoSource(Function):
  def __init__(self):
    """

    """
    Function.__init__(self)
    self.source = ''
    self.images = None
    self.frame_id = 0
    self.fps = 0
    self.total_simulation = 0
    
    
  def configure(self, config: Dict[str, Any]):
    """
    Config the video source path.

    Parameters
    ----------
    config  : dict
          content a `source` which represent the path where the source can be found.
    """
    if 'source' not in config.keys():
      raise ValueError('Missing source parameter')
    self.source = config['source']
    if 'type' not in config.keys():
      raise ValueError('Missing type (image|video) parameter')
    self.type = config['type']
    if 'frame_rate' not in config.keys():
      raise ValueError('Missing frame_rate parameter')
    self.fps = config['frame_rate']
    # duration of the experiment in minutes, default 6 minutes.
    duration = config['duration'] if 'duration' in config.keys() else 6
    self.total_simulation = self.fps * (60 * duration) # total generated frames.
    self.sampler = FrameSampler(self.fps)
    logging.info('Frame rate is set to {}'.format(self.fps))


  def push(self, id, msg):
    """

    :param port:
    :return:
    """
    logging.warning('Received frame at time {}. This should not happen'.format(datetime.datetime.now().timestamp()))


  async def run(self):
    """

    :return:
    """
    logging.debug('Starting streaming of source {}'.format(self.source))
    if not len(self.outgoing):
      logging.error('Not out ports set')
      return

    self.cap = cv2.VideoCapture(self.source)
    async def read_frame():
      logging.debug('[source] Reading next frame {}/{} from {}'.format(self.frame_id, self.total_simulation, self.source))
      task = asyncio.sleep(1 / self.fps)
      # try to read a new frame according to the frame rate.
      while True:
        if self.sampler.keep_next():
          ret, img = self.cap.read()
          msg = Message(at=time.time(), data=img)
          break
        else: # drop the next frame
          ret, img = self.cap.read()
        if not ret:
          msg = None
          break
      if img is None or self.frame_id > self.total_simulation:
        logging.warning('No frame left to read from {}.'.format(self.source))
        msg = None
      logging.debug('Read frame. ret={}.'.format(self.frame_id))
      await task
      return msg
    await asyncio.sleep(5)
    while True:
      msg = await read_frame()
      if msg is None:
        return
      await self.next(data=[msg])
      self.frame_id += 1


  def get_async_tasks(self):
    """

    :return:
    """
    return Function.get_async_tasks(self) + [self.run()]