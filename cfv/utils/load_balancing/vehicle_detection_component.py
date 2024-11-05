import datetime
import glob
import logging
import cv2
import time
import asyncio
import queue
from typing import List
import multiprocessing

from cfv.functions.function import Function
from cfv.net.message import Message
from cfv.utils.track import csv_writer, trace_mark
from cfv.utils.general import get_pipeline_id


# >>>>>>> test purposes
PIPELINE_ID = get_pipeline_id()
# <<<<<<< test purposes


class VehicleDetectionComponent(multiprocessing.Process):
  """VehicleDetectionComponent is multiprocessing component that runs in parallel to the load balancer and communicate through the given queue.
  """
  
  def __init__(self, config, queue: multiprocessing.JoinableQueue):
    """Initialization of new component that runs in parallel with the load balancer.

    Args:
      push (callable): that represent the function that performs the operation of the component (e.g., object classification, detection, etc.)
      lb_port (int): the input port of the parent.
        This is only used because when experiments are run with cfv.sh and stop ctl+c, the port connection opened by the load balancer (inport) is always used and the experiment can no longer be run unless we close them manually. It can be omitted if another solution is found to solve this problem.
      queue (multiprocessing.Queue): the shared queue on which the component take data from.
    """
    super(VehicleDetectionComponent, self).__init__(daemon=True)
    self.queue = queue
    self.configure(config=config)

  def configure(self, config):      
    if 'data' not in config.keys():
      raise ValueError("Missing data parameter 'data'")
    
    self.CONFIDENCE_THRESHOLD = 0.5 # minimum probability to filter weak detections
    self.full_pb_path = glob.glob('{}/*.pb'.format(config['data']))[0]
    self.labels = glob.glob('{}/*.csv'.format(config['data']))[0]
    with open(self.labels, 'r') as f:
      labels = {}
      for line in f.readlines():
        try:
          cols = line.split(',')
          labels[int(cols[0])] = cols[1]
        except Exception: pass
    self.labels = labels
    
    self.net = cv2.dnn.readNetFromTensorflow(self.full_pb_path)
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
      self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
      self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    logging.debug('You are now using {} pb and {}... labels.'.format(self.full_pb_path, self.labels))
  
  
  @trace_mark(path='cfv_log/{}/vehicle_detection'.format(PIPELINE_ID), mode='a')
  async def push(self, id, data):
    '''

    :param port:
    :return:
    '''  
    
    logging.info('[vehicle-detection] Received frame at time {} from port {}'.format(datetime.datetime.now().timestamp(), id))
    data: List[Message] = [data] if isinstance(data, Message) else data
    images = [msg.get_data() for msg in data]
    # blob = cv2.dnn.blobFromImages(images, scalefactor=1.0, size=(224, 224), swapRB=False, crop=False)
    # self.net.setInput(blob)
    # preds = self.net.forward()
    # indices = np.argmax(preds, axis=1)
    # if self.outgoing:
    #   for i, cls_idx in enumerate(indices):
    #     if preds[i][cls_idx] > self.CONFIDENCE_THRESHOLD and self.labels[cls_idx] == 'vehicles':
    #       await self.outgoing[0].push(data[i])
    
    await asyncio.sleep(.1)
    # if self.outgoing:
    #   for msg in data:
    #     if np.random.random() > .95:
    #       await self.outgoing[0].push(msg)
    
    # >>>>>>> test purposes
    # at = time.time()
    # try:
    #   qsize = self.incoming[0].get_qsize()
    #   asyncio.create_task(csv_writer(
    #     dir='cfv_log/{}/vehicle_detection'.format(PIPELINE_ID),
    #     filename='qsize_avg_response_time',
    #     mode='a',
    #     data=[{
    #       'at': at,
    #       'qsize': qsize,
    #       'start_at': msg.at,
    #       'end_at': at,
    #     } for msg in data]
    #   ))
    # except:
    #   asyncio.create_task(csv_writer(
    #     dir='cfv_log/{}/vehicle_detection'.format(PIPELINE_ID),
    #     filename='qsize_avg_response_time',
    #     mode='a',
    #     data=[{
    #       'at': at,
    #       'number_trans': msg.arguments['number_trans'],
    #       'start_at': msg.at,
    #       'end_at': at,
    #     } for msg in data]
    #   ))
    # <<<<<<< test purposes
    logging.debug('Applied vehicle detection at time {}'.format(datetime.datetime.now().timestamp()))
    
  async def _start_running(self):
    # consume from the queue till the parent get killed or finished.
    while True:
      # make sure to wait for the first message, otherwise the loop may crash the execution.
      msg = self.queue.get()
      self.queue.task_done()
      data = [msg] # append the first element.
      # as long as the batch size has not been reached and there is an element in the queue.
      try:
        while len(data) < 50:
          msg = self.queue.get(timeout=.1)
          self.queue.task_done()
          data.append(msg)
      except queue.Empty as e: print(e)
      # try:
      #   while len(data) < 50:
      #     msg = self.queue.get(block=False)
      #     data.append(msg)
      #     print(len(data), end=';')
      # except queue.Empty:
      #   print('err', end=' ')
      elapsed = time.time()
      await self.push(0, data)
      elapsed = time.time() - elapsed
      # throughput = len(data) / elapsed
      # self.processing_profilling[:-1] = self.processing_profilling[1:]
      # self.processing_profilling[-1] = throughput
      # # self.processing_profilling[-1] = len( batch) / (elapsed + 1e-10)
      # self.mp_processing_rate.value = np.nanmedian(self.processing_profilling)
    
  def run(self):
    asyncio.run(self._start_running())