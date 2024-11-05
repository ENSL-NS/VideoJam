import cv2
import glob
import logging
import datetime
import numpy as np
from typing import List

from cfv.functions.function import Function
from cfv.net.message import Message


class YoloDetection(Function):
  def __init__(self):
    '''

    '''
    Function.__init__(self)
    

  def configure(self, config):
    if 'data' not in config.keys():
      raise ValueError('Missing data parameter')
    
    self.CONFIDENCE_THRESHOLD = 0.5 # minimum probability to filter weak detections
    self.NMS_THRESHOLD = 0.4 # threshold for non maxima supression
    self.weights = glob.glob('{}/*.weights'.format(config['data']))[0]
    self.labels = glob.glob('{}/*.txt'.format(config['data']))[0]
    self.cfg = glob.glob('{}/*.cfg'.format(config['data']))[0]
    with open(self.labels, 'r') as f:
      self.labels = [c.strip() for c in f.readlines()]
    logging.debug('You are now using {} weights ,{} configs and {} labels.'.format(self.weights, self.cfg, self.labels))

    self.net = cv2.dnn.readNetFromDarknet(self.cfg, self.weights)
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
      self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
      self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

  
  async def push(self, id, data):
    '''

    :param port:
    :return:
    '''
    
    logging.debug('Received frame at time {} from port {}'.format(datetime.datetime.now().timestamp(), id))
    data: List[Message] = [data] if isinstance(data, Message) else data
    images = [msg.get_data() for msg in data]
    
    batch_boxes, batch_configs, batch_classes = self.detect(images)
    
    N = len(data)
    for i in range(N):
      for bbox, conf, cls in zip(batch_boxes[i], batch_configs[i], batch_classes[i]):
        # (x, y) = (boxes[i][0], boxes[i][1])
        # (w, h) = (boxes[i][2], boxes[i][3])
        # cv2.rectangle(data[i].get_data(), (x, y), (x + w, y + h), (0, 0, 255), 2)
        (x, y) = (int(bbox[0]), int(bbox[1]))
        (w, h) = (int(bbox[2]), int(bbox[3]))

        cv2.rectangle(data[i].get_data(), (x, y), (x + w, y + h), (0, 0, 255), 2)
        text = "{}: {:.4f}".format(self.labels[cls], conf)
        cv2.putText(
            data[i].get_data(), text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )
    logging.debug('Applied yolo at time {}'.format(datetime.datetime.now().timestamp()))
    await self.next(data=data)
    return data
    
    
  def detect(self, images):
    '''
    

    Parameters
    ----------
    image : np.ndarray
      The image to process.
    Returns
    -------
    (N, boxes, configs, classes)
    '''
    N = len(images)
    H, W, C = images[0].shape
    blob = cv2.dnn.blobFromImages(images, 1 / 255., (416, 416), swapRB=True, crop=False)

    self.net.setInputsNames(['input'])
    self.net.setInputShape('input', (N, C, H, W))
    self.net.setInput(blob)
    outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

    _boxes = [[] for _ in range(N)]
    _confidences = [[] for _ in range(N)]
    _class_idx = [[] for _ in range(N)]

    for output in outputs:
      for i, detections in enumerate(output):
        detections = detections[np.any(detections[:, 5:] > self.CONFIDENCE_THRESHOLD, axis=1)]
        # get class indeces with the highest confidence
        class_idx = np.argmax(detections[:, 5:], axis=1) + 5
        # get confidence of each detection
        confidences = detections[np.arange(class_idx.shape[0]), class_idx]
        # computes boxes for each detection
        boxes = detections[:, :4] * np.array([W, H, W, H])
        boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2. # [x,y] = center[x,y] - [w,h]/2
        _boxes[i].append(boxes)
        _confidences[i].append(confidences)
        _class_idx[i].append(class_idx)

    for i in range(N):
      _boxes[i] = np.concatenate(_boxes[i])
      _confidences[i] = np.concatenate(_confidences[i])
      _class_idx[i] = np.concatenate(_class_idx[i])

    
    # Performs non maximum suppression given boxes and corresponding scores.
    # cv.dnn.NMSBoxes(bboxes, scores, score_threshold, nms_threshold[, eta[,top_k]]) -> indices
    # bboxes	a set of bounding boxes to apply NMS.
    #   It should be a list instead of np.ndarray for opencv version 4.5.x
    # scores	a set of corresponding confidences.
    # score_threshold	a threshold used to filter boxes by score.
    # nms_threshold	a threshold used in non maximum suppression.
    # indices	the kept indices of bboxes after NMS.
    # eta	a coefficient in adaptive threshold formula: nms_thresholdi+1=eta⋅nms_thresholdi.
    # top_k	if >0, keep at most top_k picked indices.
    for i in range(N):
      # make sure that bboxes is list instead of np.ndarray for version 4.5.x (works with both types with version 4.8.x)
      bboxes = _boxes[i] # convert to type list for opencv version 4.5.x
      idxs = cv2.dnn.NMSBoxes(bboxes, _confidences[i], self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
      _boxes[i] = _boxes[i][idxs]
      _confidences[i] = _confidences[i][idxs]
      _class_idx[i] = _class_idx[i][idxs]
    return _boxes, _confidences, _class_idx