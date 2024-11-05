import csv
import json
import logging
import multiprocessing
import os
import subprocess
import time
from datetime import datetime

import psutil


def get_cpu_infos(pid: int) -> dict:
  res = subprocess.Popen(["cfv-master/cfv/utils/.bash/total_cpu_usage.py.sh", str(pid)], stdout=subprocess.PIPE)
  output = res.stdout.read().decode()
  res.kill()
  return json.loads(output)


class ProcessTracker(multiprocessing.Process):
  def __init__(self, pid: int, *, trace: dict):
    """
    Parameters
    ----------
    pid       : int
          pid of the process to track.
    trace     : dict
          dictionary that will hold trace's values.
    Examples
    --------
    trace = multiprocessing.Manager().dict()
    tracer = ProcessTracker(os.getpid(), interval=0.0, trace=trace)
    tracer.start()
    # block to process
    tracer.kill()
    # exploit data from trace 
    """
    super(ProcessTracker, self).__init__()
    self.VIRTUAL_MEMORY_TOTAL = psutil.virtual_memory().total
    self._pid = pid
    self._trace = trace
    self._trace["elapsed_time"] = 0.0
    self._trace["cpu_time"] = 0.0
    self._trace["elapsed_cpu_time"] = 0.0
    self._trace["cpu_percent"] = 0.0
    self._trace["ram"] = 0.0
    self._trace["ram_percent"] = 0.0

  def run(self):
    process = psutil.Process(self._pid)
    start = time.time()
    start_cpu_times = process.cpu_times()

    while True:
      cpu_times = process.cpu_times()
      ram = process.memory_info().rss
      self._trace["elapsed_time"] = time.time()-start # elapsed execution time
      self._trace["cpu_time"] = (cpu_times.user+cpu_times.system)/psutil.cpu_count()
      self._trace["elapsed_cpu_time"] = (cpu_times.user+cpu_times.system)-(start_cpu_times.user+start_cpu_times.system)
      self._trace["cpu_percent"] = ((cpu_times.user+cpu_times.system)/psutil.cpu_count())*100/(time.time()-process.create_time())
      self._trace["ram"] = ram
      self._trace["ram_percent"] = ram*100/self.VIRTUAL_MEMORY_TOTAL



class ProcessTrackerFile(multiprocessing.Process):
  def __init__(self, pid: int, *, dir="./", interval=.1):
    """
    Parameters
    ----------
    pid       : int
          pid of the process to track
    dir      : str
          the directory where to save the file.
    interval  : floa
          When `interval` is 0.0 or None (default) compares process times to system CPU times elapsed since last call, returning immediately (non-blocing). That means that the first time this is called it will return a meaningful 0.0 value.

          When `interval` is > 0.0 compares process times to system CPU times elapsed before and after the interval (blocing).

          In this case is recommended for accuracy that this function be called with at least 0.1 seconds between calls.
    """
    super(ProcessTrackerFile, self).__init__()
    self.VIRTUAL_MEMORY_TOTAL = psutil.virtual_memory().total
    self._pid = pid
    self._interval = interval
    if not os.path.exists(dir):
      os.makedirs(dir)
    self._filename = f"{dir}/{self.__class__.__name__}_interval={self._interval}.csv"
    logging.debug("Parent process to track is {}".format(self._pid))

  def run(self):
    logging.debug("Start tracking prodess' ip={}".format(self._pid))
    process = psutil.Process(self._pid)

    with open(self._filename, "w") as file:
      headers = ("at", "cpu_time", "cpu_percent", "elapsed_time", "ram", "ram_percent")
      # open the csv writer
      writer = csv.DictWriter(file, delimiter=",", lineterminator="\n", fieldnames=headers)
      writer.writeheader()
      while True:
        cpu_times = process.cpu_times()
        ram = process.memory_info().rss # bytes
        ram_percent = ram / self.VIRTUAL_MEMORY_TOTAL * 100
        cpu_percent=process.cpu_percent(interval=self._interval)
        writer.writerow({
          "at": datetime.now(),
          "cpu_time": cpu_times.usrcpu_times.system,         # user time plus system time
          "elapsed_time": time.time() - process.create_time(),    # elapsed time, time since the process has been created to now
          "cpu_percent": cpu_percent,
          "ram": ram,
          "ram_percent": ram_percent,
        })



