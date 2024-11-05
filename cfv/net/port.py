class InPort():
  '''
  General definition of incoming port
  '''
  def __init__(self, id, callback):
    '''

    :param id: Port identifier
    :param callback: Callback function to call when receiving a new message
    '''
    self.id = id
    self.callback = callback

  def get_runners(self):
    '''
    Get the tasks to run in asynchronous mode
    :return: []
    '''
    return []

  def is_ready(self):
    '''
    Check if a port is ready
    '''
    return True


class OutPort():
  '''
  General definition of outgoing port
  '''
  def __init__(self, id, nextPort):
    '''

    :param id: Port identifier
    :param nextPort: InPort to send messages to
    '''
    self.id = id
    self.nextPort = nextPort

  def get_runners(self):
    '''
    Get the tasks to run in asynchronous mode
    :return: []
    '''
    return []
  
  def on_reset(self):
    """Called when connection is reset.
    """
    self.connected = False
  
  def on_connect(self):
    """Called when connection is set.
    """
    self.connected = True

  def is_ready(self):
    '''
    Check if a port is ready
    '''
    return self.connected

