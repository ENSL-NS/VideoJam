from cfv.manager.edge import Edge
from cfv.functions.functions_directory import get_function_names
import inspect

class Vertex:
  def __init__(self):
    '''

    '''
    self.id = 0
    self.name = ""
    self.type = ""
    self.in_ports = {}
    self.out_ports = {}
    self.parameters = {}
    self.last_edge_id = 0


  def configure(self, config):
    '''

    :param config:
    :return:
    '''
    name = str(self.id)
    if "name" in config.keys():
      name = config["name"]
    self.set_name(name)

    if "type" not in config.keys():
      raise ValueError("Function type is required for all nodes")
    self.set_type(config["type"])

    if "parameters" in config.keys():
      self.parameters = config["parameters"]

    if "in_ports" in config.keys():
      for port in config["in_ports"]:
        edge = Edge()
        edge.configure(port)
        print("Adding port {} to node {}".format(self.last_edge_id, self.name))
        self.add_in_port(self.last_edge_id, edge)
        self.last_edge_id += 1

    if "out_ports" in config.keys():
      for port in config["out_ports"]:
        edge = Edge()
        edge.configure(port)
        self.add_out_port(self.last_edge_id, edge)
        self.last_edge_id += 1


  def set_name(self, name):
    '''

    :param name:
    :return:
    '''
    self.name = name


  def get_name(self):
    '''

    :return:
    '''
    return self.name


  def set_type(self, t):
    '''

    '''
    if t not in get_function_names():
      # raise ValueError("Function type {} is not a valid function type. Valid function types are {}".format(t, inspect.getmembers(functions)))
      raise ValueError("Function type {} is not a valid function type".format(t))
    else:
      self.type = t


  def set_parameter(self, key, value):
    '''

    :param key:
    :param value:
    :return:
    '''
    self.parameters[key] = value

  def get_parameter(self, key):
    '''

    :param key:
    :return:
    '''
    if key in self.parameters:
      return self.parameters[key]
    else:
      return None


  def add_in_port(self, id, edge):
    '''

    :param id:
    :param edge:
    :return:
    '''
    self.in_ports[id] = edge


  def add_out_port(self, id, edge):
    '''

    :param id:
    :param edge:
    :return:
    '''
    self.out_ports[id] = edge