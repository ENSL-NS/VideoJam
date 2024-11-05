import json

from cfv.manager.vertex import Vertex
from cfv.manager.edge import Edge

class Graph:

  def __init__(self):
    '''

    '''
    self.__nodes = []
    self.__map_to_nodes = {}
    self.__last_vertex_id = 0


  def load_from_json(self, filename):
    '''

    :param filename:
    :return:
    '''
    with open(filename, "r") as f:
      json_data = json.loads(f.read())

      if "nodes" not in json_data.keys():
        raise ValueError("No nodes configuration in the file")

      nodes = json_data["nodes"]
      for node in nodes:
        if "name" in node.keys():
          if node["name"] in self.__map_to_nodes.keys():
            raise ValueError("Node with name {} already exists".format(node["name"]))
        new_node = Vertex()
        new_node.id = self.__last_vertex_id
        self.__last_vertex_id += 1
        new_node.configure(node)
        self.__nodes.append(new_node)
        self.__map_to_nodes[new_node.name.lower()] = new_node


  def get_node_by_id(self, id):
    '''

    :param id:
    :return:
    '''
    if len(self.__nodes) < id:
      raise ValueError("No node with id {}".format(id))
    return self.__nodes[id]


  def get_node_by_name(self, name):
    '''

    :param id:
    :return:
    '''
    if name.lower() not in self.__map_to_nodes.keys():
      raise ValueError("No node with that id")
    return self.__map_to_nodes[name.lower()]


  def iter_nodes(self):
    '''

    :return:
    '''
    return iter(self.__nodes)