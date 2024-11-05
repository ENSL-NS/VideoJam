from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

def total_size(o, handlers={}, verbose=False):
  """ Returns the approximate memory footprint an object and all of its contents.

  Automatically finds the contents of the following builtin containers and
  their subclasses:  tuple, list, deque, dict, set and frozenset.
  To search other containers, add handlers to iterate over their contents:

  handlers = {SomeContainerClass: iter,
              OtherContainerClass: OtherContainerClass.get_elements}

  Parameters
  ----------
  o         : any
            object to measure
  handlers  : dict
            for recursivity purpose

  Examples
  --------
  >> my_array = np.array([133, 1, 1233])
  >> convert_byte(total_size(my_array))

  with a custome object :
  class userClass(object):
    def __init__(self):
        self.integer = 1
        self.array = []

  myClass = userClass()

  def myHandler(obj):
    assert isinstance(obj, userClass)
    yield obj.integer
    yield obj.array

  total_size(myClass, {userClass: myHandler}, verbose=True)
  
  Citations
  ---------
  Link to implementation code source : https://code.activestate.com/recipes/577504/

  """
  dict_handler = lambda d: chain.from_iterable(d.items())
  all_handlers = {
    tuple: iter,
    list: iter,
    deque: iter,
    dict: dict_handler,
    set: iter,
    frozenset: iter,
  }
  all_handlers.update(handlers)     # user handlers take precedence
  seen = set()                      # track which object id's have already been seen
  default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

  def sizeof(o):
    if id(o) in seen:       # do not double count the same object
      return 0
    seen.add(id(o))
    s = getsizeof(o, default_size)

    if verbose:
      print(s, type(o), repr(o), file=stderr)

    for typ, handler in all_handlers.items():
      if isinstance(o, typ):
        s += sum(map(sizeof, handler(o)))
        break
    return s

  return sizeof(o)

def convert_byte(num, suffix="B"):
    """
    Parameters
    ----------

    Citations
    ---------
    From idris.fr
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0

    return f"{num:.1f}Yi{suffix}"


##### Example call #####

if __name__ == '__main__':
  d = dict(a=1, b=2, c=3, d=[4,5,6,7], e='a string of chars')
  print(total_size(d, verbose=True))