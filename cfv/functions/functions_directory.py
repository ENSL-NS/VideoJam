import pkgutil
import inspect
import importlib
import cfv.functions as functions

def get_function_names():
  classes = []
  for importer, modname, ispkg in pkgutil.iter_modules(functions.__path__):
    for name, obj in inspect.getmembers(importlib.import_module("{}.{}".format(functions.__name__, modname))):
      if inspect.isclass(obj):
        if name not in classes:
          classes.append(name)
  return classes


def get_class_by_name(name):
  for importer, modname, ispkg in pkgutil.iter_modules(functions.__path__):
    for classname, obj in inspect.getmembers(importlib.import_module("{}.{}".format(functions.__name__, modname))):
      if inspect.isclass(obj):
        if classname == name:
          return obj