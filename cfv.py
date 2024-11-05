#!/usr/bin/python3.7

import os
import sys
import time
import argparse
import asyncio
import logging
from cfv.manager.local_manager import LocalManager

logging.basicConfig(
  format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
  level=logging.DEBUG,
  datefmt='%H:%M:%S',
  stream=sys.stderr,
)



def main():
  parser = argparse.ArgumentParser(description='Execute video analytics processing pipeline.')
  parser.add_argument('config', metavar='config', type=str, help='pipeline configuration file')
  parser.add_argument('-L', '--log-dir', help='Log directory')
  parser.add_argument('-D', '--debug', action='store_true', help='Print logs at debug level (default info)')
  parser.add_argument('-I', '--info', action='store_true', help='Print logs at info level (default info)')
  parser.add_argument('-W', '--warn', action='store_true', help='Print logs at warn level (default info)')
  parser.add_argument('-E', '--error', action='store_true', help='Print logs at error level (default info)')

  args = vars(parser.parse_args())

  if args['log_dir']:
    os.environ['LOG_DIR'] = args['log_dir']
  else:
    os.environ['LOG_DIR'] = ''
    

  if args['debug']:
    logging.getLogger().setLevel(logging.DEBUG)
  elif args['info']:
    logging.getLogger().setLevel(logging.INFO)
  elif args['warn']:
    logging.getLogger().setLevel(logging.WARN)
  elif args['error']:
    logging.getLogger().setLevel(logging.ERROR)  

  local_manager = LocalManager()
  local_manager.process_config_json(args['config'])
  # newer version
  asyncio.run(local_manager.start())
  
  # less upgrated version of asyncio
  # loop = asyncio.get_event_loop()
  # loop.run_until_complete(local_manager.start())


if __name__== '__main__':
  main()