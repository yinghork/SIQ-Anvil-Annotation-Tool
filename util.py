import random
import os
import torch

import numpy as np

import logging

from datetime import datetime

def set_seed(my_seed):

    os.environ['PYTHONHASHSEED'] = str(my_seed)
    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  # This can slow down training


class get_logging:
    def __init__(self, path, name=''):
        self.path = path
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # create console handler and set level to info
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        # formatter = logging.Formatter('%(asctime)s %(name)-s %(levelname)-s: %(message)s',
        #                            datefmt='[%Y-%m-%d %H:%M]')
        formatter = logging.Formatter('%(asctime)s %(levelname)-s: %(message)s',
                                   datefmt='[%Y-%m-%d %H:%M:%S]')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # create debug file handler and set level to debug
        handler = logging.FileHandler(os.path.join(self.path, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"),"w")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(name)-s %(levelname)-s: %(message)s',
                                   datefmt='[%Y-%m-%d %H:%M:%S]')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, *args):
        self.logger.info(*args)

    def debug(self, *args):
        self.logger.debug(*args)

    def warning(self, *args):
        self.logger.info(*args)

    def error(self, *args):
        self.logger.error(*args)