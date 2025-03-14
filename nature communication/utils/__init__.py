import sys
import os

sys.path.append(os.path.dirname(__file__))
from Config import Config
from DataIter import init_data_iter
from show_result import *
from save_loss import save_loss
