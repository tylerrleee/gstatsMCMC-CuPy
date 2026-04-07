import numpy as np
import cupy as cp
import pytest
from copy import deepcopy
import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GPU.MCMC_cu import sgs as _preprocess_gpu
from gstatsMCMC.MCMC import sgs as _preprocess_cpu