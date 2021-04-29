import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
from matplotlib import cm
import time
import math as m


"""
This file contains general use functions
"""


def round_sig(x, sig=2):
    return round(x, sig - int(m.floor(m.log10(abs(x)))) - 1)
