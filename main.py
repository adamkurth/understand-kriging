import numpy as np
import pandas as pd
from math import *

nugget = 2.5
sill = 7.5
rang = 10

def semivariance(nug, sill, ran, h):
    sv = nug + sill*(3/2*h/ran - 1/2*(h/ran)**3)
    if sv.shape[0] > 1: 
        
        onescol = np.ones(sv.shape[0])
        sv = np.insert(sv, sv.shape[1], onescol, axis=1)
        onesrow = np.ones(sv.shape[1])
        sv = np.insert(sv, sv.shape[0], onesrow, axis=0)
        sv[sv.shape[0]-1, sv.shape[1]-1] = 0
    else: 
        onescol = np.ones(sv.shape[0])
        sv = np.insert(sv, sv.shape[1], onescol, axis=1)
        onesrow = np.ones(sv.shape[1])
        sv = np.insert(sv, sv.shape[0], onesrow, axis=0)
        sv[sv.shape[0]-1, sv.shape[1]-1] = 0
    return sv