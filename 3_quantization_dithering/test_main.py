import pytest
import numpy as np
from main import *

def test_colorFit():
    
    testVec =\
    [
        {
            "pallete":np.linspace(0,1,3).reshape(3,1),
            "pixel": 0.43,
            "retval": 0.5 
        },
        {
            "pallete":np.linspace(0,1,3).reshape(3,1),
            "pixel": 0.66,
            "retval": 0.5 
        },
        {
            "pallete":np.linspace(0,1,3).reshape(3,1),
            "pixel": 0.8,
            "retval": 1.0 
        },
        {
            "pallete":np.linspace(0,1,3).reshape(3,1),
            "pixel": 0.24,
            "retval": 0.0 
        },
        {
            "pallete":PALETT8,
            "pixel": np.array([0.25,0.25,0.5]),
            "retval": np.array([0.0,0.0,0.0]) 
        },
        {
            "pallete":PALETT16,
            "pixel": np.array([0.25,0.25,0.5]),
            "retval": np.array([0.5,0.5,0.5]) 
        }
    ]
    
    for testParams in testVec:
        assert(np.all(colorFit(testParams['pixel'],testParams['pallete']) == testParams['retval']))
