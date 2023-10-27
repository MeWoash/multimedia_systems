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
        }
    ]
    
    for testParams in testVec:
        assert(colorFit(testParams['pixel'],testParams['pallete']) == testParams['retval'])
