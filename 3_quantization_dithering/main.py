import numpy as np
import os
import sys
import matplotlib.pyplot as plt

#doc generation
from io import BytesIO
from docx import Document
from docx.shared import Inches

INPUT_DIR = os.path.normpath(os.path.join(__file__,'..','input'))


def colorFit(pixel: np.ndarray, Pallet:np.ndarray):
    if not isinstance(Pallet, np.ndarray):
        raise Exception("Pallet must be ndarray (N,M)")
    elif isinstance(pixel, np.ndarray):
        if pixel.shape[1] != Pallet.shape[1]:
            raise Exception("Pallet and pixel must have shape (X,N) and (Y,N)")
    elif isinstance(pixel, float):
        if Pallet.shape[1]!=1:
            raise Exception("Pallet must have shape of (N,1)")
    
    closestColor = Pallet[np.argmin(np.linalg.norm(np.subtract(pixel, Pallet),axis=1))].squeeze()
    return closestColor

if __name__ == "__main__":
    p=np.linspace(0,1,3).reshape(3,1)
    a=0.43
    print(colorFit(a,p))