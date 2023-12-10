import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

import cv2

#doc generation
from io import BytesIO
from docx import Document
from docx.shared import Inches, Cm

import scipy.fftpack

INPUT_DIR = os.path.normpath(os.path.join(__file__,'..','input'))

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a.astype(float), axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.astype(float), axis=0 , norm='ortho'), axis=1 , norm='ortho')

class ver1:
    Y=np.array([])
    Cb=np.array([])
    Cr=np.array([])
    ChromaRatio="4:4:4"
    QY=np.ones((8,8))
    QC=np.ones((8,8))
    shape=(0,0,3)

class ver2:
    def __init__(self, Y, Cb, Cr, OGShape, Ratio="4:4:4", QY=np.ones((8,8)), QC=np.ones((8,8))):
        self.shape = OGShape
        self.Y=Y
        self.Cb=Cb
        self.Cr=Cr
        self.ChromaRatio=Ratio
        self.QY=QY
        self.QC=QC


def CompressJPEG(RGB, Ratio="4:4:4", QY=np.ones((8,8)), QC=np.ones((8,8))):
    # RGB -> YCrCb
    YCrCb=cv2.cvtColor(RGB,cv2.COLOR_RGB2YCrCb).astype(int)
    # JPEG = verX(...); zapisać dane z wejścia do kalsy
    JPEG = ver2(Y=YCrCb[:,:,0],
                Cb=YCrCb[:,:,1],
                Cr=YCrCb[:,:,2],
                OGShape=YCrCb.shape,
                Ratio=Ratio,
                QY=QY,
                QC=QC)
    # Tu chroma subsampling
    # JPEG.Y=CompressLayer(JPEG.Y,JPEG.QY)
    # JPEG.Cr=CompressLayer(JPEG.Cr,JPEG.QC)
    # JPEG.Cb=CompressLayer(JPEG.Cb,JPEG.QC)
    # tu dochodzi kompresja bezstratna
    # return JPEG
    return None

def DecompressJPEG(JPEG):
    # dekompresja bezstratna
    # Y=DecompressLayer(JPEG.Y,JPEG.QY)
    # Cr=DecompressLayer(JPEG.Cr,JPEG.QC)
    # Cb=DecompressLayer(JPEG.Cb,JPEG.QC)
    # Tu chroma resampling
    # tu rekonstrukcja obrazu

    # YCrCb -> RGB
    # RGB=cv2.cvtColor(YCrCb.astype(np.uint8),cv2.COLOR_YCrCb2RGB)
    # return RGB
    return None

if __name__ == "__main__":
    pass