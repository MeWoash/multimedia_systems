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

def imgToUInt8(img):
    if np.issubdtype(img.dtype, np.floating):
        img = (img*255).astype('uint8')   
    return img

def zigzag(A):
    template= n= np.array([
            [0,  1,  5,  6,  14, 15, 27, 28],
            [2,  4,  7,  13, 16, 26, 29, 42],
            [3,  8,  12, 17, 25, 30, 41, 43],
            [9,  11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
            ])
    if len(A.shape)==1:
        B=np.zeros((8,8))
        for r in range(0,8):
            for c in range(0,8):
                B[r,c]=A[template[r,c]]
    else:
        B=np.zeros((64,))
        for r in range(0,8):
            for c in range(0,8):
                B[template[r,c]]=A[r,c]
    return B

def CompressBlock(block,Q):
    block = block - 128
    block = dct2(block)
    # return vector
    return block

def DecompressBlock(vector,Q):
    ###
    # return block
    pass

## podział na bloki
# L - warstwa kompresowana
# S - wektor wyjściowy
def CompressLayer(L,Q):
    S=np.array([])
    for w in range(0,L.shape[0],8):
        for k in range(0,L.shape[1],8):
            block=L[w:(w+8),k:(k+8)]
            S=np.append(S, CompressBlock(block,Q))
    return S

## wyodrębnianie bloków z wektora 
# L - warstwa o oczekiwanym rozmiarze
# S - długi wektor zawierający skompresowane dane
def DecompressLayer(S,Q):
    L=np.zeros( (128,128) )
    for idx,i in enumerate(range(0,S.shape[0],64)):
        vector=S[i:(i+64)]
        m=L.shape[0]/8
        k=int((idx%m)*8)
        w=int((idx//m)*8)
        L[w:(w+8),k:(k+8)]=DecompressBlock(vector,Q)
    return L

def chroma_subsampling(A, Ratio):
    B = np.copy(A)
    if Ratio == "4:2:2":
        B = B[:, ::2]
    elif Ratio == "4:2:0":
        pass
    else:
        pass
    return B


def chroma_resampling(A, Ratio):
    B = np.copy(A)
    if Ratio == "4:2:2":
        B = np.repeat(B, 2)
    elif Ratio == "4:2:0":
        pass
    else:
        pass
    return B

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
    JPEG.Cb = chroma_subsampling(JPEG.Cb, JPEG.ChromaRatio)
    JPEG.Cr = chroma_subsampling(JPEG.Cr, JPEG.ChromaRatio)

    #Kompresja stratna
    JPEG.Y=CompressLayer(JPEG.Y,JPEG.QY)
    print(JPEG.Y)
    JPEG.Cr=CompressLayer(JPEG.Cr,JPEG.QC)
    JPEG.Cb=CompressLayer(JPEG.Cb,JPEG.QC)
    
    # tu dochodzi kompresja bezstratna
    #TODO

    return JPEG

def DecompressJPEG(JPEG):

    # dekompresja bezstratna
    Y=DecompressLayer(JPEG.Y, JPEG.QY)
    Cr=DecompressLayer(JPEG.Cr, JPEG.QC)
    Cb=DecompressLayer(JPEG.Cb, JPEG.QC)

    # Tu chroma resampling
    Cr = chroma_resampling(Cr, JPEG.ChromaRatio)
    Cb = chroma_resampling(Cb, JPEG.ChromaRatio)

    # tu rekonstrukcja obrazu
    YCrCb=np.dstack([Y,Cr,Cb]).clip(0,255).astype(np.uint8)

    # YCrCb -> RGB
    RGB=cv2.cvtColor(YCrCb.astype(np.uint8),cv2.COLOR_YCrCb2RGB)

    return RGB

if __name__ == "__main__":

    file = "photo1.jpg"
    file_path = os.path.join(INPUT_DIR, file)
    img = plt.imread(file_path)
    img = imgToUInt8(img)[:128, :128, :]

    compressed = CompressJPEG(img, Ratio="4:2:2")
    decompressed = DecompressJPEG(compressed)

    fig, axs = plt.subplots(1,2,figsize = (8,3))

    axs[0].imshow(img)
    # axs[1].imshow(decompressed)

    plt.show()