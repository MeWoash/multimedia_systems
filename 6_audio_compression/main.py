import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

from scipy.interpolate import interp1d

#doc generation
from io import BytesIO
from docx import Document
from docx.shared import Inches, Cm

#sound
import sounddevice as sd
import soundfile as sf
import scipy.fftpack

INPUT_DIR = os.path.normpath(os.path.join(__file__,'..','input'))

def plotAudio(Signal: np.ndarray, Fs:int, TimeMargin=[0, 0.02], fsize = None, axs = None) -> None:
    
    if fsize is None:
        fsize = 2**10

    if axs is None:
        fig, axs = plt.subplots(2,1)
        fig.tight_layout()
    else:
        fig = axs[0].get_figure()

    xTime = np.arange(0, Signal.shape[0])/Fs
    axs[0].plot(xTime, Signal)
    axs[0].set_xlim(TimeMargin)
    axs[0].set_xlabel('s')

    yf = scipy.fftpack.fft(Signal, fsize)

    xFreq = np.arange(0, Fs/2, Fs/fsize)
    ydB = 20*np.log10( np.abs(yf[:fsize//2]))

    axs[1].plot(xFreq, ydB)
    axs[1].set_xlabel("Hz")
    axs[1].set_ylabel('dB')

    ymax = np.argmax(ydB)
    xmax = xFreq[ymax]
    axs[1].axvline(xmax, linestyle="dotted", color='r')

    return fig, axs

def quant(data, bit):
    d=float(2**bit-1)
    typ=data.dtype
    #if float
    if np.issubdtype(data.dtype, np.floating):
        m = -1.0
        n = 1.0
    #if int
    else:
        m = np.iinfo(data.dtype).min
        n = np.iinfo(data.dtype).max

    DataF = data.astype(float)
    DataF = ((DataF-m)/(n-m))
    DataF = np.round(DataF*d)
    DataF = ((DataF/d)*(n-m))+m

    # kwantyzacja na DataF
    return DataF.astype(typ)


def A_Law_encode(x, A = 87.6):
    x_out = np.copy(x)
    mask1 = np.abs(x_out)<(1/A)
    mask2 = np.logical_not(mask1)

    x_out[mask1] = (A*np.abs(x_out[mask1])) / (1+np.log(A))
    x_out[mask2] = (1+np.log(A*np.abs(x_out[mask2]))) / (1 + np.log(A))
    x_out = x_out * np.sign(x)

    return x_out


def A_Law_decode(y, A = 87.6):
    y_out = np.copy(y)
    mask1 = np.abs(y_out) < 1/(1+np.log(A))
    mask2 = np.logical_not(mask1)

    y_out[mask1] = (np.abs(y_out[mask1])*(1+np.log(A)))/(A)
    y_out[mask2] = np.exp(np.abs(y_out[mask2])*(1+np.log(A))-1) / (A)

    y_out = y_out * np.sign(y)

    return y_out


def Mu_Law_encode(x, mu=255):
    x_out = np.copy(x)

    x_out = np.log(1+mu*np.abs(x_out))/np.log(1+mu)

    x_out = x_out * np.sign(x)
    return x_out

def Mu_Law_decode(y, mu=255):
    y_out = np.copy(y)

    y_out = (1/mu)*(np.power(1+mu, np.abs(y_out))-1)

    y_out = y_out * np.sign(y)

    return y_out


if __name__ == "__main__":

    document = Document()
    for section in document.sections:
        section.top_margin = Cm(1)
        section.bottom_margin = Cm(1)
        section.left_margin = Cm(0.5)
        section.right_margin = Cm(0.5)
    document.add_heading('Lab6 - Milosz Zubala (zm49455)', 0)

    #TESTING
    xx=np.linspace(-1,1,1000)

    TEST_VECTOR = [ #x, y, encode, decode, nbits
                    [xx, xx, A_Law_encode, A_Law_decode, 8],
                    [xx, xx, Mu_Law_encode, Mu_Law_decode, 8]
                ]
    
    document.add_heading(f"Testing", 1)

    for x, y, encode, decode, bits in TEST_VECTOR:
        document.add_heading(f"Methods: {encode.__name__}, {decode.__name__}",2)

        encoded = encode(x)
        encoded_quant = quant(encode, bits)
        decoded = decode(encoded_quant)

        fig, axs = plt.subplots(1,2,figsize = (8,3))

        axs[0].set_title(f"Compression {encode.__name__}, bits: {bits}")
        axs[0].plot(x, x, label="Original")
        axs[0].plot(x,encoded, linestyle="--", label="Compressed")
        axs[0].legend()

        axs[1].set_title(f"Decompression {decode.__name__}")
        axs[1].plot(x, x, label="Original")
        axs[1].plot(x, decoded, linestyle="--", label="Decompressed")
        axs[1].legend()

        memfile = BytesIO()
        fig.savefig(memfile)
        document.add_picture(memfile)
        memfile.close()


    document.save(os.path.normpath(os.path.join(__file__,'..','output','lab6_milosz_zubala.docx')))
    plt.show()



