import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

#doc generation
from io import BytesIO
from docx import Document
from docx.shared import Inches, Cm

INPUT_DIR = os.path.normpath(os.path.join(__file__,'..','input'))

pallet8 = np.array(
[
    [0.0, 0.0, 0.0,],
    [0.0, 0.0, 1.0,],
    [0.0, 1.0, 0.0,],
    [0.0, 1.0, 1.0,],
    [1.0, 0.0, 0.0,],
    [1.0, 0.0, 1.0,],
    [1.0, 1.0, 0.0,],
    [1.0, 1.0, 1.0,],
])

pallet16 =  np.array(
[
    [0.0, 0.0, 0.0,], 
    [0.0, 1.0, 1.0,],
    [0.0, 0.0, 1.0,],
    [1.0, 0.0, 1.0,],
    [0.0, 0.5, 0.0,], 
    [0.5, 0.5, 0.5,],
    [0.0, 1.0, 0.0,],
    [0.5, 0.0, 0.0,],
    [0.0, 0.0, 0.5,],
    [0.5, 0.5, 0.0,],
    [0.5, 0.0, 0.5,],
    [1.0, 0.0, 0.0,],
    [0.75, 0.75, 0.75,],
    [0.0, 0.5, 0.5,],
    [1.0, 1.0, 1.0,], 
    [1.0, 1.0, 0.0,]
])

def get_text_width(document):
    """
    Returns the text width in mm.
    """
    section = document.sections[0]
    return (section.page_width - section.left_margin - section.right_margin) / 36000

def imgToFloat(img):
    if np.issubdtype(img.dtype, np.unsignedinteger):
        img = img/255.0
    return img    

def calcY1(img):
    Y1 = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    return Y1

def colorFit(pixel: np.ndarray, Pallet:np.ndarray) -> np.ndarray:
    closestColor = Pallet[np.argmin(np.linalg.norm(pixel - Pallet, axis=1))].squeeze()
    # print(f"Pixel {pixel} -> {closestColor} in palett: {Pallet}")
    return closestColor

def kwant_colorFit(img:np.ndarray, Pallet:np.ndarray) -> np.ndarray:
    out_img = img.copy()

    img_shape = out_img.shape
    for w in range(img_shape[0]):
        for k in range(img_shape[1]):
            out_img[w,k] = colorFit(img[w,k], Pallet)

    return out_img


def show_photos(photo_data: list[np.ndarray],
                photo_title: list[str],
                greyscale: bool,
                figsize = (8,3)) -> [plt.Figure, plt.Axes]:
    assert(len(photo_data)==len(photo_title))

    n_axes = len(photo_data)
    fig, axs = plt.subplots(1,len(photo_data), figsize=figsize, layout = 'compressed')
    if n_axes==1:
         axs = [axs]
    for i in range(n_axes):
        if greyscale:
            axs[i].imshow(photo_data[i], cmap=plt.cm.gray)
        else:
            axs[i].imshow(photo_data[i])
        axs[i].set_title(photo_title[i])
    return fig, axs

def n_bit_greyscale_pallet(n: int=2):
    bits = 2**n
    return np.linspace(0,1,bits).reshape(bits,1)

images = [
    #color
    'SMALL_0001.tif', 
    'SMALL_0004.jpg',
    'SMALL_0005.jpg',
    'SMALL_0006.jpg',
    'SMALL_0007.jpg',
    'SMALL_0008.jpg',
    'SMALL_0009.jpg',
    'SMALL_0010.jpg',
    # greyscale
    'GS_0001.tif',
    'GS_0002.png',
    'GS_0003.png',
    'SMALL_0002.png',
    'SMALL_0003.png',
]


if __name__ == "__main__":
    
    columns_quantization = ["filename", "greyscale"]
    data_quantization = [
                        ['GS_0001.tif', True],
                        ['GS_0002.png', True],
                        # ['GS_0003.png', True],
                        # ['SMALL_0009.jpg', False],
                        # ['SMALL_0007.jpg', False],
                        # ['SMALL_0006.jpg', False],
                        # ['SMALL_0005.jpg', False]
                        ]
    df_quantization = pd.DataFrame(data=data_quantization, columns=columns_quantization)

    document = Document()
    for section in document.sections:
        section.top_margin = Cm(1)
        section.bottom_margin = Cm(1)
        section.left_margin = Cm(0.5)
        section.right_margin = Cm(0.5)
    document.add_heading('Lab3 - Milosz Zubala (zm49455)', 0)

    document.add_heading(f"Kwantyzacja",1)
    for index, row in df_quantization.iterrows():
        current_image = plt.imread(os.path.join(INPUT_DIR, row['filename']))
        current_image = imgToFloat(current_image)

        if row['greyscale']==True:
            if len(current_image.shape)>2:
                current_image = calcY1(current_image)
            p1 = kwant_colorFit(current_image, n_bit_greyscale_pallet(1))
            p2 = kwant_colorFit(current_image, n_bit_greyscale_pallet(2))
            p4 = kwant_colorFit(current_image, n_bit_greyscale_pallet(4))
            fig,ax = show_photos([current_image, p1, p2, p4],
                        ['original', 'Pallet 1-bit','Pallet 2-bit','Pallet 4-bit'],
                        True)
        else:
            p8 = kwant_colorFit(current_image, pallet8)
            p16 = kwant_colorFit(current_image, pallet16)
            fig,ax = show_photos([current_image, p8, p16],
                        ['original','Pallet 8-bit','Pallet 16-bit'],
                        False)
        fig.suptitle(f"Quantization")

        memfile = BytesIO()
        fig.savefig(memfile)
        document.add_heading(f"Zdjecie: {row['filename']}",2)
        document.add_picture(memfile)
        memfile.close()
        

    document.save(os.path.normpath(os.path.join(__file__,'..','output','lab3_milosz_zubala.docx')))
    # plt.show()