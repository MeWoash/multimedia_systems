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

def dithering_random(img:np.ndarray, Pallet:np.ndarray) -> np.ndarray:
    return img

def dithering_ordered(img:np.ndarray, Pallet:np.ndarray) -> np.ndarray:
    return img

def dithering_floyd_steinberg(img:np.ndarray, Pallet:np.ndarray) -> np.ndarray:
    return img


if __name__ == "__main__":
    
    columns = ["filename", "greyscale", "pallets"]
    data = [
                        ['GS_0001.tif', True, {"pallet_1_bit":n_bit_greyscale_pallet(1),
                                               "pallet_2_bit":n_bit_greyscale_pallet(2),
                                               "pallet_4_bit":n_bit_greyscale_pallet(4)}],

                        # ['GS_0002.png', True, {"pallet_1_bit":n_bit_greyscale_pallet(1),
                        #                        "pallet_2_bit":n_bit_greyscale_pallet(2),
                        #                        "pallet_4_bit":n_bit_greyscale_pallet(4)}],

                        # ['GS_0003.png', True, {"pallet_1_bit":n_bit_greyscale_pallet(1),
                        #                        "pallet_2_bit":n_bit_greyscale_pallet(2),
                        #                        "pallet_4_bit":n_bit_greyscale_pallet(4)}],

                        # ['SMALL_0009.jpg', False, {"pallet_8_bit":pallet8,
                        #                            "pallet_16_bit":pallet16}],

                        # ['SMALL_0007.jpg', False, {"pallet_8_bit":pallet8,
                        #                            "pallet_16_bit":pallet16}],

                        # ['SMALL_0006.jpg', False, {"pallet_8_bit":pallet8,
                        #                            "pallet_16_bit":pallet16}],

                        # ['SMALL_0005.jpg', False, {"pallet_8_bit":pallet8,
                        #                            "pallet_16_bit":pallet16}],
                        ]
    df = pd.DataFrame(data=data, columns=columns)



    document = Document()
    for section in document.sections:
        section.top_margin = Cm(1)
        section.bottom_margin = Cm(1)
        section.left_margin = Cm(0.5)
        section.right_margin = Cm(0.5)
    document.add_heading('Lab3 - Milosz Zubala (zm49455)', 0)

    document.add_heading(f"Kwantyzacja i dithering",1)
    for index, row in df.iterrows():
        current_image = plt.imread(os.path.join(INPUT_DIR, row['filename']))
        current_image = imgToFloat(current_image)

        document.add_heading(f"Zdjecie: {row['filename']}",2)
        if row['greyscale']==True:
            if len(current_image.shape)>2:
                current_image = calcY1(current_image)
            
            for p_name, pallet in row['pallets'].items():
                
                document.add_heading(f"Paleta {p_name}",3)

                data = [current_image]
                labels = ['original']

                data.append(kwant_colorFit(current_image, pallet))
                labels.append('quantization')

                if len(np.unique(pallet)) == 2:
                    data.append(dithering_floyd_steinberg(current_image, pallet))
                    labels.append('dith random')
                
                data.append(dithering_ordered(current_image, pallet))
                labels.append('dith ordered')

                data.append(dithering_floyd_steinberg(current_image, pallet))
                labels.append('dith floyd stteinberg')

                fig,ax = show_photos(data, labels, True)

                fig.suptitle(p_name)
                memfile = BytesIO()
                fig.savefig(memfile)
                
                document.add_picture(memfile)
                memfile.close()
            
        else:
            
            for p_name, pallet in row['pallets'].items():
                
                document.add_heading(f"Paleta {p_name}",3)

                data = [current_image]
                labels = ['original']

                data.append(kwant_colorFit(current_image, pallet))
                labels.append('quantization')

                data.append(dithering_ordered(current_image, pallet))
                labels.append('dith ordered')

                data.append(dithering_floyd_steinberg(current_image, pallet))
                labels.append('dith floyd stteinberg')

                fig,ax = show_photos(data, labels, True)

                fig.suptitle(p_name)
                memfile = BytesIO()
                fig.savefig(memfile)
                
                document.add_picture(memfile)
                memfile.close()


    document.save(os.path.normpath(os.path.join(__file__,'..','output','lab3_milosz_zubala.docx')))
    plt.show()