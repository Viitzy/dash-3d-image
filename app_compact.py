import dash
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash import html
from dash import dcc
import plotly.graph_objects as go
from skimage import data, img_as_ubyte, segmentation, measure, color
from dash_canvas.utils import array_to_data_url
import plotly.graph_objects as go
# import plot_common
# import image_utils
import numpy as np
from nilearn import image
import nibabel as nib
import plotly.express as px
# import shape_utils
from sys import exit
import io
import base64
import skimage
import time
import os
from skimage import io, segmentation
import subprocess
from PIL import Image
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2

img = image.load_img("assets/BraTS19_2013_10_1_flair.nii")
img = img.get_fdata().transpose(2, 0, 1)[::-1].astype("float")
img = img_as_ubyte((img - img.min()) / (img.max() - img.min()))

num_slices = img.shape[0]
seg = []
segb = []
# Processar cada fatia com seu algoritmo de segmentação
for i in range(num_slices):
    slice = img[i, :, :]
    # filename = f'top_view/top_view_{i}.pgm'
    # io.imsave(filename, slice)
    
    # input_path = f'top_view/top_view_{i}.pgm'
    # output_path = f'top_view/output_{i}.pgm'
    # result = subprocess.run(
    #     ["./SICLE/bin/RunSICLE", "--img", input_path, "--out", output_path],
    #     check=True,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE
    # )    
    
    filename = f'top_view/output_{i}.pgm'
    processed_image = io.imread(filename)
    # Carregar a imagem original e a imagem segmentada
    original_img = cv2.imread(f'top_view/top_view_{i}.pgm', cv2.IMREAD_GRAYSCALE)
    segmented_img = cv2.imread(f'top_view/output_{i}.pgm', cv2.IMREAD_GRAYSCALE)
    # Calcular a máscara de segmentação
    # Aqui, você pode ajustar a forma de calcular a diferença dependendo de como a segmentação foi realizada
    # Neste exemplo, simplesmente subtraímos a imagem segmentada da original
    # mask = cv2.absdiff(original_img, segmented_img)
    # Limiar para criar uma máscara binária
    _, mask = cv2.threshold(original_img, 1, 255, cv2.THRESH_BINARY)
    # Usar a máscara para limpar a imagem segmentada
    # Aqui estamos assumindo que a área de interesse é branca na imagem segmentada
    cleaned_img = cv2.bitwise_and(segmented_img, mask)
    borders = segmentation.find_boundaries(cleaned_img, mode='thick')
    # borders_uint8 = (borders * 255).astype(np.uint8)
    seg.append(cleaned_img)
    segb.append(borders)

seg = np.array(seg)
segb = np.array(segb)

np.save('seg_SICLE.npy', seg)
np.save('segb_SICLE.npy', segb)

# segl = image_utils.label_to_colors(
#     segb, colormap=["#000000", "#E48F72"], alpha=[0, 128], color_class_offset=0
# )