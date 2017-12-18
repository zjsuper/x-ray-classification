# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:04:19 2017

@author: Sicheng ZHou
"""

# import the necessary packages
#from keras.preprocessing import image as image_utils
import tensorflow as tf
import numpy as np
import argparse
import skimage.util as skut
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as skio
import cv2
import skimage.color as skcol
from PIL import Image
from pylab import *
import copy


IMG_DIR = './test2/'
#image_files1 = os.listdir(IMG_DIR+'health')
image_files2 = os.listdir(IMG_DIR+'disease')

image_list = {}
for file in image_files2:
    #print (file)
    image_list[file] = skio.imread(IMG_DIR+'disease/'+file,as_grey=True)
    
count = 0
    
for file,img in image_list.items():
    cropped_Img = img[int(512-400):int(512+400),int(512-400):int(512+400)]

    #calculate histogram
    imhist,bins = histogram(cropped_Img.flatten(),256,normed=True)

    #calculate cdf
    cdf = imhist.cumsum()

    #cdf normalization(0-1 to 0-255)
    cdf = cdf*255/cdf[-1]
    final_Img = interp(cropped_Img.flatten(),bins[:256],cdf)

    #transfrom histogram to image array
    final_Img = final_Img.reshape(cropped_Img.shape)

    # save image
    cv2.imwrite(file, final_Img,[int(cv2.IMWRITE_JPEG_QUALITY),95])
    count += 1
    print(count)

