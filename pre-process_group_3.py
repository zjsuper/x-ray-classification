# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:04:19 2017

@author: Sicheng Zhou
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
image_files1 = os.listdir(IMG_DIR+'health')
image_files2 = os.listdir(IMG_DIR+'disease')

image_list = {}
for file in image_files1:
    #print (file)
    image_list[file] = cv2.imread(IMG_DIR+'health/'+file)

count = 0
    
for file,img in image_list.items():
    # crop image
    cropped_Img = img[int(512-400):int(512+400),int(512-400):int(512+400)]   

    #separate r,g,b channels
    r =cropped_Img[:,:,0]
    g =cropped_Img[:,:,1]
    b =cropped_Img[:,:,2]

    #calculate histogram for r,g,b channel
    imhist_r,bins_r = histogram(r,256,normed=True)
    imhist_g,bins_g = histogram(g,256,normed=True)
    imhist_b,bins_b = histogram(b,256,normed=True)

    #calculate cdf  for r,g,b channel
    cdf_r = imhist_r.cumsum()
    cdf_g = imhist_g.cumsum()
    cdf_b = imhist_b.cumsum()

    # cdf normalization(0-1 to 0-255)
    cdf_r = cdf_r*255/cdf_r[-1]
    cdf_g = cdf_g*255/cdf_g[-1]
    cdf_b = cdf_b*255/cdf_b[-1]

    #calculate histogram for r,g,b channel
    im_r = interp(r.flatten(),bins_r[:256],cdf_r)
    im_g = interp(g.flatten(),bins_g[:256],cdf_g)
    im_b = interp(b.flatten(),bins_b[:256],cdf_b)

    # transfrom histogram to image rgb array
    im_r = im_r.reshape([im.shape[0],im.shape[1]])
    im_g = im_g.reshape([im.shape[0],im.shape[1]])
    im_b = im_b.reshape([im.shape[0],im.shape[1]])

    cropped_Img[:,:,0] = im_r
    cropped_Img[:,:,1] = im_g
    cropped_Img[:,:,2] = im_b

    # save image
    cv2.imwrite(file, cropped_Img,[int(cv2.IMWRITE_JPEG_QUALITY),95])

    # count number
    count += 1
    print(count)
