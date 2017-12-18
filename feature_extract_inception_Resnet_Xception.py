# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 22:12:56 2017

@author: Sicheng Zhou
"""

import os
import shutil
import cv2
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import h5py


def write_gap(MODEL, image_size, lambda_func=None):
    """
    Extract features from images using different pre-trained models and save it in .h5 files
    :param MODEL: the pre-trained model to use
    :param image_size: imput image dimension for pre-trained model
    :lambda_func: pre-process functions for pre-trained model
    :return: None
    """
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor

    if lambda_func:
        x = Lambda(lambda_func)(x)
    
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    gen_train = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False)

    train_generator = gen_train.flow_from_directory("train4_GrayHE", image_size, shuffle=False, 
                                              batch_size=16)
    test_generator = gen_train.flow_from_directory("test4_GrayHE", image_size, shuffle=False, 
                                             batch_size=16)
    
    train = model.predict_generator(train_generator, train_generator.nb_sample)
    test = model.predict_generator(test_generator, test_generator.nb_sample)

    with h5py.File("gap_%s.h5"%MODEL.__name__) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)
        h.create_dataset("test_label", data=test_generator.classes)

      
write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)

write_gap(Xception, (299, 299), xception.preprocess_input)

write_gap(ResNet50, (224, 224))
     