import tensorflow as tf
import numpy as np
import os
import cv2
from numpy import genfromtxt
from  tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from  tensorflow.keras.models import Model
from  tensorflow.keras.layers import BatchNormalization
from  tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
import h5py

def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    # resize the image to 96 x 96
    img1 = cv2.resize(img1, (96, 96))
    print(img1.shape)
    
    img = img1[...,::-1]
    print(img.shape)

    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    #img=img/255.0
    print(img.shape)
    x_train = np.array([img])
    print(x_train.shape)
    embedding = model.predict_on_batch(x_train)
    return embedding

# loads and resizes an image
def resize_img(image_path, save_path):
    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (96, 96))
    cv2.imwrite(image_path, img)
