import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import Sequential, Model, load_model
import os


def get_feature(image_folder,image_model):
    feature_map = {}
    for name in tqdm(os.listdir(image_folder)):
        img_path = image_folder + '/' + name
        img = load_img(img_path, target_size = (299,299))
        img = img_to_array(img)
        img = img.reshape((1,img.shape[0], img.shape[1], img.shape[2]))
        if(check == 0):
            print(img.shape)
            check+=1
        image = preprocess_input(img)
        feature = image_model.predict(image,verbose = 0)
        feature_map[name] = feature
    return feature_map