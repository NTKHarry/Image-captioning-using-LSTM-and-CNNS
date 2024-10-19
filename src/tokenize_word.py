import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding,Flatten,Dropout,BatchNormalization, UpSampling2D,add, Bidirectional
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications import InceptionResNetV2
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import plot_model
from keras.utils import to_categorical

import gensim.downloader as api
import os
from PIL import Image
from tqdm import tqdm
import string

def tokenize_word(text):
    max_len = 50            #maximum length to pad (max word that will appear in the sentence)
    tok = Tokenizer()
    tok.fit_on_texts(text)  # Fit tokenizer on training data
    vocab_size = len(tok.word_index) + 1  # Adding 1 for the padding index
    return (max_len,vocab_size,tok)