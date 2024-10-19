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

def create_embed_matrix(tok,vocab_size):
    glove_file_path = r'D:\tfolder\codingFile\AIlearning\projects\Image_captioning\glove.6B.200d.txt'
    embeding_dim = 200
    embed_matrix = np.zeros((vocab_size,embeding_dim))
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Split the line into components
            values = line.split()  # Split by whitespace
            word = values[0]  # The first element is the word
            vector = np.array(values[1:], dtype=float)  # The remaining elements are the vector, converted to float
            if word in tok.word_index:
                id = tok.word_index.get(word)
                embed_matrix[id] = vector
    return embed_matrix

def datagen(img_names, caption_map,feature_map,tok,max_len, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n=0
    while 1: 
        for name in img_names:
            n +=1
            captions = caption_map[name]
            for cap in captions:
                seq = tok.texts_to_sequences([cap])[0]
                for i in range(1,len(seq)):
                    seq_in = seq[:i]
                    seq_out = seq[i]
                    seq_in = sequence.pad_sequences([seq_in], maxlen=max_len,padding='post')[0]
                    seq_out = to_categorical([seq_out], num_classes=vocab_size)[0]
                    X1.append(feature_map[name][0])
                    X2.append(seq_in)
                    y.append(seq_out)
            if n == batch_size:
                yield (np.array(X1), np.array(X2)), np.array(y)
                X1, X2, y = list(), list(), list()
                n = 0

def cap_gen_model(vocab_size, embeding_dim, max_len, embed_matrix):
    input1 = Input(name='img_feature_input', shape=(1536,))
    fe1 = Dropout(0.4)(input1)
    fe2 = Dense(256, activation='relu')(fe1)

    input2 = Input(name='text_feature_input', shape=(max_len,))
    se1 = Embedding(vocab_size, embeding_dim, input_length=max_len, weights=[embed_matrix], trainable=False)(input2)

    # First Bidirectional LSTM layer
    se2 = Bidirectional(LSTM(256, return_sequences=True))(se1)
    se3 = Dropout(0.4)(se2)

    # Second LSTM layer (can be normal LSTM or another Bidirectional LSTM)
    se4 = LSTM(256)(se3)  # You can choose to make this one Bidirectional as well
    se5 = Dropout(0.4)(se4)

    se6 = Dense(256, activation='relu')(se5)

    # Combine with image features
    combine = add([fe2, se6])
    layer = Dense(256, activation='relu')(combine)
    output = Dense(vocab_size, activation='softmax')(layer)

    model = Model(inputs=[input1, input2], outputs=output)
    return model

