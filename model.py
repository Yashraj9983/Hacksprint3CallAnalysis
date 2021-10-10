# IMPORT NECESSARY LIBRARIES
# import numpy as np
# from tensorflow.keras import layers, models, Model, optimizers
# import tensorflow as tf
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img
# import pandas as pd
# from sklearn.metrics import confusion_matrix
# import sys
# import matplotlib.pyplot as plt
# import itertools

# import warnings
# # ignore warnings 
# if not sys.warnoptions:
#     warnings.simplefilter("ignore")
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# from tensorflow import keras 
# from tensorflow.keras.models import Sequential, Model, model_from_json
# from tensorflow.keras.layers import Conv1D, MaxPooling2D, AveragePooling1D
# from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
# from tensorflow.keras.layers import Dense, Embedding, LSTM
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.constraints import max_norm
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import cv2
# from sklearn.utils import shuffle
# from tensorflow.python.keras import layers, models, Model, optimizers
# from tensorflow.keras import regularizers
# from tensorflow.keras import layers, models, Model, optimizers
# from keras.utils import np_utils
# from keras.utils.np_utils import to_categorical
# from random import randint
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.callbacks import ReduceLROnPlateau
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from matplotlib import pyplot

from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import decode_predictions
from tensorflow.keras.applications.vgg19 import VGG19

checkpoint_path = 'models/vgg19augm20.h5'

img_height, img_width = 224,224
conv_base = vgg19.VGG19(weights='imagenet', pooling='avg', include_top=False, input_shape = (img_width, img_height, 3))

for layer in conv_base.layers[:12]:
    layer.trainable = False

model=models.Sequential()
model.add(conv_base)
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(8,activation='softmax'))
model.summary()
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=learning_rate), metrics = ['acc'])
model.load_weights(checkpoint_path)