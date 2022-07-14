# -*- coding: utf-8 -*-
"""

### CNN and Attention Modules

"""

from google.colab import drive
drive.mount('/content/drive')

"""Import Libraries"""

import numpy as np
import tensorflow as tf
import random as python_random
import keras
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers.experimental import preprocessing
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout, Lambda, Add, Activation, Concatenate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications import VGG16, EfficientNetB1, Xception, EfficientNetB0
from keras.applications.mobilenet import MobileNet
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.utils.vis_utils import plot_model
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
from keras import models
import h5py
from keras import metrics
from keras import backend as K

"""Set GPU"""

import tensorflow as tf
print(tf.__version__)
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

"""Read Function"""

def set_data(train,test, batchSize, image_size):
#  np.random.seed(1234)
#  python_random.seed(1234)
#  tf.random.set_seed(1234)

 
 Image_size = [image_size,image_size]

 train_datagen= ImageDataGenerator(validation_split=0.3,rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,rescale=1./255,
                                   zoom_range=0.2,
                                   horizontal_flip=0.5, vertical_flip=0.5
                                   )

 test_datagen = ImageDataGenerator(rescale=1./255)

 #Training dataset
 train_set = train_datagen.flow_from_directory(
                train,
                target_size=Image_size,
                batch_size=batchSize, 
                color_mode="rgb",
                #shuffle=True,             
                interpolation='bicubic',
                class_mode='categorical'
                )
 #Testing dataset
 test_set= test_datagen.flow_from_directory(
              test,
              target_size=Image_size,
              color_mode = "rgb", interpolation='bicubic',
              class_mode='categorical', shuffle=False
             )
 validation_set = train_datagen.flow_from_directory(
    train, # same directory as training data
    target_size=Image_size,color_mode = "rgb",interpolation='bicubic',
    batch_size=batchSize)
 return train_set, test_set, validation_set;

"""Draw Function"""

def plot_hist(hist):
    plt.figure(3)
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
    
    plt.figure(4)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

"""Transfer Learning using VGG16"""

def create_model():
  # both spatial and channel atention
  VGGModel = VGG16(include_top=False, weights="imagenet", input_shape=(128,128,3))
  basemodel = Model(VGGModel.input, VGGModel.layers[-2].output)
  basemodel.trainable = False
  basemodel = unfreeze_model(basemodel, -3)
  # basemodel.summary()
  for i, layer in enumerate(basemodel.layers):
      print(i, layer.name, layer.trainable)
  x = basemodel.output
  # channel Attention
  se1 = layers.GlobalAveragePooling2D(name="ch_avg_pool")(x)
  se1 = layers.Reshape((1,1,512))(se1)
  se1 = layers.Dense(64,activation="swish",kernel_initializer='he_normal', use_bias=False)(se1)
  se1 = layers.Dense(512,kernel_initializer='he_normal', use_bias=False)(se1)

  se2 = layers.GlobalMaxPooling2D(name="ch_max_pool")(x)
  se2 = layers.Reshape((1,1,512))(se2)
  se2 = layers.Dense(64,activation="swish",kernel_initializer='he_normal', use_bias=False)(se2)
  se2 = layers.Dense(512,kernel_initializer='he_normal', use_bias=False)(se2)
  cbam_featurechannel = Add()([se1,se2])
  cbam_featurechannel = Activation('sigmoid')(cbam_featurechannel)
  x = layers.Multiply()([cbam_featurechannel, x])
  
  # spatial Attention
  print(x.shape)
  avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(x)
  max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(x)
  concat = Concatenate(axis=3)([avg_pool, max_pool])
  cbam_feature = layers.Conv2D(filters = 1,
					kernel_size=5,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)
  x = layers.Multiply()([cbam_feature, x])


  x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
  x = layers.Dropout(0.4)(x)
  x = layers.Dense(256, activation="relu")(x)
  x = layers.Dropout(0.3)(x)
  x = layers.Dense(128, activation="relu")(x)
  x = layers.Dropout(0.3)(x)
  outputs = layers.Dense(5, activation="softmax", name="pred")(x)
  model = tf.keras.Model(basemodel.input, outputs)
  #optimizer = tf.keras.optimizers.Adam(learning_rate=0.00015)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
  model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
  model.summary()
  return model

"""Unfreeze layers"""

def unfreeze_model(model, num_of_layers):
    for layer in model.layers[num_of_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    return model

VGGModel = VGG16(include_top=False, weights="imagenet", input_shape=(128,128,3))
  basemodel = Model(VGGModel.input, VGGModel.layers[-2].output)
  basemodel.trainable = False
  basemodel = unfreeze_model(basemodel, -3)
  # basemodel.summary()
  for i, layer in enumerate(basemodel.layers):
      print(i, layer.name, layer.trainable)
  x = basemodel.output
  # channel Attention
  se1 = layers.GlobalAveragePooling2D(name="ch_avg_pool")(x)
  se1 = layers.Reshape((1,1,512))(se1)
  se1 = layers.Dense(32,activation="relu",kernel_initializer='he_normal', use_bias=False)(se1)
  se1 = layers.Dense(512,kernel_initializer='he_normal', use_bias=False)(se1)

  se2 = layers.GlobalMaxPooling2D(name="ch_max_pool")(x)
  se2 = layers.Reshape((1,1,512))(se2)
  se2 = layers.Dense(32,activation="relu",kernel_initializer='he_normal', use_bias=False)(se2)
  se2 = layers.Dense(512,kernel_initializer='he_normal', use_bias=False)(se2)
  cbam_featurechannel = Add()([se1,se2])
  cbam_featurechannel = Activation('sigmoid')(cbam_featurechannel)
  x = layers.Multiply()([cbam_featurechannel, x])
  
  # spatial Attention
  print(x.shape)
  avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(x)
  max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(x)
  concat = Concatenate(axis=3)([avg_pool, max_pool])
  cbam_feature = layers.Conv2D(filters = 1,
					kernel_size=5,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)
  x = layers.Multiply()([cbam_feature, x])


  x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
  x = layers.Dropout(0.4)(x)
  x = layers.Dense(256, activation="relu")(x)
  x = layers.Dropout(0.3)(x)
  x = layers.Dense(128, activation="relu")(x)
  x = layers.Dropout(0.3)(x)
  outputs = layers.Dense(5, activation="softmax", name="pred")(x)
  model = tf.keras.Model(basemodel.input, outputs)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
  model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
  model.summary()

import tensorflow as tf
print(tf.__version__)

"""Training"""

batchSize = 32
epoches = 100;
image_size = 128;
train_path = '/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/dataset_update_22.06.17/ceylon_epigraphy_periods_v3/train'
# train_path = '/content/drive/classification_of_inscriptions_periods/step2_training/dataset_update_22.06.17/ceylon_epigraphy_periods_v3/train'
test_path = '/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/dataset_update_22.06.17/ceylon_epigraphy_periods_v3/test'
# test_path = '/content/drive/classification_of_inscriptions_periods/step2_training/dataset_update_22.06.17/ceylon_epigraphy_periods_v3/test'
train_set, test_set, validation_set = set_data(train_path,test_path, batchSize, image_size)
model = create_model()
checkpoint_path = "/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/dataset_update_22.06.17/model/weights_best_VGGnew.hdf5"
# checkpoint_path = "/content/drive/classification_of_inscriptions_periods/step2_training/dataset_update_22.06.17/model/weights_best_VGGnew.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  monitor='val_categorical_accuracy',mode='max',
                                                save_best_only=True,
                                                verbose=1)
history=model.fit(train_set, epochs = epoches, validation_data= validation_set, callbacks=[cp_callback], shuffle=True)
model.save('/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/dataset_update_22.06.17/model/mymodel')
# model.save('/content/drive/classification_of_inscriptions_periods/step2_training/dataset_update_22.06.17/model/mymodel')
results = model.evaluate(test_set,batch_size=32)
accuracy = results[1]
predict_labels=model.predict(test_set,batch_size=batchSize)
test_labels=test_set.classes
#plot_hist(history)
print(accuracy)

print(test_labels)
print(predict_labels.argmax(axis=1))
from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(test_labels, predict_labels.argmax(axis=1), target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']))
confusion = confusion_matrix(test_labels, predict_labels.argmax(axis=1))
print('Confusion Matrix\n')
print(confusion)

mode_loaded = keras.models.load_model('/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/dataset_update_22.06.17/model/mymodel')

batchSize = 32
epoches = 100;
image_size = 128;
train_path = '/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/dataset_update_22.06.17/ceylon_epigraphy_periods_v3/train'
test_path = '/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/dataset_update_22.06.17/ceylon_epigraphy_periods_v3/test'
train_set, test_set, validation_set = set_data(train_path,test_path, batchSize, image_size)
results = mode_loaded.evaluate(test_set,batch_size=32)
accuracy = results[1]
predict_labels=mode_loaded.predict(test_set,batch_size=batchSize)
test_labels=test_set.classes

test_labels = test_set.labels
print(test_labels)
classes = np.argmax(predict_labels, axis=1)
print(classes)
from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(test_labels, predict_labels.argmax(axis=1),  target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']))
confusion = confusion_matrix(test_labels, predict_labels.argmax(axis=1))
print('Confusion Matrix\n')
print(confusion)


import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
