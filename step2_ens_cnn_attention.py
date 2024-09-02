import tensorflow as tf
print(tf.__version__)
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import numpy as np
import random as python_random
import keras
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Input, Average
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications import VGG16, Xception
from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras import models
import h5py
from tensorflow.keras import metrics
import pandas as pd
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras import backend as K
from collections import Counter
import cv2

from google.colab import drive
drive.mount('/content/drive')

def set_data(train, test, batchSize, image_size):
    Image_size = [image_size, image_size]

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=0.2,
        vertical_flip=0.2,
    )

    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_set = train_datagen.flow_from_directory(
        train,
        target_size=Image_size,
        batch_size=batchSize,
        color_mode="rgb",
        interpolation='bicubic',
        class_mode='categorical'
    )

    validation_set = train_datagen.flow_from_directory(
        train,
        target_size=Image_size,
        batch_size=batchSize,
        color_mode="rgb",
        interpolation='bicubic',
        class_mode='categorical'
    )

    test_set = test_datagen.flow_from_directory(
        test,
        target_size=Image_size,
        color_mode="rgb",
        interpolation='bicubic',
        class_mode='categorical',
        shuffle=False
    )

    return train_set, validation_set, test_set

def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1,1,channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(input_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)

    return multiply([input_feature, cbam_feature])

def create_vgg16_model():
    vgg_input = Input(shape=(128, 128, 3), name='vgg_input')
    vgg_base = VGG16(weights='imagenet', include_top=False, input_tensor=vgg_input)
    vgg_base._name = 'vgg'
    for layer in vgg_base.layers:
        layer._name = 'vgg_' + layer.name
    vgg_base.trainable = False
    vgg_base = unfreeze_model(vgg_base, -5)
    vgg_output = vgg_base.output
    vgg_output = channel_attention(vgg_output)
    vgg_output = spatial_attention(vgg_output)
    vgg_output = GlobalAveragePooling2D()(vgg_output)

    vgg_output = layers.Dropout(0.4)(vgg_output)
    vgg_output = layers.Dense(256, activation="relu")(vgg_output)
    vgg_output = layers.Dropout(0.3)(vgg_output)
    vgg_output = layers.Dense(128, activation="relu")(vgg_output)
    vgg_output = layers.Dropout(0.3)(vgg_output)
    vgg_output = Dense(5, activation='softmax', name='vgg_output')(vgg_output)

    vgg_model = Model(inputs=vgg_input, outputs=vgg_output)
    vgg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return vgg_model

def create_xception_model():
    xception_input = Input(shape=(128, 128, 3), name='xception_input')
    xception_base = Xception(weights='imagenet', include_top=False, input_tensor=xception_input)
    xception_base._name = 'xception'
    for layer in xception_base.layers:
        layer._name = 'xception_' + layer.name
    xception_base.trainable = False
    xception_base = unfreeze_model(xception_base, -10)
    xception_output = xception_base.output
    xception_output = channel_attention(xception_output)
    xception_output = spatial_attention(xception_output)
    xception_output = GlobalAveragePooling2D()(xception_output)

    xception_output = layers.Dropout(0.4)(xception_output)
    xception_output = layers.Dense(256, activation="relu")(xception_output)
    xception_output = layers.Dropout(0.3)(xception_output)
    xception_output = layers.Dense(128, activation="relu")(xception_output)
    xception_output = layers.Dropout(0.3)(xception_output)
    xception_output = Dense(5, activation='softmax', name='xception_output')(xception_output)

    xception_model = Model(inputs=xception_input, outputs=xception_output)
    xception_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return xception_model

def create_convnext_model():
    convnext_input = Input(shape=(128, 128, 3), name='convnext_input')
    convnext_base = ConvNeXtBase(weights='imagenet', include_top=False, input_tensor=convnext_input)
    convnext_base._name = 'convnext'
    for layer in convnext_base.layers:
        layer._name = 'convnext_' + layer.name
    convnext_base.trainable = False
    convnext_base = unfreeze_model(convnext_base, -10)
    convnext_output = convnext_base.output
    convnext_output = channel_attention(convnext_output)
    convnext_output = spatial_attention(convnext_output)
    convnext_output = GlobalAveragePooling2D()(convnext_output)

    convnext_output = layers.Dropout(0.4)(convnext_output)
    convnext_output = layers.Dense(256, activation="relu")(convnext_output)
    convnext_output = layers.Dropout(0.3)(convnext_output)
    convnext_output = layers.Dense(128, activation="relu")(convnext_output)
    convnext_output = layers.Dropout(0.3)(convnext_output)
    convnext_output = Dense(5, activation='softmax', name='convnext_output')(convnext_output)

    convnext_model = Model(inputs=convnext_input, outputs=convnext_output)
    convnext_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return convnext_model

def ensemble_voting(models, img_normalized):
    predictions = []
    for model in models:
        prediction = np.argmax(model.predict(np.array([img_normalized])))
        predictions.append(prediction)

    # Majority voting
    vote_counts = Counter(predictions)
    majority_vote = vote_counts.most_common(1)[0][0]
    return majority_vote

def evaluate_ensemble(models, seg_test_folders):
    true_value = []
    ensemble_pred = []

    for folder in os.listdir(seg_test_folders):
        test_image_ids = os.listdir(os.path.join(seg_test_folders, folder))
        for image_id in test_image_ids[:int(len(test_image_ids))]:
            path = os.path.join(seg_test_folders, folder, image_id)
            true_value.append(test_set.class_indices[folder])
            img = cv2.resize(cv2.imread(path), (128, 128))
            img_normalized = img / 255

            ensemble_prediction = ensemble_voting(models, img_normalized)
            ensemble_pred.append(ensemble_prediction)

    return true_value, ensemble_pred

def unfreeze_model(model, num_of_layers):
    for layer in model.layers[num_of_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    return model

batchSize = 32
image_size = 128
train_path = '/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/character_eras/train'
test_path = '/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/character_eras/test'
train_set, validation_set, test_set = set_data(train_path, test_path, batchSize, image_size)

initial_learning_rate = 0.0002
max_epochs = 100
decay_steps = len(train_set) * max_epochs

def cosine_decay_scheduler(epoch, lr):
    new_lr = initial_learning_rate/2 * (1+np.cos(np.pi * epoch / decay_steps))
    return new_lr

lr_scheduler = LearningRateScheduler(cosine_decay_scheduler)

def generate_data(generator):
    while True:
        batch_x, batch_y = next(generator)
        yield batch_x, batch_y

# Create individual models
vgg16_model = create_vgg16_model()
xception_model = create_xception_model()
convnext_model = create_convnext_model()

# Set paths for saving model weights
vgg16_weights_path = "/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/vgg16_weights.weights.h5"
xception_weights_path = "/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/xception_weights.weights.h5"
convnext_weights_path = "/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/convnext_weights.weights.h5"

# Train individual models
vgg16_history = vgg16_model.fit(
    generate_data(train_set),
    steps_per_epoch=len(train_set),
    epochs=max_epochs,
    validation_data=generate_data(validation_set),
    validation_steps=len(validation_set),
    callbacks=[lr_scheduler]
)
vgg16_model.save_weights(vgg16_weights_path)

xception_history = xception_model.fit(
    generate_data(train_set),
    steps_per_epoch=len(train_set),
    epochs=max_epochs,
    validation_data=generate_data(validation_set),
    validation_steps=len(validation_set),
    callbacks=[lr_scheduler]
)
xception_model.save_weights(xception_weights_path)

convnext_history = convnext_model.fit(
    generate_data(train_set),
    steps_per_epoch=len(train_set),
    epochs=max_epochs,
    validation_data=generate_data(validation_set),
    validation_steps=len(validation_set),
    callbacks=[lr_scheduler]
)
convnext_model.save_weights(convnext_weights_path)

# Create ensemble model
ensemble_models = [vgg16_model, xception_model, convnext_model]

# Define input and output of the ensemble model
ensemble_input = Input(shape=(128, 128, 3))
ensemble_outputs = [model(ensemble_input) for model in ensemble_models]
ensemble_output = Average()(ensemble_outputs)
ensemble_model = Model(inputs=ensemble_input, outputs=ensemble_output)

# Compile the ensemble model
ensemble_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluate individual models on test set
vgg16_true_value, vgg16_pred = evaluate_ensemble([vgg16_model], test_path)
vgg16_accuracy = accuracy_score(vgg16_true_value, vgg16_pred)

xception_true_value, xception_pred = evaluate_ensemble([xception_model], test_path)
xception_accuracy = accuracy_score(xception_true_value, xception_pred)

convnext_true_value, convnext_pred = evaluate_ensemble([convnext_model], test_path)
convnext_accuracy = accuracy_score(convnext_true_value, convnext_pred)

# Evaluate ensemble model on test set
true_value, ensemble_pred = evaluate_ensemble(ensemble_models, test_path)
ensemble_accuracy = accuracy_score(true_value, ensemble_pred)

# Print accuracies
print("Individual Model Test Metrics:")
print("VGG-16 Accuracy: {:.4f}".format(vgg16_accuracy))
print("Xception Accuracy: {:.4f}".format(xception_accuracy))
print("ConvNeXtBase Accuracy: {:.4f}".format(convnext_accuracy))
print("\nEnsemble Model Test Metrics:")
print("Accuracy: {:.4f}".format(ensemble_accuracy))

# Generate classification report for ensemble model
class_names = ['early_brahmi', 'later_brahmi', 'medieval_sinhala', 'modern_sinhala', 'transitional_brahmi']
print('\nClassification Report\n')
print(classification_report(true_value, ensemble_pred, target_names=class_names))

# Generate confusion matrix for ensemble model
confusion = confusion_matrix(true_value, ensemble_pred)
print('Confusion Matrix\n')
print(confusion)

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Ensemble Model')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0, ha='right')
plt.tight_layout()
plt.show()

# Save individual models
vgg16_model_path = "/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/vgg16_model"
xception_model_path = "/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/xception_model"
convnext_model_path = "/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/convnext_model"

tf.keras.models.save_model(vgg16_model, vgg16_model_path)
tf.keras.models.save_model(xception_model, xception_model_path)
tf.keras.models.save_model(convnext_model, convnext_model_path)

# Save ensemble model
ensemble_model_path = "/content/drive/MyDrive/classification_of_inscriptions_periods/step2_training/ens_model"
tf.keras.models.save_model(ensemble_model, ensemble_model_path)
