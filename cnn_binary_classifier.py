"""
+ CNN model to classify normal and pneumonia chest x-ray images
    with attention mechanism based on the interpretability Saliency Maps.
+ Dataset source: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
"""

import tensorflow as tf
import keras
import keras.backend
import keras.layers
import keras.models
import keras.utils

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras import backend as K

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import innvestigate
import innvestigate.utils as iutils


## Set Dataset path
dir_Dset = "cxr_dataset/"
train_data_dir = str(dir_Dset + "/train/")
validation_data_dir = str(dir_Dset + "/val/")
test_data_dir = str(dir_Dset + "/test/")

# ## Explore Dataset
# img_normal_name = 'NORMAL2-IM-0588-0001.jpeg'
# img_normal = load_img(str(train_data_dir + "/NORMAL/" + img_normal_name))
# plt.savefig("img_patient-normal.png")
# print("-- SAVED IMG NORMAL --")
#
# img_pneumonia_name = 'person63_bacteria_306.jpeg'
# img_pneumonia = load_img(str(train_data_dir + "/PNEUMONIA/" + img_pneumonia_name))
# plt.savefig("img_patient-pneumonia.png")
# print("-- SAVED IMG PNEUMONIA --")


## Set CXR image dimensions
img_width, img_height = 150, 150

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


## Set CNN hyperparameters
nb_train_samples = 4968
nb_validation_samples = 264
epochs = 20
batch_size = 16


## Create a sequential CNN model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation="sigmoid"),
])

# print("++ Model Layers: {}".format(model.layers))
# print("++ Model Input: {}".format(model.input))
# print("++ Model Output: {}".format(model.output))

## Generate a model graph object
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


## Set augmentations for training and test
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

## Data generator with augmentations
train_generator = train_datagen.flow_from_directory(train_data_dir,
                            target_size=(img_width, img_height),
                            batch_size=batch_size,
                            class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
                            validation_data_dir,
                            target_size=(img_width, img_height),
                            batch_size=batch_size,
                            class_mode='binary')

test_generator = test_datagen.flow_from_directory(
                            test_data_dir,
                            target_size=(img_width, img_height),
                            batch_size=batch_size,
                            class_mode='binary')

## Training the model
history = model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size)

model.save_weights('cnn_weights.h5')

## Test evaluation the model
scores = model.evaluate_generator(test_generator, steps=batch_size)
print("++ Test accuracy: %.2f%% ++" % (scores[1]*100))

## Saving the convergence curves
# print(history.history.keys())
## Accuracy plot
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("outputs/plot_ccn_accuracy.png")
plt.close()

## Loss plot
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("outputs/plot_cnn_loss.png")



########################################################################
## Interpretability with saliency maps
########################################################################

## Loading an image in a tensor shape and adding the bach dimensions
img = cv2.imread(str(test_data_dir+"/NORMAL/IM-0001-0001.jpeg"))
img_resize = cv2.resize(img, dsize=(img_width, img_height), interpolation=cv2.INTER_CUBIC)
img_expand = np.expand_dims(img_resize, axis=0)

# score_img = model.evaluate(img_expand, batch_size=1)
# print("++ Test Image Accuracy: {} ++".format(score_img))

plt.figure(1)
plt.imshow(img_expand.squeeze(), cmap='gray', interpolation='nearest')
plt.savefig("outputs/cxr_image.png")
plt.close()



### Using the analyzer iNNvestigate

## Linear method
# # Stripping the softmax activation from the model
# model_wo_sm = iutils.keras.graph.model_wo_softmax(model)
# # Creating an analyzer
# gradient_analyzer = innvestigate.analyzer.Gradient(model_wo_sm)

# ## Applying the analyzer
# analysis = gradient_analyzer.analyze(img_expand)


## Using DeepTaylor method as analyzer
analyzer = innvestigate.create_analyzer("deep_taylor", model)
analysis = analyzer.analyze(img_expand)

## Aggregate along color channels and normalize to [-1, 1]
a = analysis.sum(axis=np.argmax(np.asarray(analysis.shape) == 3))
a /= np.max(np.abs(a))

# Displaying the gradient
plt.figure(2)
plt.imshow(a[0], cmap='seismic', clim=(-1, 1))
plt.savefig("outputs/cxr_linearizedNetworkFunction.png")
plt.close

print("!!!"*20)
print("-- CNN BINARY CLASSIFIER --")
print("!!!"*20)
