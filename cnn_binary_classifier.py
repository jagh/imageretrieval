"""
+ A cnn model to classify normal and pneumonia chest x-ray images.
+ Dataset source: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
+ Script base in the binary classifier kernel: https://www.kaggle.com/kosovanolexandr/keras-nn-x-ray-predict-pneumonia-86-54
"""


from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import os
import numpy as np
import matplotlib.pyplot as plt


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


## Create Sequential model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

#print("++ Model Layers: {}".format(model.layers))
print("++ Model Input: {}".format(model.input))
print("++ Model Output: {}".format(model.output))

## Generate a model graph object
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


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

model.save_weights('ccn_weights.h5')

## Test evaluation the model
scores = model.evaluate_generator(test_generator)
print("++ Test accuracy: %.2f%% ++" % (scores[1]*100))


## Save convergence plots
print(history.history.keys())

## Accuracy plot
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("plot_ccn_accuracy.png")
plt.close()

## Loss plot
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("plot_cnn_loss.png")
