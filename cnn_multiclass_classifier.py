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

## iNNvestigate
import os
import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt

import innvestigate
import innvestigate.utils as iutils

## Load and preprocess images
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report




## Set Dataset path
dir_Dset = "cxr_dataset/"
train_data_dir = str(dir_Dset + "/train/")
# train_data_dir = str(dir_Dset + "/train_sample")
validation_data_dir = str(dir_Dset + "/val/")
test_data_dir = str(dir_Dset + "/test/")
visualization_samples = str(dir_Dset + "/visualization_samples/")

## Set CXR image dimensions
# img_width, img_height = 150, 150
img_width, img_height = 224, 224

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


## Set CNN hyperparameters
nb_train_samples = 4968
# nb_train_samples = 400
nb_validation_samples = 264
epochs = 100
batch_size = 32     #16



#############################################################
## Data preprocessing
#############################################################
def classDataLoader(image_paths):
    train_data = []
    train_labels = []
    ## Loop over the image paths
    for img_path in image_paths:
        ## Extract the class label from the filename
        label = img_path.split(os.path.sep)[-2]

        ## Load image, swap color channels and resize it
        image = cv2.imread(img_path)
        image = cv2.resize(image, dsize=(img_width, img_height), interpolation=cv2.INTER_CUBIC)
        # image = np.expand_dims(image, axis=0)

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (img_width, img_height))

        ## Updated the data and labels list, respectively
        train_data.append(image)
        train_labels.append(label)

    return train_data, train_labels


## Load and preprocess the chest x-ray image
path_class_1 = glob(os.path.join(train_data_dir+"/NORMAL/*"))
path_class_2 = glob(os.path.join(train_data_dir+"/PNEUMONIA/*"))

## Load and preprocess the chest x-ray image
data_class_1, labels_class_1 = classDataLoader(path_class_1)
data_class_2, labels_class_2 = classDataLoader(path_class_2)

print("++ Tdata_class_1,: {} || {} ++".format(len(data_class_1), len(data_class_2 )))

## Concatenate dataset and transform to NumPy arrays
temp_data = data_class_1 + data_class_2
temp_labels = labels_class_1 + labels_class_2

data = np.array(temp_data)
labels = np.array(temp_labels)

## perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# labels = np_utils.to_categorical(labels)
labels = keras.utils.to_categorical(labels)


## Data partition
(train_x, valid_x, train_y, valid_y) = train_test_split(data, labels,
	test_size=0.05, stratify=labels, random_state=42)

print("++ Dataset Distribution  ++")
print("++ Train set: {} || {} ++".format( train_x.shape, train_y.shape ))
print("++ Train set: {} || {} ++".format( valid_x.shape, valid_y.shape ))


#############################################################
## Image Augmentation and Transformations
#############################################################
## Initialize the training data augmentation object
# train_augmentation =  ImageDataGenerator(
#                             rotation_range=30,
#                             zoom_range=0.15,
#                             width_shift_range=0.2,
#                             height_shift_range=0.2,
#                             shear_range=0.15,
#                             horizontal_flip=True,
#                             fill_mode="nearest")

train_augmentation = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
valid_augmentation = ImageDataGenerator(rescale=1. / 255)


print("++ Augmentation Set  ++")
print("++ Train set: {} ++".format( type(train_augmentation) ))
print("++ Train set: {} ++".format( type(valid_augmentation) ))



## define the ImageNet mean subtraction (in RGB order) and set the
## the mean subtraction value for each of the data augmentation
## objects
# mean = np.array([123.68, 116.779, 103.939], dtype="float32")
# train_augmentation.mean = mean
# vald_augmentation.mean = mean



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
    keras.layers.Dense(2, activation="softmax"),
    # keras.layers.Dense(1, activation="sigmoid"),
    ])
model.summary()

print("++ Model Layers: {}".format(model.layers))
print("++ Model Input: {}".format(model.input))
print("++ Model Output: {}".format(model.output))

## Generate a model graph object
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

## binary_crossentropy
## categorical_crossentropy
## sparse_categorical_crossentropy
## kullback_leibler_divergence


## Train
history = model.fit_generator(
        train_augmentation.flow(train_x, train_y, batch_size=batch_size),
                steps_per_epoch=len(train_x) // batch_size,
                validation_data=valid_augmentation.flow(valid_x, valid_y),
                validation_steps=len(valid_x) // batch_size,
                epochs=epochs
                )






# ## Set augmentations for training and test
# train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# test_datagen = ImageDataGenerator(rescale=1. / 255)
#
# ## Data generator with augmentations
# train_generator = train_datagen.flow_from_directory(train_data_dir,
#                             target_size=(img_width, img_height),
#                             batch_size=batch_size,
#                             class_mode='binary')
#
# validation_generator = test_datagen.flow_from_directory(
#                             validation_data_dir,
#                             target_size=(img_width, img_height),
#                             batch_size=batch_size,
#                             class_mode='binary')
#
# test_generator = test_datagen.flow_from_directory(
#                             test_data_dir,
#                             target_size=(img_width, img_height),
#                             batch_size=batch_size,
#                             class_mode='binary')
#
#
## Training the model
# history = model.fit_generator(train_generator,
#                     steps_per_epoch=nb_train_samples // batch_size,
#                     epochs=epochs,
#                     validation_data=validation_generator,
#                     validation_steps=nb_validation_samples // batch_size)



# # model.save_weights('cnn_weights.h5')
#
# ## Test evaluation the model
# # scores = model.evaluate_generator(test_generator, steps=batch_size)
# # print("++ Test accuracy: %.2f%% ++" % (scores[1]*100))
# #
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
plt.close()



########################################################################
## Interpretability visualization with saliency maps
########################################################################
def load_images(path, img_w=img_width, img_h=img_height):
    paths = glob(path)
    for p in paths:
        img = cv2.imread(p)
        img_resize = cv2.resize(img, dsize=(img_width, img_height), interpolation=cv2.INTER_CUBIC)
        yield np.expand_dims(img_resize, axis=0)

def deepTaylorAnalyzer(img_expand):
    analyzer = innvestigate.create_analyzer("deep_taylor", model)
    analysis = analyzer.analyze(img_expand)

    ## Aggregate along color channels and normalize to [-1, 1]
    a = analysis.sum(axis=np.argmax(np.asarray(analysis.shape) == 3))
    a /= np.max(np.abs(a))
    return a


def deepTaylor_SoftAnalyzer(img_expand):

    ## Stripping the softmax activation from the model
    model_wo_sm = iutils.keras.graph.model_wo_softmax(model)

    ## Creating an analyzer
    analyzer = innvestigate.create_analyzer("deep_taylor", model_wo_sm)

    ## Applying the analyzer
    analysis = analyzer.analyze(img_expand)

    ## Aggregate along color channels and normalize to [-1, 1]
    a = analysis.sum(axis=np.argmax(np.asarray(analysis.shape) == 3))
    a /= np.max(np.abs(a))
    return a


## Displaying the gradient
index = 0
for img in load_images(str(visualization_samples+"/*")):
    ## Using DeepTaylor method as analyzer
    img_analized = deepTaylor_SoftAnalyzer(img)

    # Displaying the gradient
    plt.figure(str(index), figsize=(6, 6))
    plt.axis('off')
    # plt.imshow(img_analized[0], cmap='seismic', clim=(-1, 1))
    plt.imshow(img_analized.squeeze(), cmap='seismic', interpolation='nearest')
    plt.tight_layout()
    plt.savefig("outputs/"+str(index)+"_sm.png")
    plt.close()

    plt.figure(str(index+1), figsize=(6, 6))
    plt.axis('off')
    plt.imshow(img.squeeze(), cmap='gray', interpolation='nearest')
    plt.tight_layout()
    plt.savefig("outputs/"+str(index)+"_cxr.png")
    plt.close()

    # score_img = model.predict_classes(img)
    # print("++ Normal Image Classification {}: {} ++".format(index, score_img))

    index = index + 1


print("!!!"*20)
print("-- CNN BINARY CLASSIFIER --")
print("!!!"*20)
