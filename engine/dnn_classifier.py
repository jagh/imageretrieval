

import collections
import tensorflow as tf
import keras.utils
from keras.preprocessing.image import ImageDataGenerator, load_img

from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Flatten
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

from keras.applications.densenet import DenseNet121, preprocess_input

Dataset = collections.namedtuple('Dataset', 'name imgs labels')


class DenseNET121:
    """Module for training a CheXpert model from a DenseNet121-ImageNet weights  """

    def __init__(self):
        pass


    def baseline_image_aumentation(self):
        """ Image augmentation and transformations """
        train_aug = ImageDataGenerator(
                                        rescale=1. / 255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)

        valid_aug = ImageDataGenerator(rescale=1. / 255)

        return train_aug, valid_aug

    def set_model(self):
        """ Transfer knowledge from DenseNet121 trained on ImageNet """

        in_img = Input(shape=((224,224,3)))
        baseModel = DenseNet121(include_top = True,
                                weights = "imagenet",
                                input_tensor = in_img,
                                )
        headModel = baseModel.layers[-2].output
        headModel = Dense(3)(headModel)
        layer_output = Activation('softmax')(headModel)

        model = Model(input=baseModel.input, output=layer_output)
        model.summary()

        ## Generate a model graph object
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model



    def fit_model(self, train_set, valid_set, epochs=2, batch_size=36):
        """ Training the model """

        train_aug, valid_aug = self.baseline_image_aumentation()
        model = self.set_model()

        ## Training model
        history = model.fit_generator(
            train_aug.flow(train_set.imgs, train_set.labels, batch_size=batch_size),
                    steps_per_epoch=len(train_set.imgs)/batch_size,    #len(train_x)/batch_size,
                    epochs=epochs,
                    validation_data=valid_aug.flow(valid_set.imgs, valid_set.labels),
                    validation_steps=len(valid_set.imgs)/batch_size,
                    )
        # model.save_weights('cnn_weights.h5')
        return history, model
