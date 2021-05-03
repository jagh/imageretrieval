
import os
import pickle
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
from keras import losses
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

from keras.applications.densenet import DenseNet121, preprocess_input

Dataset = collections.namedtuple('Dataset', 'name imgs labels')

## GPU environment flag
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

## prevent OOM issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


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

    def fit_model(self, train_set, valid_set, dir_dnn_train, epochs=5, batch_size=32):
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
        model.save_weights(os.path.join(dir_dnn_train, 'DenseNet161-CXP_weights.h5'))
        return history, model


    def set_binary_model(self):
        """ Transfer knowledge from DenseNet121 trained on ImageNet """

        dense_model = DenseNet121(include_top=True, weights='imagenet', input_shape=(224,224,3))
        output = Dense(1, activation='sigmoid', name='clf')(dense_model.layers[-2].output)
        model = Model(inputs=dense_model.input, outputs=output)

        def auroc(y_true, y_pred):
            return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

        model.compile(optimizer='adadelta', loss=losses.binary_crossentropy, metrics=['accuracy', auroc])
        # model.compile(optimizer='adadelta', loss=losses.binary_crossentropy, metrics=['accuracy'])

        return model

    def fit_binary_model(self, train_set, valid_set, dir_dnn_train, epochs=5, batch_size=32):
        """ Training the model
            https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/"""

        # train_aug, valid_aug = self.baseline_image_aumentation()
        model = self.set_binary_model()

        # ## Training model
        # history = model.fit_generator(
        #     train_aug.flow(train_set.imgs, train_set.labels, batch_size=batch_size),
        #             steps_per_epoch=len(train_set.imgs)/batch_size,    #len(train_x)/batch_size,
        #             epochs=epochs,
        #             validation_data=valid_aug.flow(valid_set.imgs, valid_set.labels),
        #             validation_steps=len(valid_set.imgs)/batch_size,
        #             )

        ## Training model
        # history = model.fit_generator(train_set.imgs, train_set.labels, epochs=epochs, verbose=1)
        history = model.fit(train_set.imgs, train_set.labels, batch_size=batch_size, epochs=epochs,
                                validation_data=(valid_set.imgs, valid_set.labels))

        model.save_weights(os.path.join(dir_dnn_train, 'DenseNet161-MMCXR_weights.h5'))
        with open(os.path.join(dir_dnn_train, 'DenseNet161-MMCXR_train_HistoryDict'), 'wb') as history_file:
            pickle.dump(history.history, history_file)

        return history, model
