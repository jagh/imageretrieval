""" Script to classify some conditions from CheXpert dataset """

import os
import pandas as pd
import matplotlib.pyplot as plt

from engine.preprocessing import CheXpert
from engine.utils import Utils
from engine.dnn_classifier import DenseNET121


def displayLearningCurves(history):
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
    plt.savefig(os.path.join(dir_dnn_train, "accuracy.png"))
    plt.close()

    ## Loss plot
    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(dir_dnn_train, "loss.png"))
    plt.close()

#######################################################################
## Workflow Launcher settings
#######################################################################
## Set experiment sandbox folders
dataset_name = "chexpert-exp1"


######################################################################
## Step-0: Generate the output pipeline directories
sandbox = Utils.make_dir("sandbox/" + dataset_name)
dir_pp_index = Utils.make_dir(sandbox + "/01-preprocessing/")
dir_dnn_train = Utils.make_dir(sandbox + "/02-training/")


######################################################################
# preprocessing
# step-1: Get labels from CheXpert dataset
metadata_dir = "/data/01_UB/CXR_Datasets/CheXpert-v1.0-small/"
# metadata_dir = "/home/jgarcia/datasets/CXR_Datasets/CheXpert-v1.0-small/"
chx_data_index = CheXpert().get_labels(os.path.join(metadata_dir, "train.csv"))  #, 223422)

## Write the custom dataframe
chx_dataframe = pd.DataFrame(chx_data_index, columns=["img_path", "label"])
chx_dataframe.to_csv(os.path.join(dir_pp_index, "chx_custom_dataset.csv"), index=False)

## step-2: Split the dataset
train, valid = CheXpert().dataset_splitting(chx_data_index)

## Saving the training set location
train_df = pd.DataFrame(list(zip(train.img_paths, train.labels, train.labels_name)),
                                columns=["img_paths", "labels", "labels_name"])
train_df.to_csv(os.path.join(dir_pp_index, "chx_train_set.csv"), index=False)

## Saving the validation set location
valid_df = pd.DataFrame(list(zip(valid.img_paths, valid.labels, valid.labels_name)),
                                columns=["img_paths", "labels", "labels_name"])
valid_df.to_csv(os.path.join(dir_pp_index, "chx_valid_set.csv"), index=False)


######################################################################
## DNN training
## step-1: Load the chest x-rays images in jpg
images_folder = "/data/01_UB/CXR_Datasets/"
# images_folder = "/home/jgarcia/datasets/CXR_Datasets"
train_set = Utils().image_loader(images_folder, train)
valid_set = Utils().image_loader(images_folder, valid)

## step-2: training the model
history, model = DenseNET121().fit_model(train_set, valid_set, dir_dnn_train)
displayLearningCurves(history)
