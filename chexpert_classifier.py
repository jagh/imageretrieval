""" Script to classify some conditions from CheXpert dataset """

import os
import pandas as pd

from engine.preprocessing import CheXpert
from engine.utils import Utils
from engine.dnn_classifier import DenseNET121

#######################################################################
## Workflow Launcher settings
#######################################################################

######################################################################
## preprocessing
## step-1: Get labels from CheXpert dataset
metadata_dir = "/data/01_UB/CXR_Datasets/CheXpert-v1.0-small/"
chx_data_index = CheXpert().get_labels(os.path.join(metadata_dir, "train.csv"))  #, 223422)

# ## Write the custom dataframe
# chx_dataframe = pd.DataFrame(chx_data_index, columns=["img_path", "label"])
# chx_dataframe.to_csv(os.path.join(metadata_dir, "chx_custom_dataset.csv"), index=False)

## step-2: Split the dataset
train, valid = CheXpert().dataset_splitting(chx_data_index)

# ## Saving the training set location
# train_df = pd.DataFrame(list(zip(train.img_paths, train.labels, train.labels_name)),
#                                 columns=["img_paths", "labels", "labels_name"])
# train_df.to_csv(os.path.join(metadata_dir, "chx_train_set.csv"), index=False)
#
# ## Saving the validation set location
# valid_df = pd.DataFrame(list(zip(valid.img_paths, valid.labels, valid.labels_name)),
#                                 columns=["img_paths", "labels", "labels_name"])
# valid_df.to_csv(os.path.join(metadata_dir, "chx_valid_set.csv"), index=False)


######################################################################
## DNN training
## step-1: Load the chest x-rays images in jpg
images_folder = "/data/01_UB/CXR_Datasets/"
train_set = Utils().image_loader(images_folder, train)
# print("train imgs:", train_set.imgs.shape)
# print("train labels:", train_set.labels.shape)

valid_set = Utils().image_loader(images_folder, valid)
# print("valid imgs:", valid_set.imgs.shape)
# print("valid labels:", valid_set.labels.shape)


## step-2: training the model
DenseNET121().fit_model(train_set, valid_set, 2)
