""" Script to classify some conditions from MIMIC-CXR dataset """

import os
from glob import glob
import pandas as pd
import collections
import matplotlib.pyplot as plt

from engine.utils import Utils
from engine.preprocessing import MIMICCXR
from engine.dnn_classifier import DenseNET121



def displayLearningCurves(history):
    ## Saving the convergence curves
    # print(history.history.keys())
    ## Accuracy plot
    plt.figure(1)
    # plt.plot(history.history['acc'])
    plt.plot(history['acc'], '.-', color="darkturquoise")
    plt.plot(history['val_acc'], '.-', color="mediumpurple", label="Validation score")
    # plt.title('model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.ylim([0, 1])
    plt.legend(['train', 'val'], loc='lower right')
    plt.savefig(os.path.join(dir_dnn_train, "postpro_accuracy.png"))
    plt.close()

    ## Loss plot
    plt.figure(2)
    plt.plot(history['loss'], '.-', color="darkturquoise")
    plt.plot(history['val_loss'], '.-', color="mediumpurple", label="Validation score")
    # plt.title('model loss')
    plt.ylabel('Loss (B. Crossentropy)')
    plt.xlabel('Epochs')
    plt.ylim([0, 2])
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(dir_dnn_train, "postpro_loss.png"))
    plt.close()

    plt.figure(3)
    plt.plot(history['auroc'], '.-', color="darkturquoise")
    plt.plot(history['val_auroc'], '.-', color="mediumpurple", label="Validation score")
    # plt.title('model aucroc')
    plt.ylabel('ROC AUC')
    plt.xlabel('Epochs')
    plt.ylim([0, 1])
    plt.legend(['train', 'val'], loc='lower right')
    plt.savefig(os.path.join(dir_dnn_train, "postpro_aucroc.png"))
    plt.close()


#######################################################################
## Workflow Launcher settings
#######################################################################
## Set experiment sandbox folders
dataset_name = "mimiccxr-nerve_exp2"


######################################################################
## Step-0: Generate the output pipeline directories
sandbox = Utils.make_dir("sandbox/" + dataset_name)
dir_pp_index = Utils.make_dir(sandbox + "/01-preprocessing/")
dir_dnn_train = Utils.make_dir(sandbox + "/02-training/")


######################################################################
# ## preprocessing
# ## Step-1: Read the selected cases for Consolidation and Pleural Effusion
# dataset_index_path = "/data/01_UB/CXR_Datasets/MIMIC-CXR/wilson_IDs/"
# mimic_IDs = MIMICCXR().read_mimic_IDs(dataset_index_path)
# print("+ mimic_IDs index: ", mimic_IDs.shape)
#
# ## Step-2: Read the mimic metadata frame
# mimic_index_dir = "/data/01_UB/CXR_Datasets/mimic-cxr-jpg/"
# mimic_index_data = pd.read_csv(os.path.join(mimic_index_dir, "mimic-cxr-2.0.0-split.csv"))
# mimic_metadata_subset = MIMICCXR().join_mimic_dataframes(mimic_index_data, mimic_IDs)
# print("+ mimic_dataset_index: ", mimic_metadata_subset.shape)
#
# # ## Step-3 Read the negbio index labesls
# # negio_label_index_data = pd.read_csv(os.path.join(mimic_index_dir, "mimic-cxr-2.0.0-negbio.csv"))
# # mimic_negio_label_name = os.path.join(dir_pp_index, "mimic_negbio_label_index.csv")
# # mimic_negbio_label_data = MIMICCXR().join_mimic_labels(mimic_metadata_subset, negio_label_index_data, mimic_negio_label_name)
# # print("+ mimic_negbio_label_index: ", mimic_negbio_label_data.shape)
#
# ## Step-3: Read the CheXpert index labels
# cxpert_label_index_data = pd.read_csv(os.path.join(mimic_index_dir, "mimic-cxr-2.0.0-chexpert.csv"))
# mimic_cxpert_label_name = os.path.join(dir_pp_index, "mimic_cxpert_label_index.csv")
# mimic_cxpert_label_data = MIMICCXR().join_mimic_labels(mimic_metadata_subset, cxpert_label_index_data, mimic_cxpert_label_name)
# print("+ mimic_cxpert_label_index: ", mimic_cxpert_label_data.shape)
#
# ##############
# ## Step-4: Read metadata index file and dataset image loader
# mimic_index_dir = "/data/01_UB/CXR_Datasets/mimic-cxr-jpg/"
# mimic_ss_metadata = pd.read_csv(os.path.join(dir_pp_index, "mimic_cxpert_label_index.csv"))
# print("+ mimic_subset_metadata", mimic_ss_metadata.shape)
#
# mmcxr_dataframe_class = MIMICCXR().build_mimic_dataset(mimic_index_dir, mimic_ss_metadata)
# mmcxr_dataframe_class = pd.DataFrame(mmcxr_dataframe_class, columns=["img_path", "label", "split"])
# # print("+ mmcxr_dataframe_class", mmcxr_dataframe_class)
# print("+ mmcxr_dataframe_class", mmcxr_dataframe_class.shape)
#
# train_df = mmcxr_dataframe_class[mmcxr_dataframe_class['split'] == 'train']
# train_df.to_csv(os.path.join(dir_pp_index, "mimic_train_set.csv"), index=False)
#
# valid_df = mmcxr_dataframe_class[mmcxr_dataframe_class['split'] == 'valid']
# valid_df.to_csv(os.path.join(dir_pp_index, "mimic_valid_set.csv"), index=False)
#
# test_df = mmcxr_dataframe_class[mmcxr_dataframe_class['split'] == 'test']
# test_df.to_csv(os.path.join(dir_pp_index, "mimic_test_set.csv"), index=False)
# print("+ train_df", train_df.shape)
# print("+ vaid_df", valid_df.shape)
# print("+ test_df", test_df.shape)

# ######################################################################
# ## DNN training
#
# ## step 1: read train and valid sets
# DataIndex = collections.namedtuple('DataIndex', 'name img_paths labels labels_name')
#
# train_df = pd.read_csv(os.path.join(dir_pp_index, "mimic_train_set.csv"))
# valid_df = pd.read_csv(os.path.join(dir_pp_index, "mimic_valid_set.csv"))
#
# ## Datasplit instances
# # num_samples = 256
# num_samples = 64
# train = DataIndex(name='train', img_paths=train_df["img_path"][:num_samples],
#                 labels=train_df["label"][:num_samples], labels_name=train_df["label"][:num_samples])
# valid = DataIndex(name='valid', img_paths=valid_df["img_path"], labels=valid_df["label"],
#                                                 labels_name=valid_df["label"])
#
# # print("train_df: ", train_df["img_path"].shape)
# # print("train_df: ", train_df["label"].shape)
# # print("valid_df: ", valid_df["img_path"].shape)
# # print("valid_df: ", valid_df["label"].shape)
#
#
# ## step-2: Load the chest x-rays images in jpg
# images_folder = "/data/01_UB/CXR_Datasets/mimic-cxr-jpg/"
# train_set = Utils().image_loader(images_folder, train)
# valid_set = Utils().image_loader(images_folder, valid)
#
#
# history, model = DenseNET121().fit_binary_model(train_set, valid_set, dir_dnn_train, 2, 32)
# displayLearningCurves(history)


#########################################################################
# ## Plot learning displayLearningCurve
# history_file = os.path.join(dir_dnn_train, "DenseNet161-MMCXR_train_HistoryDict")
# history_dict = pd.read_pickle(history_file)
# history = pd.DataFrame.from_dict(history_dict)
# # print("history", history)
# # print("history", history['acc'])
# print("history", history.shape)
#
# displayLearningCurves(history)



#########################################################################
## Load model and test
from keras.models import load_model
import numpy as np
from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support


## step 1: read train and valid sets
DataIndex = collections.namedtuple('DataIndex', 'name img_paths labels labels_name')
test_df = pd.read_csv(os.path.join(dir_pp_index, "mimic_test_set-small.csv"))
# test_df = pd.read_csv(os.path.join(dir_pp_index, "mimic_test_set.csv"))

## Datasplit instances
test = DataIndex(name='test', img_paths=test_df["img_path"][:], labels=test_df["label"][:],
                                        labels_name=test_df["label"][:])
# print("test_df: ", test_df["img_path"].shape)
# print("test_df: ", test_df["label"].shape)
# print("valid_df: ", valid_df["img_path"].shape)
# print("valid_df: ", valid_df["label"].shape)

# step-2: Load the chest x-rays images in jpg
images_folder = "/data/01_UB/CXR_Datasets/mimic-cxr-jpg/"
test_set = Utils().image_loader(images_folder, test)

## Step-3: Load the weights of the model
weights_file = os.path.join(dir_dnn_train, "DenseNet161-MMCXR_weights.h5")
# print("+ Weights file", weights_file)

## Get reference model from dnn_classifier class and load the weights
model = DenseNET121().set_binary_model()
model.load_weights(weights_file)

score = model.evaluate(test_set.imgs, test_set.labels, verbose=1)
print('+ Test loss:', score[0])
print('+ Test accuracy:', score[1])


## Model Prediction
predictions = model.predict(test_set.imgs)
# print("+ test predictions", predictions)

## Normalizing labels prediec
# predictions = predictions.astype(np.float)
predictions = predictions.astype(np.int)
print("+ test predictions", predictions.ravel().tolist())
print("+ test_set labels:", test_set.labels)


## Compute confusion_matrix
# labels = ["Positive", "Negative"]
conf_matrix = confusion_matrix(test_set.labels, predictions)
print("+ Conf. matrix: \n", conf_matrix)


######################################################3
# ## Plot the the confusion matrix by model selected
import matplotlib.pyplot as plt
from itertools import product

# if ax is None:
#     fig, ax = plt.subplots()
# else:
#     fig = ax.figure

fig, ax = plt.subplots()

cm = confusion_matrix(test_set.labels, predictions)
n_classes = 2
cmap = plt.cm.Blues     #'viridis'
text_ = None
include_values = True
values_format = None
display_labels = None
colorbar = True
xticks_rotation = 'horizontal'

im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)


if include_values:
    text_ = np.empty_like(cm, dtype=object)

    # print text with appropriate color depending on background
    thresh = (cm.max() + cm.min()) / 2.0

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min

        if values_format is None:
            text_cm = format(cm[i, j], '.2g')
            if cm.dtype.kind != 'f':
                text_d = format(cm[i, j], 'd')
                if len(text_d) < len(text_cm):
                    text_cm = text_d
        else:
            text_cm = format(cm[i, j], values_format)

        text_[i, j] = ax.text(
                            j, i, text_cm,
                            ha="center", va="center",
                            color=color)

if display_labels is None:
    display_labels = np.arange(n_classes)
else:
    display_labels = display_labels
if colorbar:
    fig.colorbar(im_, ax=ax)
ax.set(xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=display_labels,
            yticklabels=display_labels,
            ylabel="True label",
            xlabel="Predicted label")

ax.set_ylim((n_classes - 0.5, -0.5))
plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
plt.savefig(os.path.join(dir_dnn_train, "postpro_cm.png"))

plt.close()
