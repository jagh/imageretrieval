""" Script to classify some conditions from MIMIC-CXR dataset """

import os
import pandas as pd


#######################################################################
## Workflow Launcher settings
#######################################################################
## Set experiment sandbox folders
dataset_name = "mimiccxr-exp1"


######################################################################
## Step-0: Generate the output pipeline directories
sandbox = Utils.make_dir("sandbox/" + dataset_name)
dir_pp_index = Utils.make_dir(sandbox + "/01-preprocessing/")
dir_dnn_train = Utils.make_dir(sandbox + "/02-training/")


######################################################################
## preprocessing
## step-1: Get labels from CheXpert dataset
metadata_dir = "/data/01_UB/CXR_Datasets/MIMIC-CXR/mimic-cxr/"
dicom_index = "cxr-record-list.csv"
mreport_index = "cxr-study-list.csv"
