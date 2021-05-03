""" Script to classify some conditions from MIMIC-CXR dataset """

import os
import pandas as pd

from engine.utils import Utils
from engine.preprocessing import MIMICCXR


#######################################################################
## Workflow Launcher settings
#######################################################################
## Set experiment sandbox folders
dataset_name = "mimiccxr-exp2"


######################################################################
## Step-0: Generate the output pipeline directories
sandbox = Utils.make_dir("sandbox/" + dataset_name)
dir_pp_index = Utils.make_dir(sandbox + "/01-preprocessing/")
dir_dnn_train = Utils.make_dir(sandbox + "/02-training/")


######################################################################
## preprocessing
## Step-1: Read the selected cases for Consolidation and Pleural Effusion
dataset_index_path = "/data/01_UB/CXR_Datasets/MIMIC-CXR/wilson_IDs/"
mimic_IDs = MIMICCXR().read_mimic_IDs(dataset_index_path)
print("+ mimic_IDs index: ", mimic_IDs.shape)

## Step-2: Read the mimic metadata frame
mimic_index_dir = "/data/01_UB/CXR_Datasets/mimic-cxr-jpg/"
mimic_index_data = pd.read_csv(os.path.join(mimic_index_dir, "mimic-cxr-2.0.0-split.csv"))
mimic_metadata_subset = MIMICCXR().join_mimic_dataframes(mimic_index_data, mimic_IDs)
print("+ mimic_dataset_index: ", mimic_metadata_subset.shape)

# ## Step-3 Read the negbio
# negio_label_index_data = pd.read_csv(os.path.join(mimic_index_dir, "mimic-cxr-2.0.0-negbio.csv"))
# mimic_negio_label_name = os.path.join(dir_pp_index, "mimic_negbio_label_index.csv")
# mimic_negbio_label_data = MIMICCXR().join_mimic_labels(mimic_metadata_subset, negio_label_index_data, mimic_negio_label_name)
# print("+ mimic_negbio_label_index: ", mimic_negbio_label_data.shape)

## Step-3 Read the negbio
cxpert_label_index_data = pd.read_csv(os.path.join(mimic_index_dir, "mimic-cxr-2.0.0-chexpert.csv"))
mimic_cxpert_label_name = os.path.join(dir_pp_index, "mimic_cxpert_label_index.csv")
mimic_cxpert_label_data = MIMICCXR().join_mimic_labels(mimic_metadata_subset, cxpert_label_index_data, mimic_cxpert_label_name)
print("+ mimic_cxpert_label_index: ", mimic_cxpert_label_data.shape)

## Step-3 Read the negbio
