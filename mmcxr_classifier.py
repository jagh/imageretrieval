""" Script to classify some conditions from MIMIC-CXR dataset """

import os
import pandas as pd

from engine.utils import Utils


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
## Step-1: Read the selected cases for Consolidation and Pleural Effusion
dataset_index_path = "/data/01_UB/CXR_Datasets/MIMIC-CXR/wilson_IDs/"
train_index = pd.read_pickle(os.path.join(dataset_index_path, "Train_ids.pickle"))
valid_index = pd.read_pickle(os.path.join(dataset_index_path, "Val_ids.pickle"))
test_index = pd.read_pickle(os.path.join(dataset_index_path, "Test_ids.pickle"))

## Convert each index list as DataFrame
train_df_index = pd.DataFrame()
train_df_index["dicom_id"] = train_index
train_df_index["split_W"] = "train"
train_df_index["id_dicom"] = train_index

valid_df_index = pd.DataFrame()
valid_df_index["dicom_id"] = valid_index
valid_df_index["split_W"] = "valid"
valid_df_index["id_dicom"] = valid_index

test_df_index = pd.DataFrame()
test_df_index["dicom_id"] = test_index
test_df_index["split_W"] = "test"
test_df_index["id_dicom"] = test_index

print("train index: ", train_df_index.shape)
print("valid index: ", valid_df_index.shape)
print("test_df_index: ", test_df_index.shape)

## Concatenate the dataframe sets
metadata_index = pd.concat([train_df_index, valid_df_index, test_df_index])
print("metadata index: ", metadata_index.shape)


##############################################################
## Step-2: Read the mimic metadata frame
mimic_index_dir = "/data/01_UB/CXR_Datasets/mimic-cxr-jpg/"
mimic_index_file = pd.read_csv(os.path.join(mimic_index_dir, "mimic-cxr-2.0.0-split.csv"))
# print("mimic_index_file: ", mimic_index_file.shape)

mimic_dataset_index = metadata_index.set_index('dicom_id').join(mimic_index_file.set_index('dicom_id'))
# print("mimic_dataset_index: ", mimic_dataset_index.shape)

## Join the two daasets to get the id study
# mimic_dataset_index.to_csv(os.path.join(dir_pp_index, "mimic_dataset_index.csv"), index=False)
mimic_dataset_index["id_study"] = mimic_dataset_index["study_id"]
mimic_dataset_index["id_subject"] = mimic_dataset_index["subject_id"]
mimic_dataset_index.pop("subject_id")

print("mimic_dataset_index: ",mimic_dataset_index.shape)


###############################################################
## Step-3 Read the negbio
negio_label_index_data = pd.read_csv(os.path.join(mimic_index_dir, "mimic-cxr-2.0.0-negbio.csv"))
mimic_negbio_label_index = mimic_dataset_index.set_index('study_id').join(negio_label_index_data.set_index('study_id'))
mimic_negbio_label_index.to_csv(os.path.join(dir_pp_index, "mimic_negbio_label_index.csv"), index=False)
print("mimic_negbio_label_index: ",mimic_negbio_label_index.shape)



chexpert_label_index_data = pd.read_csv(os.path.join(mimic_index_dir, "mimic-cxr-2.0.0-chexpert.csv"))
mimic_chexpert_label_index = mimic_dataset_index.set_index('study_id').join(chexpert_label_index_data.set_index('study_id'))
mimic_chexpert_label_index.to_csv(os.path.join(dir_pp_index, "mimic_chexpert_label_index.csv"), index=False)
print("mimic_chexpert_label_index: ",mimic_chexpert_label_index.shape)
