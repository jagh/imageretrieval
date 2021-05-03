""" Script to classify some conditions from MIMIC-CXR dataset """

import os
from glob import glob
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


######################################################################
## Trainig process
## Step-1: Read metadata index file and dataset image loader




mimic_index_dir = "/data/01_UB/CXR_Datasets/mimic-cxr-jpg/"
mimic_ss_metadata = pd.read_csv(os.path.join(dir_pp_index, "mimic_cxpert_label_index.csv"))
print("+ mimic_subset_metadata", mimic_ss_metadata)
print("+ mimic_subset_metadata", type(mimic_ss_metadata))

num_folders = 1

mmcxr_images_metada = []

## Read each image for the mimic structure folder
files_path = glob(str(mimic_index_dir + "/files/*"))
for file_path in files_path:
    file_folder = glob(str(file_path + "/*"))
    # print("file_path", file_path)
    for subject_path in file_folder[:10]:

        ## skip html file
        if "html" in subject_path:
            pass
        else:
            # print("+ subject_path", subject_path)
            ## get subject id
            subject = subject_path.split(os.path.sep)[-1]
            p_id = subject.split('p')[1]

            ## Iloc subject in the mimic_subset_metadata
            # mm_subject_id = mimic_ss_metadata[mimic_ss_metadata['subject_id']==10000935]
            subject_row = mimic_ss_metadata[mimic_ss_metadata['subject_id']==int(p_id)]
            if subject_row.empty:
                pass
            else:
                print("+ subject_row -> " , subject_row.empty)
                subject_folder = glob(str(subject_path + "/s*"))
                for study_path in subject_folder:
                    # print("+ study_path", study_path)
                    study_folder = glob(study_path + "/*.jpg")
                    for dicom_path in study_folder:
                        # print("+ dicom_path", dicom_path)

                        ## get dicom id
                        dicom_file = dicom_path.split(os.path.sep)[-1]
                        dicom_id = dicom_file.split('.jpg')[0]
                        # print("+ type dicom_id", dicom_id)
                        # print("+ type dicom_id", type(dicom_id))

                        ## Iloc dicom in the mimic_subset_metadata
                        dicom_row = mimic_ss_metadata[mimic_ss_metadata['id_dicom']==dicom_id]
                        if dicom_row.empty:
                            pass
                        else:
                            print("+ dicom_id", dicom_id)
                            print("+ dicom_path", dicom_path)
                            print("+ dicom row ->", dicom_row['Pleural Effusion'].iloc[0])
                            # print("+ Type ->", type(dicom_row['Pleural Effusion']))
                            mmcxr_images_metada.append((dicom_path, dicom_row['Pleural Effusion'].iloc[0]))


mmcxr_dataframe = pd.DataFrame(mmcxr_images_metada, columns=["img_path", "label"])
print("mmcxr_dataframe", mmcxr_dataframe)


## Metadata Iloc subject
# for subject in mimic_metadta["subject_id"][:10]:
#     print("subject:", subject)
#     subject_path = os.path.join(mimic_index_dir, "files")



# subject_path = os.path.join(mimic_folder, "files", "p10", "p10000935")
# subject_folder = glob(str(subject_path + "/*"))
# print("subject_folder: ", subject_folder)

# mmcxr_images_index = []
#
# ## Read each image for the mimic structure folder
# files_path = glob(str(mimic_index_dir + "/files/*"))
# for file_path in files_path:
#     file_folder = glob(str(file_path + "/*"))
#     print("file_path", file_path)
#     for subject_path in file_folder[:2]:
#         print("subject_path", subject_path)
#         subject_folder = glob(str(subject_path + "/s*"))
#         for study_path in subject_folder:
#             print("+ study_path", study_path)
#             study_folder = glob(study_path + "/*.jpg")
#             for dicom_path in study_folder:
#                 print("+ dicom_path", dicom_path)



#
#
# for study_path in subject_folder:
#     study_folder = glob(str(study_path + "/*.jpg"))
#     for id_dicom in study_folder:
#         print("+ p10", id_dicom)
