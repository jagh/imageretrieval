"""
    Script to classify some conditions from MIMIC-CXR dataset
    Launcher script in backgorund: nohup python mmcxr_classifier.py > mmcxr_nerve_exp-3.log &

    Classifier workflow:
    01- Preprocessing MIMIC-CXR dataset with +&- cases with pleural effusion
    02- Data augmentation to simulate different scanner artifacts
    03- DNN training using a binary classifier from a pretreined DenseNET-121
    04- DNN evaluation using separate test set and compute the CM
    05- Visualizing the saliency maps to analyze possible shortcut learning
"""

import os
from glob import glob
import pandas as pd
import collections
import matplotlib.pyplot as plt

from engine.utils import Utils
from engine.augmenters import Augmenter
from engine.preprocessing import MIMICCXR
from engine.dnn_classifier import DenseNET121
from engine.metrics import MetricsDisplay
from engine.analyzer import Analyzer



#######################################################################
## Workflow Launcher settings
#######################################################################
## Set experiment sandbox folders
dataset_name = "mimiccxr-Debug"
# dataset_name = "mimiccxr_30epochs-T1"
# dataset_name = "mimiccxr_30epochs-T2"

######################################################################
## Step-0: Generate the output pipeline directories
sandbox = Utils.make_dir("sandbox/" + dataset_name)
dir_pp_index = Utils.make_dir(sandbox + "/01-preprocessing/")
dir_dnn_train = Utils.make_dir(sandbox + "/02-training/")


# ######################################################################
# ## Preprocessing a partial Dataset
# ## Step-1: Read the selected cases for Pleural Effusion (Wilson)
# ## Desktop path
# dataset_index_path = "/data/01_UB/CXR_Datasets/MIMIC-CXR/wilson_IDs/"
# # ## Server path
# # dataset_index_path = "/home/jgarcia/datasets/wilson_IDs/"
# mimic_IDs = MIMICCXR().read_mimic_IDs(dataset_index_path)
# print("+ mimic_IDs index: ", mimic_IDs.shape)
#
# ## Step-2: Read the mimic metadata dataframe
# # ## Desktop path
# # mimic_index_dir = "/data/01_UB/CXR_Datasets/mimic-cxr-jpg/"
# ## Server path
# mimic_index_dir = "/home/jgarcia/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/"
# mimic_index_data = pd.read_csv(os.path.join(mimic_index_dir, "mimic-cxr-2.0.0-split.csv"))
# mimic_metadata_subset = MIMICCXR().join_mimic_dataframes(mimic_index_data, mimic_IDs)
# print("+ mimic_dataset_index: ", mimic_metadata_subset.shape)
#
# ## Step-4: Read metadata index file and dataset image loader
# # ## Desktop path
# # dataset_index_path = "/data/01_UB/CXR_Datasets/MIMIC-CXR/wilson_IDs/"
# ## Server path
# dataset_index_path = "/home/jgarcia/datasets/wilson_IDs/"
# mimic_ss_metadata = pd.read_csv(os.path.join(dir_pp_index, "mimic_cxpert_label_index.csv"))
# print("+ mimic_subset_metadata", mimic_ss_metadata.shape)
#
# ## Step-3: Filter the Dicom-IDs with the images downloaded in the Desktop
# # ## Desktop path
# # dataset_index_path = "/data/01_UB/CXR_Datasets/MIMIC-CXR/wilson_IDs/"
# ## Server path
# dataset_index_path = "/home/jgarcia/datasets/wilson_IDs/"
# files_path = glob(str(desktop_images_folder + "/files/*"))
#
#
# desktop_images_metada = []
#
# ## Read each image for the mimic structure folder
# for file_path in files_path:
#     file_folder = glob(str(file_path + "/*"))
#     # print("file_path", file_path)
#     for subject_path in file_folder[:]:
#
#         ## skip html file
#         if "html" in subject_path:
#             pass
#         else:
#             # print("+ subject_path", subject_path)
#             ## get subject id
#             subject = subject_path.split(os.path.sep)[-1]
#             p_id = subject.split('p')[1]
#
#             ## Iloc subject in the mimic_subset_metadata
#             # mm_subject_id = mimic_ss_metadata[mimic_ss_metadata['subject_id']==10000935]
#             subject_row = mimic_ss_metadata[mimic_ss_metadata['subject_id']==int(p_id)]
#             if subject_row.empty:
#                 pass
#             else:
#                 # print("+ subject_row -> " , subject_row.empty)
#                 subject_folder = glob(str(subject_path + "/s*"))
#                 for study_path in subject_folder:
#                     # print("+ study_path", study_path)
#                     study_folder = glob(study_path + "/*.jpg")
#                     for dicom_path in study_folder:
#                         # print("+ dicom_path", dicom_path)
#
#                         ## get dicom id
#                         dicom_file = dicom_path.split(os.path.sep)[-1]
#                         dicom_id = dicom_file.split('.jpg')[0]
#
#                         ## Iloc dicom in the mimic_subset_metadata
#                         dicom_row = mimic_ss_metadata[mimic_ss_metadata['id_dicom']==dicom_id]
#                         if dicom_row.empty:
#                             pass
#                         else:
#                             # print("+ Type ->", type(dicom_row['Pleural Effusion']))
#                             desktop_images_metada.append((dicom_path, dicom_row['Pleural Effusion'].iloc[0], dicom_row['split_W'].iloc[0]))
#
#
# # print("+ desktop_images_metada: ", desktop_images_metada)
# mmcxr_dataframe_class = pd.DataFrame(desktop_images_metada, columns=["img_path", "label", "split"])
# print("+ mmcxr_dataframe_class", mmcxr_dataframe_class.head())
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
# ## Data augmentation
# ## step 1: read train and valid sets
# DataIndex = collections.namedtuple('DataIndex', 'name img_paths labels labels_name')
# train_df = pd.read_csv(os.path.join(dir_pp_index, "mimic_train_set.csv"))
# valid_df = pd.read_csv(os.path.join(dir_pp_index, "mimic_valid_set.csv"))
# test_df = pd.read_csv(os.path.join(dir_pp_index, "mimic_test_set.csv"))
# print("+ train_df", train_df.shape)
# print("+ valid_df", valid_df.shape)
# print("+ test_df", test_df.shape)
#
#
# ## Datasplit instances
# ns = 1024   #12245    ## num_samples
# train = DataIndex(name='train', img_paths=train_df["img_path"][:ns], labels=train_df["label"][:ns],
#                                                 labels_name=train_df["label"][:ns])
# valid = DataIndex(name='valid', img_paths=valid_df["img_path"], labels=valid_df["label"],
#                                                  labels_name=valid_df["label"])
#
# ## step-2: Load the chest x-rays images in jpg
# # ## Desktop path
# # images_folder = "/data/01_UB/CXR_Datasets/mimic-cxr-jpg/"
# ## Server path
# images_folder = "/home/jgarcia/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/"
# train_raw = Utils().image_loader(images_folder, train)
# valid_raw = Utils().image_loader(images_folder, valid)
# # print("+ train_raw: ", type(train_raw.imgs[0]))
# # print("+ train_raw: ", train_raw.imgs[0].shape)
#
# ## Call the augmentation
# # train_intst_imgs = Augmenter().rescaleIntensity(train_raw.imgs)
# # valid_intst_imgs = Augmenter().rescaleIntensity(valid_raw.imgs)
# # train_rnoise_imgs = Augmenter().randomGaussianFilter(train_raw.imgs, 2)
# # valid_rnoise_imgs = Augmenter().randomGaussianFilter(valid_raw.imgs, 2)
# # Augmenter().plotting_fake_vendors(train_raw.imgs)


# ######################################################################
# ## DNN training
# ## step 1: read train and valid sets
# ## step-2: Load the chest x-rays images in jpg
# num_epochs = 5   #100  ##num_samples
# batch_size = 256    #32
# # learning_rate = 0.001
# # weight_decay = 0.0001
#
# # ## Desktop path
# # images_folder = "/data/01_UB/CXR_Datasets/mimic-cxr-jpg/"
# # # ## Server path
# # # images_folder = "/home/jgarcia/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/"
#
#
# Dataset = collections.namedtuple('Dataset', 'name imgs labels')
#
# ## Transformation-0:
# train_set = Dataset(name=train_raw.name, imgs=train_raw.imgs, labels=train_raw.labels)
# valid_set = Dataset(name=valid_raw.name, imgs=valid_raw.imgs, labels=valid_raw.labels)
#
# # ## Transformation-1:
# # train_set = Dataset(name=train_raw.name, imgs=train_intst_imgs, labels=train_raw.labels)
# # valid_set = Dataset(name=valid_raw.name, imgs=valid_intst_imgs, labels=valid_raw.labels)
#
# # ## Transformation-2:
# # train_set = Dataset(name=train_raw.name, imgs=train_rnoise_imgs, labels=train_raw.labels)
# # valid_set = Dataset(name=valid_raw.name, imgs=valid_rnoise_imgs, labels=valid_raw.labels)
#
# history, model = DenseNET121().fit_binary_model(train_set, valid_set, dir_dnn_train, num_epochs, batch_size)
#
#
# #########################################################################
# ## Plot learning displayLearningCurve
# history_file = os.path.join(dir_dnn_train, "DenseNet161-MMCXR_train_HistoryDict")
# history_dict = pd.read_pickle(history_file)
# history = pd.DataFrame.from_dict(history_dict)
#
# acc_filename = os.path.join(dir_dnn_train, "postpro_accuracy.png")
# MetricsDisplay().plot_accuracy(history, filename=acc_filename)
#
# loss_filename = os.path.join(dir_dnn_train, "postpro_loss.png")
# MetricsDisplay().plot_loss(history, filename=loss_filename)
#
# auc_filename = os.path.join(dir_dnn_train, "postpro_aucroc.png")
# MetricsDisplay().plot_aucroc(history, filename=auc_filename)



# #########################################################################
# ## DNN Evaluation
# ## read test index file
# # test_df = pd.read_csv(os.path.join(dir_pp_index, "mimic_test_set-small.csv"))
# test_df = pd.read_csv(os.path.join(dir_pp_index, "mimic_test_set.csv"))
#
# ## Load the mimic-cxr images in jpg
# images_folder = "/data/01_UB/CXR_Datasets/mimic-cxr-jpg/"
# # images_folder = "/home/jgarcia/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/"
#
# ## Load the model weights
# weights_file = os.path.join(dir_dnn_train, "DenseNet161-MMCXR_weights.h5")
#
# ## Predict the test values and compute the confusion matrix
# cm = DenseNET121().model_evaluation(test_df, images_folder, weights_file)
#
# ## Plot and save the confusion matrix
# cm_filename = os.path.join(dir_dnn_train, "postpro_cm.png")
# MetricsDisplay().plot_confusion_matrix(cm, filename=cm_filename)



# ########################################################################
# ## Interpretability visualization with saliency maps
# ## Set directory path
# dir_vi_samples = str(sandbox + "/03-visualization_samples")
# dir_output_sm = Utils.make_dir(sandbox + "/04-output_sailency_maps")
# # print('visualization_samples', visualization_samples)
#
# ## Load the model weights
# weights_file = os.path.join(dir_dnn_train, "DenseNet161-MMCXR_weights.h5")
# # weights_file = os.path.join(dir_dnn_train, "DenseNet161-MMCXR_train_HistoryDict")
# print("+ weights_file: ", weights_file)
# model = DenseNET121().set_binary_model()
# model.load_weights(weights_file)
#
#
# ## Displaying the gradient
# index = 0
# for img in Analyzer().load_images(glob(str(dir_vi_samples + "/*"))):
#     ## Using DeepTaylor method as analyzer
#     # img_analized = deepTaylor_SoftAnalyzer(img, model)
#     img_analized = Analyzer().deepTaylorAnalyzer(img, model)
#
#     # Displaying the gradient
#     plt.figure(str(index), figsize=(6, 6))
#     plt.axis('off')
#     # plt.imshow(img_analized.squeeze(), cmap='seismic', interpolation='nearest')
#     # plt.imshow(img_analized.squeeze(), cmap='Spectral', clim=(-1, 1))
#     plt.imshow(img.squeeze(), interpolation='none')
#     plt.imshow(img_analized.squeeze(), interpolation='none',
#                                     cmap='seismic', clim=(-1, 1))
#     # plt.set_cmap("seismic")
#     # plt.imshow(img_analized.squeeze(),
#     #                 interpolation='none',
#     #                 alpha=0.7)
#
#     plt.tight_layout()
#     plt.savefig(str(dir_output_sm + '/' + str(index)+ "_sm.png"))
#     plt.close()
#
#     ############################################################
#     plt.figure(str(index+1), figsize=(6, 6))
#     plt.axis('off')
#     plt.imshow(img.squeeze(), interpolation='none')
#     plt.imshow(img_analized.squeeze(),
#                                 interpolation ='none',
#                                 alpha = 0.3)
#     # plt.set_cmap("gist_rainbow")
#     plt.tight_layout()
#     plt.savefig(str(dir_output_sm + '/' + str(index)+ "_cxr.png"))
#     plt.close()
#
#     index = index + 1
