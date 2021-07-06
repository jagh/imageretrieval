"""
    Script to classify some conditions from MIMIC-CXR dataset
    Nerve launcher script: nohup python mmcxr_classifier.py > mmcxr_nerve_exp-3.log &
"""

import os
from glob import glob
import pandas as pd
import collections
import matplotlib.pyplot as plt

from engine.utils import Utils
from engine.preprocessing import MIMICCXR
from engine.dnn_classifier import DenseNET121
from engine.metrics import MetricsDisplay





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
# # ## Desktop path
# # dataset_index_path = "/data/01_UB/CXR_Datasets/MIMIC-CXR/wilson_IDs/"
# ## Server path
# dataset_index_path = "/home/jgarcia/datasets/wilson_IDs/"
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




import torch
import torchio as tio

import cv2
import matplotlib.pyplot as plt

######################################################################
## Data augmentation

# ## step 1: read train and valid sets
DataIndex = collections.namedtuple('DataIndex', 'name img_paths labels labels_name')
train_df = pd.read_csv(os.path.join(dir_pp_index, "mimic_train_set.csv"))
valid_df = pd.read_csv(os.path.join(dir_pp_index, "mimic_valid_set.csv"))
test_df = pd.read_csv(os.path.join(dir_pp_index, "mimic_test_set.csv"))
print("+ train_df", train_df.shape)
print("+ valid_df", valid_df.shape)
print("+ test_df", test_df.shape)


## Datasplit instances
ns = 256   #12245    ## num_samples
train = DataIndex(name='train', img_paths=train_df["img_path"][:ns], labels=train_df["label"][:ns],
                                                labels_name=train_df["label"][:ns])
valid = DataIndex(name='valid', img_paths=valid_df["img_path"], labels=valid_df["label"],
                                                 labels_name=valid_df["label"])

## step-2: Load the chest x-rays images in jpg
# ## Desktop path
# images_folder = "/data/01_UB/CXR_Datasets/mimic-cxr-jpg/"
## Server path
images_folder = "/home/jgarcia/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/"
train_raw = Utils().image_loader(images_folder, train)
valid_raw = Utils().image_loader(images_folder, valid)
# print("+ train_raw: ", type(train_raw.imgs[0]))
# print("+ train_raw: ", train_raw.imgs[0].shape)


#######################################
## Transformation-1:
# transform = tio.transforms.RandomNoise()
def rescaleIntensity(index):
    params_rescaleIn = [(-0.27, 1), (-0.22, 1), (-0.17, 1), (-0.1, 1)]
    # print("-- params_rescaleIn: ", params_rescaleIn[index])
    intensity_transform = tio.RescaleIntensity(out_min_max=params_rescaleIn[index], percentiles=(0.5, 99.5)) #  (0, 1),
    train_intst_imgs = intensity_transform(train_raw.imgs)
    valid_intst_imgs = intensity_transform(valid_raw.imgs)
    # print('+ train_intst_imgs: ', type(train_intst_imgs[0]))
    # print('+ train_intst_imgs: ', train_intst_imgs[0].shape)
    return train_intst_imgs, valid_intst_imgs


#######################################
## Transformation-2:
def randomGaussianFilter(index):
    params_noiseTr = [(0.25, 0.85), (0.50, 0.90), (0.75, 0.95), (1, 1)]
    # print("-- params_noiseTr: ", params_noiseTr[3])
    random_noise_transform = tio.transforms.RandomBlur(params_noiseTr[3])
    train_rnoise_imgs = random_noise_transform(train_raw.imgs)
    valid_rnoise_imgs = random_noise_transform(valid_raw.imgs)
    # print('+ random_noise_imgs: ', type(train_rnoise_imgs[0]))
    # print('+ random_noise_imgs: ', train_rnoise_imgs[0].shape)
    return train_rnoise_imgs, valid_rnoise_imgs


#######################################
## Plotting fake vendors
plot_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
img_index = 3
fig, axes = plt.subplots(1, 9)
for i, slice in enumerate(plot_list):
    if i < 4:
        train_intst_imgs, _ = rescaleIntensity(i)
        axes[i].imshow(train_intst_imgs[img_index])
    elif i == 4:
        axes[i].imshow(train_raw.imgs[img_index])   #, cmap="gray", origin="lower")
    elif i > 4:
        train_rnoise_imgs, _ = randomGaussianFilter(i)
        axes[i].imshow(train_rnoise_imgs[img_index])
    else: pass

plt.show()



# ######################################################################
# ## DNN training
# ## step 1: read train and valid sets
# ## step-2: Load the chest x-rays images in jpg
# num_epochs = 100  ##num_samples
# batch_size = 32
#
# # ## Desktop path
# # images_folder = "/data/01_UB/CXR_Datasets/mimic-cxr-jpg/"
# # # ## Server path
# # # images_folder = "/home/jgarcia/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/"
#
#
# Dataset = collections.namedtuple('Dataset', 'name imgs labels')
#
# # ## Transformation-0:
# # train_set = Dataset(name=train_raw.name, imgs=train_raw.imgs, labels=train_raw.labels)
# # valid_set = Dataset(name=valid_raw.name, imgs=valid_raw.imgs, labels=valid_raw.labels)
#
# # ## Transformation-1:
# # train_set = Dataset(name=train_raw.name, imgs=train_intst_imgs, labels=train_raw.labels)
# # valid_set = Dataset(name=valid_raw.name, imgs=valid_intst_imgs, labels=valid_raw.labels)
#
# ## Transformation-2:
# train_set = Dataset(name=train_raw.name, imgs=train_rnoise_imgs, labels=train_raw.labels)
# valid_set = Dataset(name=valid_raw.name, imgs=valid_rnoise_imgs, labels=valid_raw.labels)
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
# # images_folder = "/data/01_UB/CXR_Datasets/mimic-cxr-jpg/"
# images_folder = "/home/jgarcia/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/"
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



# import cv2
# import numpy as np
# import innvestigate
# import innvestigate.utils as iutils
#
# ########################################################################
# ## Interpretability visualization with saliency maps
# def load_images(paths, img_w=224, img_h=224):
#     # paths = glob(path)
#     # print('++ paths', paths)
#     for p in paths:
#         img = cv2.imread(p)
#         img_resize = cv2.resize(img, dsize=(img_w, img_h), interpolation=cv2.INTER_CUBIC)
#         yield np.expand_dims(img_resize, axis=0)
#
# def deepTaylorAnalyzer(img_expand, model):
#     analyzer = innvestigate.create_analyzer("deep_taylor", model)
#     analysis = analyzer.analyze(img_expand)
#
#     ## Aggregate along color channels and normalize to [-1, 1]
#     a = analysis.sum(axis=np.argmax(np.asarray(analysis.shape) == 3))
#     a /= np.max(np.abs(a))
#     return a
#
# def deepTaylor_SoftAnalyzer(img_expand, model):
#
#     ## Stripping the softmax activation from the model
#     model_wo_sm = iutils.keras.graph.model_wo_softmax(model)
#     # model_wo_sm = iutils.keras.graph.get_model_execution_graph(model, keep_input_layers=False)
#
#     ## Creating an analyzer
#     analyzer = innvestigate.create_analyzer("deep_taylor", model_wo_sm)
#
#     ## Applying the analyzer
#     analysis = analyzer.analyze(img_expand)
#
#     ## Aggregate along color channels and normalize to [-1, 1]
#     # a = analysis.sum(axis=np.argmax(np.asarray(analysis.shape) == 3))
#     # a /= np.max(np.abs(a))
#
#     # Handle input depending on model and backend.
#     channels_first = keras.backend.image_data_format() == "channels_first"
#     color_conversion = "BGRtoRGB"
#
#     # Apply common postprocessing, e.g., re-ordering the channels for plotting.
#     a = imgnetutils.postprocess(analysis, color_conversion, channels_first)
#     a = imgnetutils.heatmap(a)
#     return a
#
#
#
# ## Set directory path
# dir_vi_samples = str(sandbox + "/03-visualization_samples")
# dir_output_sm = Utils.make_dir(sandbox + "/04-output_sailency_maps")
# # print('visualization_samples', visualization_samples)
#
# ## Load the model weights
# weights_file = os.path.join(dir_dnn_train, "DenseNet161-MMCXR_weights.h5")
# model = DenseNET121().set_binary_model()
# model.load_weights(weights_file)
#
#
# ## Displaying the gradient
# index = 0
# for img in load_images(glob(str(dir_vi_samples + "/*"))):
#     ## Using DeepTaylor method as analyzer
#     # img_analized = deepTaylor_SoftAnalyzer(img, model)
#     img_analized = deepTaylorAnalyzer(img, model)
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
