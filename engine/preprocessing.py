
import os
import pandas as pd
import collections
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


DataIndex = collections.namedtuple('DataIndex', 'name img_paths labels labels_name')

class CheXpert:
    """ Module for preprocessing Chexpert Dataset """

    def __init__(self):
        self.chx_data_index = []


    def get_labels(self, metadata_dir, num_rows = 2000):
        """
        Getting the conditions to build the dataset index
        """
        metadata = pd.read_csv(metadata_dir)
        for row in metadata[:][:num_rows].iterrows():
            ## Get the image path
            img_path = row[1]['Path']

            ## Get the label
            if row[1]['Pleural Effusion'] == 1 and row[1]['Frontal/Lateral'] == 'Frontal' and row[1]['AP/PA'] == 'AP':
                # custom_chx_dataset.append(str(row[1]['Path']+"|"+'PLE'))
                self.chx_data_index.append([row[1]['Path'], 'P'])

            elif row[1]['Consolidation'] == 1 and row[1]['Frontal/Lateral'] == 'Frontal' and row[1]['AP/PA'] == 'AP':
                self.chx_data_index.append([row[1]['Path'], 'C'])

            elif row[1]['Pleural Effusion'] != 1 and row[1]['Consolidation'] != 1 and \
                    row[1]['Pleural Effusion'] != 0 and row[1]['Consolidation'] != 0 and \
                    row[1]['Pleural Effusion'] != -1 and row[1]['Consolidation'] != -1 and \
                    row[1]['Frontal/Lateral'] == 'Frontal' and row[1]['AP/PA'] == 'AP':
                self.chx_data_index.append([row[1]['Path'], 'O'])

            else:
                pass

        return self.chx_data_index


    def dataset_splitting(self, chx_data_index = [], test_size = 0.10):
        """
        Split dataset in Training and Validation folds.
        """
        ## Check if a data_index is empty
        if chx_data_index:
             self.chx_data_index = chx_data_index

        ## Check if data_index is a list
        if isinstance(self.chx_data_index, list):
            self.chx_data_index = pd.DataFrame(self.chx_data_index, columns=["img_path", "label_name"])


        ## Convert labels in one-hot
        mlb = MultiLabelBinarizer(classes=["P", "C", "O"], sparse_output=False)
        mlb.fit(self.chx_data_index["label_name"])

        X_train, X_valid, y_train, y_valid = train_test_split(self.chx_data_index['img_path'],
                                                                self.chx_data_index['label_name'],
                                                                test_size=test_size)

        ## Datasplit instances
        train = DataIndex(name='train', img_paths=X_train.values,
                            labels=mlb.transform(y_train.values), labels_name=y_train.values)
        valid = DataIndex(name='valid', img_paths=X_valid.values,
                            labels=mlb.transform(y_valid.values), labels_name=y_valid.values)

        return train, valid





from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut

class MIMICCXR:
        """ Module for preprocessing MIMIC-CHX Dataset """

        def __init__(self):
            self.mimichx_data_index = []


        def read_mimic_IDs(self, dataset_index_path):
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

            ## Concatenate the dataframe sets
            metadata_index = pd.concat([train_df_index, valid_df_index, test_df_index])
            return metadata_index


        def join_mimic_dataframes(self, mimic_index_data, metadata_subset):
            mimic_metadata_subset = metadata_subset.set_index('dicom_id').join(mimic_index_data.set_index('dicom_id'))
            mimic_metadata_subset["id_study"] = mimic_metadata_subset["study_id"]
            mimic_metadata_subset["id_subject"] = mimic_metadata_subset["subject_id"]
            mimic_metadata_subset.pop("subject_id")
            return mimic_metadata_subset
            

        def join_mimic_labels(self, mimic_metadata_subset, label_index_data, mimic_label_file_name):
            mimic_subset_label_data = mimic_metadata_subset.set_index('study_id').join(label_index_data.set_index('study_id'))
            mimic_subset_label_data.to_csv(mimic_label_file_name, index=False)
            return mimic_subset_label_data



        ## Convert DICOM images into PNG
        def dicom_2_png(self, dataset_path, metadata_name, downsampling_path):
            """
            dicom to png converter
            """
            dicom_metadata = pd.read_csv(str(dataset_path+metadata_name), sep=",", header=0)
            print("++ Metadata: {}".format(dicom_metadata.shape))

            for dcm in dicom_metadata["patientId"]:
                ## Read dicom image and covert pixels into a numpy array
                ds = dcmread(str(dataset_path+"/stage_2_train_images/"+dcm+".dcm"))
                windowed = apply_voi_lut(ds.pixel_array, ds)
                ## Write the image converted
                cv2.imwrite(os.path.join(downsampling_path, str(dcm+".png")), windowed)




# ###################################################################
# ## Launcher for CheXpert dataset
# ###################################################################
#
# import os
# from preprocessing import CheXpert
#
# ## step-1: Get labels
# metadata_dir = "/data/01_UB/CXR_Datasets/CheXpert-v1.0-small/"
# chx_data_index = CheXpert().get_labels(os.path.join(metadata_dir, "train.csv"))
#
# ## Write the custom dataframe
# chx_dataframe = pd.DataFrame(chx_data_index, columns=["img_path", "label"])
# chx_dataframe.to_csv(os.path.join(metadata_dir, "chx_custom_dataset.csv"), index=False)
#
#
# ## step-2: Split the dataset
# train, valid = CheXpert().dataset_splitting(chx_data_index)
#
# ## Saving the training set location
# train_df = pd.DataFrame(list(zip(train.img_paths, train.labels, train.labels_name)),
#                                 columns=["img_paths", "labels", "labels_name"])
# train_df.to_csv(os.path.join(metadata_dir, "chx_train_set.csv"), index=False)
#
# ## Saving the validation set location
# valid_df = pd.DataFrame(list(zip(valid.img_paths, valid.labels, valid.labels_name)),
#                                 columns=["img_paths", "labels", "labels_name"])
# valid_df.to_csv(os.path.join(metadata_dir, "chx_valid_set.csv"), index=False)
