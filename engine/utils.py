

import os, pickle
import cv2
import collections
import numpy as np

DataIndex = collections.namedtuple('DataIndex', 'name img_paths labels labels_name')
Dataset = collections.namedtuple('Dataset', 'name imgs labels')

class Utils:
    """ Module for defining utility functions """


    def __init__(self):
        pass

    def make_dir(directory, sandbox = None) -> str:
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not sandbox:
            return str(directory)
        path = os.popen('cd').readline()
        return str(path+"/"+directory)

    def image_loader(self, images_folder, data_index, img_width = 224, img_height = 224):
        """ Loading the chest X-ray image and applied interpolation.
            data_index: receives a collection with format ('DataIndex', 'name img_paths labels').
        """
        imgs_list = []
        labels_list = []
        cases_not_found = []

        for i in range(len(data_index.img_paths)):
            try:
                ## Load image, swap color channels and resize it
                image = cv2.imread(os.path.join(images_folder, data_index.img_paths[i]))
                image = cv2.resize(image, dsize=(img_width, img_height), interpolation=cv2.INTER_CUBIC)

                ## Appening the images and labels_list
                imgs_list.append(image)
                labels_list.append(data_index.labels[i])

            except cv2.error:
                cases_not_found.append((row))
                print("!! Case not found ->", row)

        ## Return a dataset instance
        return Dataset(name=data_index.name, imgs=np.array(imgs_list), labels=np.array(labels_list))
