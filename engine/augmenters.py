
import torch
import torchio as tio

import cv2
import matplotlib.pyplot as plt


class Augmenter:
    """ Image augmentation to simulate variations from different scanner artifacts """

    def __init__(self):
        self.params_rescaleIn = [(-0.27, 1), (-0.22, 1), (-0.17, 1), (-0.1, 1)]
        self.params_noiseTr = [(0.25, 0.85), (0.50, 0.90), (0.75, 0.95), (1, 1)]


    def rescaleIntensity(self, imgs, index=3):
        """ Using TorchIO intensity transformation;
        The index tuple consists of rescale values between a [-1, 1]. """
        intensity_transform = tio.RescaleIntensity(out_min_max=self.params_rescaleIn[index],
                                                        percentiles=(0.5, 99.5)) #  (0, 1),
        return intensity_transform(imgs)


    def randomGaussianFilter(self, imgs, index=3):
        """ Using TorchIO random-size gaussian filter;
        The index tuple represents the ranges in mm of the standard deviations of
        the gaussian kernel used to transform the image along each axis. """
        random_noise_transform = tio.transforms.RandomBlur(self.params_noiseTr[index])
        return random_noise_transform(imgs)


    def plotting_fake_vendors(self, imgs, img_index=1):
        plot_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        fig, axes = plt.subplots(1, 9)
        for i, slice in enumerate(plot_list):
            if i < 4:
                intst_imgs = self.rescaleIntensity(imgs, i)
                axes[i].imshow(intst_imgs[img_index])
            elif i == 4:
                axes[i].imshow(imgs[img_index])   #, cmap="gray", origin="lower")
            elif i > 4:
                rnoise_imgs = self.randomGaussianFilter(imgs, i-5)
                axes[i].imshow(rnoise_imgs[img_index])
            else: pass
        plt.show()
