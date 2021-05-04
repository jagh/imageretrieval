

import numpy as np
from itertools import product
import matplotlib.pyplot as plt



class MetricsDisplay:
    """ Todule to display training and test metrics """

    def __init__(self):
        pass

    def plot_accuracy(self, history, filename):
        """ Plotting keras history record """
        plt.figure(1)
        plt.plot(history['acc'], '.-', color="darkturquoise")
        plt.plot(history['val_acc'], '.-', color="mediumpurple", label="Validation score")
        # plt.title('model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.ylim([0, 1])
        plt.legend(['train', 'val'], loc='lower right')
        # plt.savefig(os.path.join(dir_dnn_train, "postpro_accuracy.png"))
        plt.savefig(filename)
        plt.close()


    def plot_loss(self, history, filename):
        plt.figure(1)
        plt.plot(history['loss'], '.-', color="darkturquoise")
        plt.plot(history['val_loss'], '.-', color="mediumpurple", label="Validation score")
        # plt.title('model loss')
        plt.ylabel('Loss (B. Crossentropy)')
        plt.xlabel('Epochs')
        plt.ylim([0, 2])
        plt.legend(['train', 'val'], loc='upper left')
        # plt.savefig(os.path.join(dir_dnn_train, "postpro_loss.png"))
        plt.savefig(filename)
        plt.close()


    def plot_aucroc(self, history, filename):
        plt.figure(1)
        plt.plot(history['auroc'], '.-', color="darkturquoise")
        plt.plot(history['val_auroc'], '.-', color="mediumpurple", label="Validation score")
        # plt.title('model aucroc')
        plt.ylabel('ROC AUC')
        plt.xlabel('Epochs')
        plt.ylim([0, 1])
        plt.legend(['train', 'val'], loc='lower right')
        # plt.savefig(os.path.join(dir_dnn_train, "postpro_aucroc.png"))
        plt.savefig(filename)
        plt.close()


    def plot_confusion_matrix(self, cm, filename = None, n_classes = 2):
        """ This function is based from sklearn plot_confusion_matrix()
            + https://github.com/scikit-learn/scikit-learn/blob/15a949460/sklearn/metrics/_plot/confusion_matrix.py#L168
        """

        fig, ax = plt.subplots()
        cm = cm
        n_classes = n_classes
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

        # if display_labels is None:
        #     plt.show()
        # else:
        plt.savefig(filename)

        plt.close()
