import numpy as np
import pdb

import numpy as np
import matplotlib.pyplot as plt

def plot_normalized_confusion_matrix(cm, class_names=None, figsize=(8,8), cmap="viridis"):
    """
    Plot normalized confusion matrix.

    Args:
        cm (np.ndarray): confusion matrix (N x N)
        class_names (list): optional class names
        figsize (tuple): figure size
        cmap (str): matplotlib colormap
    """

    # normalize by row (true class)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, cmap=cmap)

    n_classes = cm.shape[0]

    # ticks
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))

    if class_names is not None:
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized Confusion Matrix")

    # add text values
    for i in range(n_classes):
        for j in range(n_classes):
            value = cm_norm[i, j]
            ax.text(
                j, i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black"
            )

    plt.tight_layout()
    plt.savefig("normalized_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    return cm_norm

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        # matrix shape(num_class, num_class) with elements 0 in our match. it will be 4*4

    # def Kappa(self):
    #     xsum = np.sum(self.confusion_matrix, axis=1)  # sum by row
    #     ysum = np.sum(self.confusion_matrix, axis=0)  # sum by column

    #     Pe = np.sum(ysum * xsum) * 1.0 / (self.confusion_matrix.sum() ** 2)
    #     P0 = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()  # predict right / all the data
    #     cohens_coefficient = (P0 - Pe) / (1 - Pe)

    #     cm_norm = plot_normalized_confusion_matrix(self.confusion_matrix)

    #     return cohens_coefficient
    def Kappa(self):
        cm = self.confusion_matrix.astype(float)

        # sum of rows (true labels)
        xsum = np.sum(cm, axis=1)

        # sum of columns (predicted labels)
        ysum = np.sum(cm, axis=0)

        total = np.sum(cm)

        # expected agreement
        Pe = np.dot(xsum, ysum) / (total ** 2)

        # observed agreement (OA)
        P0 = np.trace(cm) / total

        # Cohen's Kappa
        cohens_coefficient = (P0 - Pe) / (1 - Pe)

        return cohens_coefficient

    def ProducerA(self):
        #
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)

    def UserA(self):
        #
        return np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        # each pred right class is in diag. sum by row is the count of corresponding class
        mAcc = np.nanmean(Acc)  #
        return mAcc, Acc

    def Mean_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(IoU)
        return MIoU, IoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        # gt_image = batch_size*256*256   pre_image = batch_size*256*256
        mask = (gt_image >= 0) & (gt_image < self.num_class)  # valid in mask show True, ignored in mask show False
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        # gt_image[mask] : find out valid pixels. elements with 0,1,2,3 , so label range in  0-15
        count = np.bincount(label, minlength=self.num_class ** 2)
        # [0, 1, 2, 3,  confusion_matrix like this:
        #  4, 5, 6, 7,  and if the element is on the diagonal, it means predict the right class.
        #  8, 9, 10,11, row means the real label, column means pred label
        #  12,13,14,15]
        # return a array [a,b....], each letters holds the count of a class and map to class0, class1...
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)