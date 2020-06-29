from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np


def imshow_batch(sample_batch):
    """
    用于显示FashionMNIST数据集
    :param sample_batch: 一个batch的image
    :return:
    """
    images = sample_batch[0]
    labels = sample_batch[1]
    images = make_grid(images, nrow=4, pad_value=255)

    # 1,2, 0
    images_transformed = np.transpose(images.numpy(), (1, 2, 0))
    plt.imshow(images_transformed)
    plt.axis('off')
    labels = labels.numpy()
    plt.title(labels)

