# sphinx_gallery_thumbnail_number = 11

import matplotlib.pyplot as plt
import numpy as np
import kornia as K
import torch

from utils import imread


def teste():
    input1: np.array = K.tensor_to_image(img11)
    input2: np.array = K.tensor_to_image(img22)

    # Some example data to display
    fig, axs = plt.subplots(2, 2, sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    (ax1, ax2), (ax3, ax4) = axs
    fig.suptitle('Sharing x per column, y per row')

    ax1.imshow(input1)
    ax2.imshow(input1)

    ax3.imshow(input1)
    ax4.imshow(input1)

    # for ax in axs.flat:
    #     ax.label_outer()

    plt.show()
def imshow3(input1:torch.Tensor,input2:torch.Tensor,coords=None):
    fig, axs = plt.subplots(2, 2,figsize=(6, 6), sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0})

    input1: np.array = K.tensor_to_image(input1)
    input2: np.array = K.tensor_to_image(input2)
    print(input1.shape,input2.shape)
    axs[0, 0].imshow(input1); axs[0, 0].axis('off');
    axs[0, 1].imshow(input2);axs[0, 1].axis('off');
    axs[1, 0].imshow(input1);axs[1, 0].axis('off');
    axs[1, 1].imshow(input2);axs[1, 1].axis('off');

    axs[0, 0].plot(coords[:, 0], coords[:, 1], 'ro');
    axs[0, 1].plot(coords[:, 0], coords[:, 1], 'ro');

    axs[1, 0].plot(coords[:, 0], coords[:, 1], 'ro');
    axs[1, 1].plot(coords[:, 0], coords[:, 1], 'ro');
    plt.show()

img11 = imread('./data/datasets/fibers/doc_0447.jpg')
img22 = imread('./data/simba.png')

points = abs(torch.rand(2, 2,3)*img11.shape[1])
print(points.shape,img11.shape,points)
imshow3(img11,img11,points)
print("show")
