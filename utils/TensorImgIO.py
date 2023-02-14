import torch
import kornia as K
from skimage import io
import numpy as np
from matplotlib import pyplot as plt

def imread(data_path: str) -> torch.Tensor:
  img = io.imread(data_path)
  img_t = K.image_to_tensor(img)
  return img_t.float() / 255.

def imshow(input:torch.Tensor,coords=None):
    fig, ax = plt.subplots()
    if coords is not None:
      ax.add_patch(plt.Circle((coords[0,0], coords[0,1]), color='r'))
      ax.add_patch(plt.Circle((coords[1,0], coords[1,1]), color='r'))

    out_np: np.array = K.tensor_to_image(input)