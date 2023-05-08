#importar todas dependencias necesarias para o arquivo
import numpy as np
import torch
from scipy.ndimage.filters import maximum_filter

def create_circular_mask(h, w, center=None, radius=None):
    """
    Create a circular mask of shape (h, w).
    :param h: height of the mask
    :param w: width of the mask
    :param center: (x, y) tuple of the center
    :param radius: radius of the circle
    :return: (h, w) binary mask
    """

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask

def my_normalize(df):
    """
    Normalize the dataframe between 0 and 1.
    :param df: dataframe
    :return: normalized dataframe
    """
    dfmax, dfmin = df.max(), df.min()
    df = (df - dfmin)/(dfmax - dfmin)
    return df

def create_circular_mask2(_H,_W):
    """
    Create a continuos circular mask of shape (h, w).
    :param h: height of the mask
    :param w: width of the mask
    :return: (h, w) normalized mask
    """
    center = (int(_W / 2), int(_H / 2))
    Y, X = np.ogrid[:_H, :_W]
    mask = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return 1-my_normalize(mask)

def create_interval_mask(img,v0,v1):
    """
    Create a mask of shape (h, w) with values between v0 and v1.
    :param img: (H, W) tensor of the image
    :param v0: lower bound
    :param v1: upper bound
    :return: (H, W) binary mask
    """
    mask = (img>= v0) * (img<v1)
    return mask

def intersection_filter_interval_mask(interval_filter, circle_filter):
    """
     Create a mask that join the interval and circular masks.
     :param interval_filter: (H, W) tensor of the interval mask
     :param circle_filter: (H, W) tensor of the circular mask
     :return: (H, W) binary mask
    """
    device = interval_filter.device
    inter_filter = interval_filter.to(device)*circle_filter.to(device)
    return inter_filter

def apply_circular_mask_box(img, box):
    """
    Apply a circular mask to the image.
    :param img: (H, W) tensor of the image
    :param box: (2, 2) tensor of the bounding box coordinates
    :return: (H, W) tensor of the masked image
    """
    pt1, pt2 = box
    crop_img = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    h, w = crop_img.shape
    mask = torch.tensor(create_circular_mask(h, w)).to(device=crop_img.device)

    masked_img = crop_img.clone()
    masked_img = masked_img * mask
    return masked_img