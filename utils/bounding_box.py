import numpy as np
import torch
from matplotlib import pyplot as plt

def bound_box(points, size=14):
    """
     Return the coordinates of the bounding box centered at each point.
    :param points: (N, 3) tensor of the local maximum coordinates
    :param size: size of the bounding box
    :return: (N, 2, 2) tensor of the bounding box coordinates
    """
    boxs = []
    half1 = size//2
    half2 = size - half1
    for point in points:
        [x, y] = point[:2]
        x, y = int(x), int(y)
        if x!= 0 or y!=0:
            box = [(x - half1, y - half1), (x + half2, y + half2)]
            boxs.append(box)
    return boxs

def segmatation_box(img,box):
    """
    Return the segmented image of the bounding box centered at each point.
    """
    pt1, pt2 = box
    crop_img = img[pt1[1]:pt2[1], pt1[0]:pt2[0]].clone()
    return crop_img

def get_features(img_batch,orient_max_batch,points,size,show=False):
    """
    Return the features of the bounding box centered at each point, each feature is a (size, size) tensor and was applied a circular mask.
    :param img_batch: (B, H, W) tensor of the image that represent the activation map
    :param points: (B, N, 3) tensor of the local maximum coordinates
    :param size: size of the bounding box
    :return: (B, N, size, size) tensor of the features
    """
    batch_result =[]
    for i, img in enumerate(img_batch):
        boxs = bound_box(points[i], size=size)
        print(type(boxs))
        result = []
        feature=img[0].clone()
        orient = orient_max_batch[i].clone()
        tem_img_feat =None
        tem_img_orie =None
        for j, box in enumerate(boxs):
            seg_feat = segmatation_box(feature,box)
            seg_orie = segmatation_box(orient,box)
            if tem_img_orie is None:
                tem_img_feat = seg_feat[None]
                tem_img_orie = seg_orie[None]
            else:
                tem_img_feat = torch.cat([tem_img_feat,seg_feat[None]],dim=0)
                tem_img_orie = torch.cat([tem_img_orie,seg_orie[None]],dim=0)
        batch_result.append([box,tem_img_feat,tem_img_orie])
    return batch_result