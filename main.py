import torch
import kornia as K

from utils import imshow, imread

if __name__ == '__main__':
    img1: torch.Tensor = imread('./data/simba.png')
    img2: torch.Tensor = imread('./data/arturito.jpg')

    resize = K.augmentation.Resize((250, 250))
    img1, img2 = resize(img1), resize(img2)

    img_batch = torch.cat([img1, img2], dim=0)

    keypoints = torch.tensor([[[175, 125.], [140., 100.]], [[120, 70.], [180., 130.]]])  # BxNx2 [x,y]
    features_kp = torch.ones(2, 1, 250, 250)
    features_ori = torch.zeros(2, 36, 250, 250)
''' 
    imgs_trans, features_kp_trans, features_ori_trans, coords_trans, mask_trans = random_augmentation(img_batch,
                                                                                                      features_kp,
                                                                                                      features_ori,
                                                                                                      keypoints)

    print(" IMAGE: {}\n Feat KP.: {}\n Feat Ori.: {}\n KP: {}\n Masks: {}".format(
        imgs_trans.shape,
        features_kp_trans.shape,
        features_ori_trans.shape,
        coords_trans.shape,

        mask_trans.shape)
    )

    imshow(imgs_trans[0],coords=coords_trans[0])
    imshow(imgs_trans[1],coords=coords_trans[1])
    imshow(features_kp[0]*mask_trans[0])
'''