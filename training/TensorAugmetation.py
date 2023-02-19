import torch
import kornia as K

def random_augmentation(batch_image,features_kp,features_ori,keypoints):
  # aplica um aumento de dados aleatorio
  # tanto na imagem original, nas features, nos pontos e na mascara
  augme_affim =K.augmentation.RandomAffine(degrees=[-180, 180],translate=[0.05,0.05], p=1.,same_on_batch=True,keepdim=True)
  augm_pespec =K.augmentation.RandomPerspective(distortion_scale=0.2, p=1.,same_on_batch=True,keepdim=True)
  aug_compost = K.augmentation.AugmentationSequential(
    augm_pespec,
    augme_affim,
    data_keys=["input","input", "input","keypoints","mask"]
  )
  _B,_C,_W,_H = batch_image.shape
  SIZE_BORDER =25
  batch_mask = torch.zeros(_B,_C,_W,_H)
  batch_mask[:,:,SIZE_BORDER:_W-SIZE_BORDER,SIZE_BORDER:_H-SIZE_BORDER]=1
  imgs_trans, feature_kp_trans, features_ori_trans,coords_trans,mask_trans = aug_compost(batch_image,features_kp,features_ori,keypoints,batch_mask)
  return imgs_trans, feature_kp_trans, features_ori_trans,coords_trans,mask_trans


def batch_default(batch_image):
  # aplica um aumento de dados aleatorio
  # tanto na imagem original, nas features, nos pontos e na mascara
  augme_affim =K.augmentation.RandomAffine(degrees=[-180, 180],translate=[0.05,0.05], p=1.,same_on_batch=True,keepdim=True)
  augm_pespec =K.augmentation.RandomPerspective(distortion_scale=0.2, p=1.,same_on_batch=True,keepdim=True)
  aug_compost = K.augmentation.AugmentationSequential(
    augm_pespec,
    augme_affim,
    data_keys=["input","input", "input","keypoints","mask"]
  )
  _B,_C,_W,_H = batch_image.shape
  SIZE_BORDER =25
  batch_mask = torch.zeros(_B,_C,_W,_H)
  batch_mask[:,:,SIZE_BORDER:_W-SIZE_BORDER,SIZE_BORDER:_H-SIZE_BORDER]=1
  imgs_trans, feature_kp_trans, features_ori_trans,coords_trans,mask_trans = aug_compost(batch_image,features_kp,features_ori,keypoints,batch_mask)
  return imgs_trans, feature_kp_trans, features_ori_trans,coords_trans,mask_trans
