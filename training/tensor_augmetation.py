import torch
import kornia as K
from config import device


'''
    Aumenta os dados aleatoriamente , mantendo a correspondencia de variação de pespectiva entre o lote de imagens,
    os mapas de ativações, a mascara e os pontos detectados
'''
def random_augmentation(batch_image,features_kp,features_ori,keypoints):

  # aplica um aumento de dados aleatorio
  # tanto na imagem original, nas features, nos pontos e na mascara
  augme_affim =K.augmentation.RandomAffine(degrees=[-90, 90],translate=[0.05,0.05], p=1.,same_on_batch=True,keepdim=True)
  augm_pespec =K.augmentation.RandomPerspective(distortion_scale=0.05, p=1.,same_on_batch=True,keepdim=True)
  aug_compost = K.augmentation.AugmentationSequential(
    augm_pespec,
    augme_affim,
    data_keys=["input","input", "input","keypoints","mask"]
  )
  _B,_C,_W,_H = batch_image.shape
  SIZE_BORDER =7
  batch_mask = torch.zeros(_B,_C,_W,_H).to(device)
  batch_mask[:,:,SIZE_BORDER:_W-SIZE_BORDER,SIZE_BORDER:_H-SIZE_BORDER]=1
  imgs_trans, feature_kp_trans, features_ori_trans, coords_trans, mask_trans = aug_compost(batch_image,
                                                                                                           features_kp,
                                                                                                           features_ori,
                                                                                                           keypoints,
                                                                                                           batch_mask)
  return imgs_trans, feature_kp_trans, features_ori_trans,coords_trans,mask_trans

def shifted_batch_tensor(batch_img, features_key, features_ori):
    batch_neg = torch.roll(batch_img, 1, 0)
    feat_key_neg = torch.roll(features_key, 1, 0)
    feat_ori_neg = torch.roll(features_ori, 1, 0)
    return batch_neg, feat_key_neg, feat_ori_neg
