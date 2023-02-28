import torch
class kp_loss:
  def __init__(self) -> None:
     self.basic = torch.nn.MSELoss()

  def __call__(self, im1,im2):
     _std = im1.std()+0.01
     return self.basic(im1,im2)/_std

class triplet_loss:
  def __init__(self) -> None:
     self.basic = torch.nn.MSELoss()

  def __call__(self, img_anchor,img_pos,img_neg):
      #https://medium.com/analytics-vidhya/triplet-loss-b9da35be21b8
     _margin = 2
     loss_pos= self.basic(img_anchor,img_pos)
     loss_neg = self.basic(img_anchor, img_neg)
     full_loss = loss_pos-loss_neg+_margin
     # print('loss_pos ', loss_pos, 'loss_neg ', loss_neg, 'full_loss ',full_loss)
     return full_loss