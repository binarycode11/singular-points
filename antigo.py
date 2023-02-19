import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import random

from e2cnn import nn, gspaces
from matplotlib import pyplot as plt

from config import args,device
from utils import load_model, save_model

MODEL_PATH='model_antigo.pt'
class MyFlowersDataset(torch.utils.data.Dataset):

    def __init__(self, train=True) -> None:
        super().__init__()
        split = 'train' if train else 'test'
        print(split)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize(size=(args.img_size, args.img_size)),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
        self.data = torchvision.datasets.Flowers102(root='./data', split=split,
                                                    download=True, transform=transform)

    def __len__(self):
        return len(self.data)

    def random_augmentation(self, img):
        degrees = [90,180,270, 0]
        index = random.randrange(0, len(degrees))
        img2 = torchvision.transforms.functional.rotate(img, degrees[index])
        # images=torchvision.transforms.functional.adjust_gamma(images,1)
        # images=torchvision.transforms.functional.resize(images,size=(150,150))
        # images=torchvision.transforms.functional.rotate(images,90)
        return img2, degrees[index]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, label = self.data[idx]
        image2, degree = self.random_augmentation(image)
        return image, image2, degree


import torch.nn.functional as F


class KeyEqGroup(torch.nn.Module):
    def __init__(self, args) -> None:
        super(KeyEqGroup, self).__init__()

        r2_act = gspaces.Rot2dOnR2(N=args.group_size)

        self.pyramid_levels = args.pyramid_levels
        self.feat_type_in = nn.FieldType(r2_act, args.num_channels * [
            r2_act.trivial_repr])  ## input 1 channels (gray scale image)

        feat_type_out1 = nn.FieldType(r2_act, args.dim_first * [r2_act.regular_repr])
        feat_type_out2 = nn.FieldType(r2_act, args.dim_second * [r2_act.regular_repr])
        feat_type_out3 = nn.FieldType(r2_act, args.dim_third * [r2_act.regular_repr])

        feat_type_ori_est = nn.FieldType(r2_act, [r2_act.regular_repr])

        self.block1 = nn.SequentialModule(
            nn.R2Conv(self.feat_type_in, feat_type_out1, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(feat_type_out1),
            nn.ReLU(feat_type_out1, inplace=True)
        )
        self.block2 = nn.SequentialModule(
            nn.R2Conv(feat_type_out1, feat_type_out2, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(feat_type_out2),
            nn.ReLU(feat_type_out2, inplace=True)
        )
        self.block3 = nn.SequentialModule(
            nn.R2Conv(feat_type_out2, feat_type_out3, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(feat_type_out3),
            nn.ReLU(feat_type_out3, inplace=True)
        )

        self.ori_learner = nn.SequentialModule(
            nn.R2Conv(feat_type_out3, feat_type_ori_est, kernel_size=1, padding=0, bias=False)
            ## Channel pooling by 8*G -> 1*G conv.
        )

        self.gpool = nn.GroupPooling(feat_type_out3)

        self.softmax = torch.nn.Softmax(dim=1)
        self.last_layer_learner = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=2 * self.pyramid_levels),
            torch.nn.Conv2d(in_channels=2 * self.pyramid_levels, out_channels=1, kernel_size=1, bias=True),
            torch.nn.ReLU(inplace=True)  ## clamp to make the scores positive values.
        )

        self.exported = False

    def forward(self, input_data):
        return self.compute_features(input_data)

    def compute_features(self, input_data):
        B, C, H, W = input_data.shape
        # print("Shape ",B,C,H,W)
        for idx_level in range(self.pyramid_levels):
            features_t, features_o = self._forward_network(input_data)

            features_t = F.interpolate(features_t, size=(H, W), align_corners=True, mode='bilinear')
            features_o = F.interpolate(features_o, size=(H, W), align_corners=True, mode='bilinear')
            if idx_level == 0:
                features_key = features_t
                features_ori = features_o
            else:
                features_key = torch.cat([features_key, features_t], axis=1)  # concatena no eixo X
                features_ori = torch.add(features_ori, features_o)  # somatorio dos kernels


        features_key = self.last_layer_learner(features_key)
        features_ori = self.softmax(features_ori)
        # print("Shape 3# ",idx_level,features_key.shape, features_ori.shape)
        return features_key, features_ori

    def _forward_network(self, input_data_resized):
        features_t = nn.GeometricTensor(input_data_resized,
                                        self.feat_type_in) if not self.exported else input_data_resized
        features_t = self.block1(features_t)
        features_t = self.block2(features_t)
        features_t = self.block3(features_t)

        # orientação
        features_o = self.ori_learner(features_t)  ## self.cpool
        features_o = features_o.tensor if not self.exported else features_o

        # keypoint
        features_t = self.gpool(features_t)
        features_t = features_t.tensor if not self.exported else features_t

        return features_t, features_o

from scipy.ndimage import maximum_filter
class KeyPointsSelection:
  def __init__(self,show=True) -> None:
     self.is_show = show

  def apply_nms(self,score_map, size):
      filter = maximum_filter(score_map, footprint=np.ones((size, size)))
      filter_t = torch.tensor(filter)
      score_map_temp = score_map * (score_map == filter_t)

      return score_map_temp


  def find_index_higher_scores(self,map, num_points = 1000, threshold = -1):
      # Best n points
      if threshold == -1:

          flatten = map.flatten()
          order_array = np.sort(flatten)

          order_array = np.flip(order_array, axis=0)

          if order_array.shape[0] < num_points:
              num_points = order_array.shape[0]

          threshold = order_array[num_points-1]

          if threshold <= 0.0:
              ### This is the problem case which derive smaller number of keypoints than the argument "num_points".
              indexes = np.argwhere(order_array > 0.0)

              if len(indexes) == 0:
                  threshold = 0.0
              else:
                  threshold = order_array[indexes[len(indexes)-1]]

              threshold = torch.tensor(threshold)

      indexes = np.argwhere(map >= threshold)

      return indexes[:num_points]


  def get_point_coordinates(self,map, scale_value=1., num_points=1000, threshold=-1, order_coord='xysr'):
      ## input numpy array score map : [H, W]
      indexes = self.find_index_higher_scores(map, num_points=num_points, threshold=threshold)
      new_indexes = []
      print(indexes.shape)
      for ind in indexes.T:
          scores = map[ind[0], ind[1]]
          if order_coord == 'xysr':
              tmp = [ind[1], ind[0], scale_value, scores]
          elif order_coord == 'yxsr':
              tmp = [ind[0], ind[1], scale_value, scores]

          new_indexes.append(tmp)

      indexes = np.asarray(new_indexes)

      return np.asarray(indexes)

  def __call__(self,img,win_size,num_points):
    print(img.shape,win_size,num_points)
    img_nms = self.apply_nms(img,win_size)
    points= self.get_point_coordinates(img_nms,num_points=num_points)
    if self.is_show:
      print(img_nms.min(), img_nms.max(),img_nms.mean())
      plt.imshow(img_nms)
      plt.show()
      plt.imshow(img)
      plt.plot(points[:,0], points[:,1], 'ro')
      plt.show()
    return points


def remove_borders(images, borders):
    ## input [B,C,H,W]
    shape = images.shape

    if len(shape) == 4:
        for batch_id in range(shape[0]):
            images[batch_id, :, 0:borders, :] = 0
            images[batch_id, :, :, 0:borders] = 0
            images[batch_id, :, shape[2] - borders:shape[2], :] = 0
            images[batch_id, :, :, shape[3] - borders:shape[3]] = 0
    elif len(shape) == 2:
        images[ 0:borders, :] = 0
        images[ :, 0:borders] = 0
        images[ shape[0] - borders:shape[0], :] = 0
        images[ :, shape[1] - borders:shape[1]] = 0
    else:
        print("Not implemented")
        exit()

    return images

import torchvision

def feature_detector_ajust(y1,y2,angles):
  idx_filter =1
  _y1 = y1.clone()
  for i in range(_y1.shape[0]):
    _y1[i] = torchvision.transforms.functional.rotate(y1[i],angles[i].item())
  return _y1,y2

class kp_loss:
  def __init__(self) -> None:
     self.basic = torch.nn.MSELoss()

  def __call__(self, im1,im2):
     _std = im1.std()+0.01
     return self.basic(im1,im2)/_std


from tqdm import tqdm
from skimage.transform import rotate
import torch.optim as optim
import gc

def train(model, train_loader, optimizer, epoch):
    model.train()
    criterion = kp_loss()

    running_loss = 0.

    for batch_idx, (im1, im2, ang) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        # prever detector/descritor
        _kp1, _orie1 = model(im1.to(device))
        _kp2, _orie2 = model(im2.to(device))

        _y1, y2 = feature_detector_ajust(_kp1, _kp2, ang.numpy())
        loss = criterion(_y1, y2)

        loss.backward()
        item = loss.item()
        running_loss += item
        # print(item,running_loss)
        optimizer.step()

    print('Train Epoch: {} \tLoss: {:.15f}'.format(
        epoch, running_loss))
    model.eval()
    with torch.no_grad():
      save_model(model,optimizer,1,0,path=MODEL_PATH)
    del im1
    del im2
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    BATCH_SIZE =8
    trainset = MyFlowersDataset(train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)


    testset = MyFlowersDataset(train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)
    x = torch.randn(2, 3, 8,8)
    model = KeyEqGroup(args).to(device)
    _k,_o = model(x.to(device))
    print("shape keypoints ",_k.shape,"shape orientation",_o.shape)

    # model=train(model,args)
    torch.manual_seed(0)

    trainset = MyFlowersDataset(train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)



    model = model.to(device)
    torch.manual_seed(0)
    args.epochs = 10
    epoch_last,loss = 0, 0
    optimizer = optim.Adam(model.parameters(),lr=0.0001,weight_decay=0.00001)

    try:
      model,optimizer,epoch_last,loss =load_model(model,optimizer,path=MODEL_PATH)
      print("Já foi treinado e",epoch_last)
    except:
      print("Não foi treinado ainda")
      # print(loss,epoch,optimizer)
    for epoch in range(epoch_last, args.epochs):
        train(model, trainloader, optimizer, epoch)
        # print(test_accuracy)

