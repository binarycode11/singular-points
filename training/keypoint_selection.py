import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.ndimage.filters import maximum_filter
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
    print('selection ',img.shape,win_size,num_points)
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


import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import maximum_filter

#TODO refatorar
class NMSHead(nn.Module):
    def __init__(self, nms_size=5, nms_min_val=1e-5):
        super(NMSHead, self).__init__()
        self.nms_size = nms_size
        self.min_val = nms_min_val

    def _get_max_coords(self, input_map):
        """ _get_max_coords uses NMS of window size (self.nms_size) and
        threshold (self.min_val) to extract the coordinates of the local
        maximums in the input_map.
        :param input_map: (H, W) tensor of the map
        :return tensor: (2, N) tensor of the local maximum coordinates
        """
        assert input_map.ndim == 2, 'invalid input shape'
        device = input_map.device
        input_map = input_map.cpu().numpy()
        max_map = maximum_filter(input_map, size=self.nms_size, mode='constant')
        max_coords = np.stack(
            ((max_map > self.min_val) & (max_map == input_map)).nonzero()
        )
        plt.imshow(((max_map > 1e-5) & (max_map == input_map)))
        plt.show()

        return torch.tensor(max_coords).flip(0).to(device)  # flip axes

    def get_max_coords(self, input_maps):
        return [self._get_max_coords(input_map[0]) for input_map in input_maps]

    def forward(self, input_map):
        bs = len(input_map)

        # non-maximum suppression
        bev_coords = self.get_max_coords(input_map)
        n_coords = [coord.shape[1] for coord in bev_coords]
        bev_pixels = torch.zeros(bs, 3, max(n_coords)).to(input_map.device)
        bev_pixels[:, 2] = 1

        for i, bev_coord in enumerate(bev_coords):
            bev_pixels[i, :2, :bev_coord.shape[1]] = bev_coord

        # convert to world scale
        bev_size = input_map.shape[2:]
        return bev_pixels

#TODO refatorar
def bound_box(points, size=14):
    boxs = []
    for point in points:
        [x, y] = point[:2]
        x, y = int(x), int(y)
        box = [(x - size, y - size), (x + size, y + size)]
        boxs.append(box)
    return boxs

#TODO refatorar
def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask

#TODO refatorar
def apply_circular_mask_box(img, box):
    pt1, pt2 = box
    crop_img = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    h, w = crop_img.shape
    mask = torch.tensor(create_circular_mask(h, w))

    masked_img = crop_img.clone()
    masked_img[~mask] = 0
    return masked_img

#TODO refatorar
def get_features(img_batch, coords):
    for i, img in enumerate(img_batch):
        points = torch.stack((coords[i, 0, :], coords[i, 1, :]), axis=-1)
        boxs = bound_box(points, size=30)

        for i, box in enumerate(boxs):
            print(i)
            masked_img = apply_circular_mask_box(img.clone(), box)
            plt.imshow(masked_img.detach())
            plt.show()


#TODO Sample usando NMS

# size =30
# feat_temp = torch.cat([_k,_k])
# feat_temp = remove_borders(feat_temp,size)
# plt.imshow(feat_temp[0][0].detach())
# plt.show()
#
# nms =NMSHead(nms_size=size)
# coords = nms.forward(feat_temp.clone().detach())

# ori_batch = torch.cat([ori_arg_max,ori_arg_max])
#
# get_features(ori_batch[:1,:,:],coords[:1,:,:])