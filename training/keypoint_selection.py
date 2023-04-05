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
