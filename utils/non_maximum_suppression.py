
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter
from config import args
class NMSHead(nn.Module):
    """
    NMSHead is a module that performs non-maximum suppression on the input
    map. It takes the input map and returns the coordinates of the local
    maximums.
    """
    def __init__(self, nms_size=5, nms_min_val=1e-5,mask = None):
        super(NMSHead, self).__init__()
        self.nms_size = nms_size
        self.min_val = nms_min_val
        self.mask =mask

    def _get_max_coords(self, input_map):
        """ _get_max_coords uses NMS of window size (self.nms_size) and
        threshold (self.min_val) to extract the coordinates of the local
        maximums in the input_map.
        :param input_map: (H, W) tensor of the map
        :return tensor: (2, N) tensor of the local maximum coordinates
        """
        assert input_map.ndim == 2, 'invalid input shape'
        device = input_map.device

        if self.mask is not None:
            input_map = input_map * self.mask
        input_map = input_map.cpu().numpy()
        max_map = maximum_filter(input_map, size=self.nms_size, mode='constant')
        max_coords = np.stack(
            ((max_map > self.min_val) & (max_map == input_map)).nonzero()
        )
        #TODO exibi mapa de ativação
        # plt.imshow(max_map)
        # plt.show()
        return torch.tensor(max_coords).flip(0).to(device)  # flip axes

    def get_max_coords(self, input_maps):
        """
        Return the coordinates of the local maximums in the input_maps , inside a batch format.
        :param input_maps: list of (H, W) tensors
        :return: list of (2, N) tensors
        """
        return [self._get_max_coords(input_map[0]) for input_map in input_maps]

    def forward(self, input_map):
        """
        Return each coordinate of the local maxima along with its scale in the input_map.
        :param input_map: (B, H, W) tensor
        :return: (B, 3, N) tensor
        """
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