import torch
from matplotlib import pyplot as plt
from torch.nn.functional import normalize
from utils import create_interval_mask,intersection_filter_interval_mask

def sum_filtered_intensity(batch,filter):
    filter = filter[None].transpose(1,0)
    batch_filtered = batch * filter
    sum_batch = torch.sum(batch_filtered,dim=(2,3))
    return sum_batch,batch_filtered


def build_histogram_orientation(ori_arg_max,_kp1,mask,n_bin,show=False):
    #cria mascara para remocao de borda
    print(ori_arg_max.min(),ori_arg_max.max())
    print(type(mask),mask.shape)
    hist_batch =None
    for i in range(n_bin):
        v_bin = 360//n_bin
        int_0 = v_bin*i
        int_1 = v_bin*(i+1)
        interval_mask = create_interval_mask(ori_arg_max,int_0,int_1)
        mask_simples  = intersection_filter_interval_mask(interval_mask,torch.tensor(mask[None]))
        sum_b, batch_filtered = sum_filtered_intensity(_kp1,mask_simples)
        print('interval :',int_0," - ",int_1)
        if show:
            plt.imshow(batch_filtered[0,0].cpu().detach())
            plt.show()
        if hist_batch is None:
            hist_batch = sum_b
        else:
            hist_batch=torch.cat((hist_batch,sum_b),1)
    return  normalize(hist_batch*1.0, p=2, dim = 0)