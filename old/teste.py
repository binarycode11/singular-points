import torch

from e2cnn import gspaces
from e2cnn import nn

import numpy as np
from kornia.feature import HardNet8
from kornia.losses import ssim_loss,psnr_loss

from utils import imread, imshow
from utils.my_dataset import FibersDataset

np.set_printoptions(precision=3, linewidth=10000, suppress=True)

from matplotlib import pyplot as plt
import kornia as K


if __name__ == '__main__':

    torch.manual_seed(0)
    r2_act = gspaces.Rot2dOnR2(N=9)


    feat_type_in = nn.FieldType(r2_act, 3*[r2_act.trivial_repr])
    feat_type_hid = nn.FieldType(r2_act, 8*[r2_act.regular_repr])
    feat_type_out = nn.FieldType(r2_act, 1*[r2_act.regular_repr])


    model = nn.SequentialModule(
        nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=3),
        nn.InnerBatchNorm(feat_type_hid),
        nn.ReLU(feat_type_hid),
        nn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=3),
        nn.InnerBatchNorm(feat_type_hid),
        nn.ReLU(feat_type_hid),
        nn.R2Conv(feat_type_hid, feat_type_out, kernel_size=3),
        # nn.PointwiseAvgPool(feat_type_out, 3),
        #nn.PointwiseAdaptiveAvgPool(feat_type_out, 150),
        #nn.PointwiseMaxPoolAntialiased(feat_type_out, 5,(5,5)),
        nn.GroupPooling(feat_type_out)
    ).eval()


    resize = K.augmentation.Resize((200, 200))
    resize32 = K.augmentation.Resize((32, 32))

    x = imread('../data/arturito.jpg')
    x = nn.GeometricTensor(resize(x),feat_type_in)
    x2 = imread('../data/simba.png')
    x2 = nn.GeometricTensor(resize(x2), feat_type_in)

    imshow(x.tensor[0])
    y = model(x)

    imshow(x2.tensor[0])
    y2 = model(x2)

    print(y.shape)
    _B, _C, _W, _H = y.shape
    mask = nn.MaskModule(feat_type_in, _W, margin=5)
    g_mask = nn.GeometricTensor(mask.mask, nn.FieldType(r2_act, [r2_act.trivial_repr]))

    imshow(y.tensor[0][0])

    # for each group element
    hard = HardNet8()
    for g in r2_act.testing_elements:
        x_transformed = x.transform(g)
        imshow(x_transformed.tensor[0])
        y_from_x_transformed = model(x_transformed)

        mask_transformed = g_mask.transform(g)
        y_from_x_transformed = y_from_x_transformed.tensor * mask_transformed.tensor
        imshow(y_from_x_transformed[0][0])

        y_transformed_from_x = y.transform(g)
        y_transformed_from_x = y_transformed_from_x.tensor * mask_transformed.tensor
        imshow(y_transformed_from_x[0][0])

        imshow(y_transformed_from_x[0][0]-y_from_x_transformed[0][0])
        loss = ssim_loss(y_from_x_transformed,y_transformed_from_x,7)
        loss2 = ssim_loss(y_from_x_transformed, y.tensor* mask_transformed.tensor, 7)
        loss3 = ssim_loss(y.tensor* mask_transformed.tensor, y2.tensor * mask_transformed.tensor, 7)

        print('loss 1 ',loss,loss2,loss3)


        # assert torch.allclose(y_from_x_transformed.tensor, y_transformed_from_x.tensor, atol=1e-5), g

    input = torch.rand(16, 1, 32, 32)
    hardnet = HardNet8()
    descs = hardnet(input)  # 16x128
    print(descs.shape)