import numpy as np
import torch
from e2cnn import gspaces, nn
import torch.nn.functional as F
from kornia import filters


class KeyEqGroup(torch.nn.Module):
    def __init__(self, args) -> None:
        super(KeyEqGroup, self).__init__()

        r2_act = gspaces.Rot2dOnR2(N=args.group_size)

        self.pyramid_levels = args.pyramid_levels
        self.scale = args.scale_pyramid
        self.feat_type_in = nn.FieldType(r2_act, args.num_channels * [
            r2_act.trivial_repr])  ## input 1 channels (gray scale image)

        feat_type_out1 = nn.FieldType(r2_act, args.dim_first * [r2_act.regular_repr])
        feat_type_out2 = nn.FieldType(r2_act, args.dim_second * [r2_act.regular_repr])
        feat_type_out3 = nn.FieldType(r2_act, args.dim_third * [r2_act.regular_repr])

        feat_type_ori_est = nn.FieldType(r2_act, [r2_act.regular_repr])

        self.block1 = nn.SequentialModule(
            nn.R2Conv(self.feat_type_in, feat_type_out1, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(feat_type_out1),
            nn.ReLU(feat_type_out1, inplace=True),
            nn.PointwiseMaxPoolAntialiased(feat_type_out1,kernel_size=2)
        )
        self.block2 = nn.SequentialModule(
            nn.R2Conv(feat_type_out1, feat_type_out2, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(feat_type_out2),
            nn.ReLU(feat_type_out2, inplace=True)
        )
        self.block3 = nn.SequentialModule(
            nn.R2Conv(feat_type_out2, feat_type_out3, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(feat_type_out3),
            nn.ReLU(feat_type_out3, inplace=True),
        )

        self.ori_learner = nn.SequentialModule(
            nn.R2Conv(feat_type_out3, feat_type_ori_est, kernel_size=1, padding=0, bias=False)
            ## Channel pooling by 8*G -> 1*G conv.
        )

        self.gpool = nn.GroupPooling(feat_type_out3)

        self.softmax = torch.nn.Softmax(dim=1)
        self.last_layer_learner = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=args.dim_third * self.pyramid_levels),
            torch.nn.Conv2d(in_channels=args.dim_third * self.pyramid_levels, out_channels=1, kernel_size=1, bias=True),
            torch.nn.ReLU(inplace=True)  ## clamp to make the scores positive values.
        )

        self.exported = args.exported

    def forward(self, input_data):
        return self.compute_features(input_data)

    def compute_features(self, input_data):
        B, C, H, W = input_data.shape
        # print("Shape ",B,C,H,W)
        for idx_level in range(-1,self.pyramid_levels-1):
            with torch.no_grad():
                input_data_resized = self.resize_pyramid(idx_level,input_data)

            features_t, features_o = self._forward_network(input_data_resized)
            # print("#1 features_t ",features_t.shape)
            features_t = F.interpolate(features_t, size=(H, W), align_corners=True, mode='bilinear')
            features_o = F.interpolate(features_o, size=(H, W), align_corners=True, mode='bilinear')
            # print("#2 features_t ", features_t.shape)
            if idx_level == -1:
                features_key = features_t
                features_ori = features_o
            else:
                features_key = torch.cat([features_key, features_t], axis=1)  # concatena no eixo X
                features_ori = torch.add(features_ori, features_o)  # somatorio dos kernels
            # print("Shape 2# ",idx_level,features_key.shape, features_ori.shape)

        # print("#3 features_key ", features_key.shape)
        features_key = self.last_layer_learner(features_key)
        features_ori = self.softmax(features_ori)
        # print("#4 features_key ", features_key.shape)
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

        # print("@@", features_t.shape)
        # keypoint
        features_t = self.gpool(features_t)
        features_t = features_t.tensor if not self.exported else features_t
        # print(features_t.shape)
        return features_t, features_o

    def resize_pyramid(self,idx_level,input_data):
        gaussian = filters.GaussianBlur2d((3, 3), (1.5, 1.5))
        input_data_blur = gaussian(input_data)

        size = np.array(input_data.shape[-2:])
        new_size = (size / (1.25 ** idx_level)).astype(int)

        input_data_resized = F.interpolate(input_data_blur, size=tuple(new_size), align_corners=True, mode='bilinear')
        return input_data_resized