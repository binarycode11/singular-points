"""
Este arquivo contém funções e classes para trabalhar com detecção de características e correspondências de imagens usando PyTorch e Kornia.
As principais funções incluem a criação de modelos de detecção personalizados, geração de parâmetros de aumento de dados, leitura de dados de flores,
carregamento e salvamento de modelos, cálculo de homografia e correspondências bidirecionais de características.
"""

import kornia
import torch
import cv2
import math
import torchvision
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.transforms import transforms, InterpolationMode
from kornia.feature.scale_space_detector import get_default_detector_config, MultiResolutionDetector
import kornia

from utils.my_dataset import FibersDataset, WoodsDataset

PS =19#19 hardnet 32


ori_module=kornia.feature.LAFOrienter(PS)
aff_module=kornia.feature.LAFAffineShapeEstimator(PS)

keynet_default_config = {
    'num_filters': 8,
    'num_levels': 3,
    'kernel_size': 5,
    'Detector_conf': {'nms_size': 5, 'pyramid_levels': 2, 'up_levels': 1, 'scale_factor_levels': 1.3, 's_mult': 15.0},
}

class CustomNetDetector(MultiResolutionDetector):
    def __init__(
        self,
        model,
        pretrained: bool = False,
        num_features: int = 60,
        keynet_conf=keynet_default_config,
        PS=PS
    ):
        ori_module=kornia.feature.LAFOrienter(PS)#kornia.feature.LAFOrienter(PS)
        aff_module=None#kornia.feature.LAFAffineShapeEstimator(PS)
        super().__init__(model, num_features, keynet_conf['Detector_conf'], ori_module, aff_module)

class AugmentationParamsGenerator:
    def __init__(self, n, shape):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        self.aug_list = kornia.augmentation.AugmentationSequential(
            kornia.augmentation.RandomAffine(degrees=360, translate=(0.2, 0.2), scale=(0.95, 1.05), shear=10,p=0.8),
            kornia.augmentation.RandomPerspective(0.2, p=0.7),
            kornia.augmentation.RandomBoxBlur((4,4),p=0.5),
            # kornia.augmentation.RandomEqualize(p=0.3),
            data_keys=["input"],
            same_on_batch=True,
            # random_apply=10,
        )

        self.index = 0
        self.data = []
        for i in range(n):
            out = self.aug_list(torch.rand(shape))
            self.data.append(self.aug_list._params)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            self.index = 0  # Reset index to start over for circular iteration

        result = self.data[self.index]
        self.index += 1
        return result

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath,device):
    model.load_state_dict(torch.load(filepath,map_location=device))
    model.eval()
    print(f"Model loaded from {filepath}")


def fixed_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dataload_flower(img_size,data_path='./data/datasets',batch_size=60):
    transform2 = transforms.Compose([
        transforms.Resize((img_size,img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    trainset = torchvision.datasets.Flowers102(root=data_path, split='train',
                                            download=True, transform=transform2)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    testset = torchvision.datasets.Flowers102(root=data_path, split='test',
                                            download=True, transform=transform2)

    num_datapoints_to_keep = math.ceil(len(testset) / 2)
    num_datapoints_to_keep = 1020
    indices_to_keep = torch.randperm(num_datapoints_to_keep)[:num_datapoints_to_keep]
    reduced_testset = torch.utils.data.Subset(testset, indices_to_keep)
    testloader = torch.utils.data.DataLoader(reduced_testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    return trainloader,testloader


def read_dataload_fibers(img_size,data_path='./data/datasets/fibers/',batch_size=44):
    transform2 = transforms.Compose([
        transforms.Resize((img_size,img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    trainset = FibersDataset(transform=transform2, train=True, path=data_path)
    testset = FibersDataset(transform=transform2, train=False, path=data_path)
    print(len(testset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
    return trainloader,testloader

def read_dataload_woods(img_size,data_path='./data/datasets/woods/',batch_size=31):
    transform2 = transforms.Compose([
        transforms.Resize((img_size,img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    trainset = WoodsDataset(transform=transform2, train=True, path=data_path,limit_train=0.301)
    testset = WoodsDataset(transform=transform2, train=False, path=data_path,limit_train=0.301)
    print(len(testset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
    return trainloader,testloader


def compute_homography(lafs1, lafs2, matches):
    src_pts = lafs1[0, matches[:, 0], :, 2].data.cpu().numpy()
    dst_pts = lafs2[0, matches[:, 1], :, 2].data.cpu().numpy()
    F, inliers_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0, 0.999, 1000)
    inliers_point = matches[torch.from_numpy(inliers_mask).bool().squeeze(), :]
    return inliers_mask

def bidirectional_match(feat1, feat2, threshold=0.5):
    feat1 = feat1.float()
    feat2 = feat2.float()

    s1, matches1 = kornia.feature.match_snn(feat1, feat2, threshold)
    s2, matches2 = kornia.feature.match_snn(feat2, feat1, threshold)

    bidirectional_matches = []
    for i, match in enumerate(matches1):
        indices = torch.where(matches2[:, 0] == match[1].item())[0]
        if indices.numel() > 0:
            for index in indices:
                if matches2[index][1].item() == match[0].item():
                     bidirectional_matches.append((match[0].item(), match[1].item()))
    return torch.tensor(bidirectional_matches)