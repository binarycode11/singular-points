import numpy as np
import torch
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter
from torch import nn
from torchvision.transforms import transforms, InterpolationMode
import kornia as K

from antigo import remove_borders, KeyPointsSelection
from training import KeyEqGroup
from utils import save_model, load_model, imread, imshow

from config import args, device
from utils.TensorImgIO import imshow2
from utils.my_dataset import FibersDataset

MODEL_PATH = "model_train.pt"
model = KeyEqGroup(args).to(device)
i_epoch = 0
loss = 0
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
try:

    model, optimizer, i_epoch, loss = load_model(model, optimizer,path=MODEL_PATH)
    print("Já foi treinado")
except:
    print("Não foi treinado ainda")


transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = FibersDataset(transform=transform,train=False,path='./data/fibers/')
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)


model.eval()
for img_batch,labels in testloader:
    img_feat_kp = torch.cat([img_batch[0], img_batch[1]], dim=-1)
    imshow(img_feat_kp)
    _kp1,_orie1 = model(img_batch.to(device))

    _kp1 = remove_borders(_kp1,15)
    select = KeyPointsSelection()
    points = select(_kp1[0][0].detach().cpu(), 20, 100)
    imshow2(img_batch[0],points)

    points = select(_kp1[1][0].detach().cpu(), 20, 100)
    imshow2(img_batch[1],points)
    print("teste")