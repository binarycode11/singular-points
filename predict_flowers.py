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

# parser.add_argument("--num_channels", required=False, default=3, type=int)
# parser.add_argument("--pyramid_levels", required=False, default=3, type=int)# min 2
# parser.add_argument("--scale_pyramid", required=False, default=1.3, type=int)# min 2
# parser.add_argument("--dim_first", required=False, default=2, type=int)
# parser.add_argument("--dim_second", required=False, default=2, type=int)
# parser.add_argument("--dim_third", required=False, default=2, type=int)
# parser.add_argument("--group_size", required=False, default=36, type=int)
# parser.add_argument("--epochs", required=False, default=10, type=int)
# parser.add_argument("--img_size", required=False, default=200, type=int)
# parser.add_argument("--batch_size", required=False, default=6, type=int)
# parser.add_argument('--path_data', required=False, default='./data', type=str)
# parser.add_argument('--path_model', required=False, default='model.pt', type=str)
# parser.add_argument('--outlier_rejection', required=False, default=False, type=bool)

MODEL_PATH = "model_flowers.pt"
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
testset = torchvision.datasets.Flowers102(root='./data', split='test',
                                            download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)


img_batch,labels = next(iter(testloader))
model.eval()
_kp1,_orie1 = model(img_batch.to(device))


print(_kp1.shape,_orie1.shape)

img_feat_kp = torch.cat([img_batch[0], img_batch[1]], dim=-1)
imshow(img_feat_kp)

img_feat_kp = torch.cat([_kp1[0], _kp1[1]], dim=-1)
imshow(img_feat_kp)

_kp1 = remove_borders(_kp1,15)
select = KeyPointsSelection()
points = select(_kp1[0][0].detach().cpu(), 10, 100)
imshow2(img_batch[0],points)

points = select(_kp1[1][0].detach().cpu(), 10, 100)
imshow2(img_batch[1],points)
print("teste")