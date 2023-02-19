import torch
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
import kornia as K

from antigo import remove_borders, KeyPointsSelection
from training import KeyEqGroup
from utils import save_model, load_model, imread, imshow

from config import args, device
from utils.TensorImgIO import imshow2

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


img1 = imread('./data/arturito.jpg')
img2 = imread('./data/simba.png')
resize = K.augmentation.Resize((args.img_size, args.img_size))
#resize = transforms.Resize(size=(196, 196))

img1, img2 = resize(img1), resize(img2)

img_vis = torch.cat([img1, img2], dim=-1)
imshow(img_vis)

img_batch = torch.cat((img1,img2), dim=0)
img_batch = img_batch.to(device)


model.eval()
_kp1,_orie1 = model(img_batch)
print(_kp1.shape,_orie1.shape)
img_feat_kp = torch.cat([_kp1[0], _kp1[1]], dim=-1)
imshow(img_feat_kp)

_kp2,_orie1 = model(img2.to(device))
print(_kp2.shape,_orie1.shape)
imshow(_kp2[0])

_kp3,_orie1 = model(img1.to(device))
print(_kp3.shape,_orie1.shape)
imshow(_kp3[0])

select = KeyPointsSelection()

img_temp = _kp3[0][0]
img_temp = remove_borders(img_temp, 3)

points = select(img_temp.detach().cpu(), 10, 70)

imshow2(img1,points)