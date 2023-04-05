import torch.optim as optim
import kornia as K

from old.antigo import remove_borders, KeyPointsSelection
from training import KeyEqGroup
from utils import load_model, imread, imshow

from config import args, device
from utils.TensorImgIO import imshow2


def tensor_normalize(img):
    resize = K.augmentation.Resize((args.img_size, args.img_size))
    normalize = K.augmentation.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    aug_compost = K.augmentation.AugmentationSequential(
        resize,
        normalize,
        data_keys=["input"]
    )
    return aug_compost(img)



MODEL_PATH = "data/models/model_flowers.pt"
model = KeyEqGroup(args).to(device)
i_epoch = 0
loss = 0
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
try:

    model, optimizer, i_epoch, loss = load_model(model, optimizer,path=MODEL_PATH)
    print("Já foi treinado")
except:
    print("Não foi treinado ainda")


img11 = imread('./data/barra01.jpeg')
img22 = imread('./data/barra02.jpeg')

img1 = tensor_normalize(img11)
img2 = tensor_normalize(img22)

def random_aug(img):
  augme_affim =K.augmentation.RandomAffine(degrees=[-180, 180],translate=[0.05,0.05], p=1.,same_on_batch=True,keepdim=True)
  augm_pespec =K.augmentation.RandomPerspective(distortion_scale=0.2, p=1.,same_on_batch=True,keepdim=True)
  aug_compost = K.augmentation.AugmentationSequential(
    augm_pespec,
    augme_affim,
    data_keys=["input"]
  )
  return aug_compost(img)

def select_points_by_filter(img,borda=10,win=5,num_points=100):
    _kp1, _orie1 = model(img.to(device))
    imshow(_kp1[0])
    select = KeyPointsSelection()
    img_temp = remove_borders(_kp1[0][0], 10)
    points = select(img_temp.detach().cpu(), 5, 100)
    imshow2(img, points)



select_points_by_filter(img1)
img1_a = random_aug(img1)
imshow(img1_a)
select_points_by_filter(img1_a)

select_points_by_filter(img2)
img2_a = random_aug(img2)
imshow(img2_a)
select_points_by_filter(img2_a)