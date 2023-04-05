import torch.optim as optim
import kornia as K

from training import KeyEqGroup, KeyPointsSelection, remove_borders
from utils import load_model, imread, imshow

from config import args, device
from utils.TensorImgIO import imshow2






MODEL_PATH = "data/models/model_wood.pt"
args.img_size = 200
args.dim_first = 2
args.dim_second = 2
args.dim_third = 2
args.batch_size = 12
args.margin_loss = 2.0
args.is_loss_ssim = True


model = KeyEqGroup(args).to(device)
i_epoch = 0
loss = 0
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
try:

    model, optimizer, i_epoch, loss = load_model(model, optimizer,path=MODEL_PATH)
    print("Já foi treinado")
except:
    print("Não foi treinado ainda")


img11 = imread('data/datasets/woods/1_1.jpg')
img22 = imread('data/datasets/woods/1_3.jpg')

def tensor_normalize(img):
    resize = K.augmentation.Resize((args.img_size, args.img_size))
    normalize = K.augmentation.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    aug_compost = K.augmentation.AugmentationSequential(
        resize,
        normalize,
        data_keys=["input"]
    )
    return aug_compost(img)

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
    img_temp = remove_borders(_kp1[0][0], borda)
    points = select(img_temp.detach().cpu(), win, num_points)
    imshow2(img, points)



select_points_by_filter(img1)
img1_a = random_aug(img1)
imshow(img1_a)
select_points_by_filter(img1_a)

select_points_by_filter(img2)
img2_a = random_aug(img2)
imshow(img2_a)
select_points_by_filter(img2_a)