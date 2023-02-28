import torch
import torch.optim as optim
import torchvision
from PIL.Image import Image
from torchvision.transforms import transforms, InterpolationMode

from training import KeyEqGroup, random_augmentation
from training.keypoint_loss import triplet_loss as criteria_loss
from utils import save_model,load_model, imshow , MyFlowersDataset

from config import args, device
from utils.my_dataset import FibersDataset

MODEL_PATH = "model_flowers.pt"

def shifted_batch_tensor(batch_img,features_key,features_ori):
    batch_neg = torch.roll(batch_img, 1, 0)
    feat_key_neg = torch.roll(features_key, 1, 0)
    feat_ori_neg = torch.roll(features_ori, 1, 0)
    return batch_neg,feat_key_neg,feat_ori_neg

transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.Flowers102(root='./data', split='train',
                                            download=True, transform=transform)
testset = torchvision.datasets.Flowers102(root='./data', split='test',
                                            download=True, transform=transform)

# trainset = FibersDataset(transform=transform,train=True,path='./data/fibers/')
# testset = FibersDataset(transform=transform,train=False,path='./data/fibers/')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)



model = KeyEqGroup(args)
# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00005)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("---")
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

running_loss = 0.

from tqdm import tqdm
from skimage.transform import rotate
import torch.optim as optim
import gc

def train(model, train_loader, optimizer, epoch,is_show=True):
    model.train()
    criterion = criteria_loss()

    running_loss = 0.
    t = tqdm(train_loader,desc="Train Epoch:{} ".format(epoch))
    for batch_idx, (batch_image, labels) in enumerate(t):

        # prever detector/descritor
        batch_image = batch_image.to(device)
        _kp1, _orie1 = model(batch_image)
        points = torch.randn(batch_image.shape[0], 2, 2)  # BxNx2 [x,y]
        batch_image_pos_trans, feature_kp_anchor_trans, features_ori_anchor_trans, coords_trans, mask_trans = random_augmentation(batch_image,
                                                                                                         _kp1, _orie1,
                                                                                                         points)
        _kp2_pos, _orie2_pos = model(batch_image_pos_trans)
        batch_image_neg_trans, _kp2_neg, _orie2_neg = shifted_batch_tensor(batch_image_pos_trans,_kp2_pos,_orie2_pos)#faz o shift com o comando roll(x,1,0)

        if is_show and batch_idx%25==0:
            # temp = torch.cat([batch_image[3],batch_image_trans[3]@mask_trans[3]], dim=-1)
            temp = torch.cat([batch_image_pos_trans[3], batch_image_pos_trans[3] * mask_trans[3],batch_image_neg_trans[3] * mask_trans[3]], dim=-1)
            imshow(temp)
            temp = torch.cat([feature_kp_anchor_trans[3]* mask_trans[3][0], _kp2_pos[3]* mask_trans[3][0], _kp2_neg[3]* mask_trans[3][0]], dim=-1)
            imshow(temp)
            # temp = torch.cat([feature_kp_anchor_trans[3]* mask_trans[3][0], _kp2_neg[3]* mask_trans[3][0]], dim=-1)
            # imshow(temp)

        # print(batch_image[0].min(), batch_image[0].max(), batch_image[0].mean())
        # print(batch_image_trans[0].min(), batch_image_trans[0].max(), batch_image_trans[0].mean())
        loss = criterion(feature_kp_anchor_trans, _kp2_pos,_kp2_neg)
        # loss = criterion(feature_kp_trans, _kp2,mask_trans.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        item = loss.item()
        running_loss += item
        t.set_description("Train Epoch:{} Loss: {:.5f}".format(epoch, running_loss))

        del batch_image
        del batch_image_neg_trans
        del _kp1,_kp2_pos,_kp2_neg,_orie1,_orie2_pos,_orie2_neg
        gc.collect()
        torch.cuda.empty_cache()

    model.eval()
    print('Train Epoch: {} \tLoss: {:.15f}'.format(
        epoch, running_loss))
    save_model(model, optimizer, epoch, running_loss,path=MODEL_PATH)



def test(model, test_loader):
    model.eval()
    criterion = criteria_loss()

    running_loss = 0.
    t = tqdm(test_loader,desc="Test Epoch: {} ".format(epoch))
    for batch_idx, (batch_image, batch_label) in enumerate(t):
        batch_image = batch_image.to(device)
        _kp1, _orie1 = model(batch_image)

        points = torch.randn(batch_image.shape[0], 2, 2)  # BxNx2 [x,y]

        batch_image_pos_trans, feature_kp_anchor_trans, features_ori_anchor_trans, coords_trans, mask_trans = \
            random_augmentation(batch_image,_kp1, _orie1,points)
        _kp2_pos,_orie2_pos = model(batch_image_pos_trans)
        batch_image_neg_trans, _kp2_neg, _orie2_neg = shifted_batch_tensor(batch_image_pos_trans, _kp2_pos,
                                                                           _orie2_pos)  # faz o shift com o comando roll(x,1,0)


        loss = criterion(feature_kp_anchor_trans, _kp2_pos,_kp2_neg)
        item = loss.item()
        running_loss += item
        t.set_description("Test Epoch:{} Loss: {:.5f}".format(epoch, running_loss))

        del batch_image
        del batch_image_neg_trans
        del _kp1,_kp2_pos,_kp2_neg,_orie1,_orie2_pos,_orie2_neg
        gc.collect()
        torch.cuda.empty_cache()

    print('Test Epoch: {} \tLoss: {:.15f}'.format(
        epoch, running_loss))




model = KeyEqGroup(args).to(device)
i_epoch = 0
loss = 0
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.000001)
try:

    model, optimizer, i_epoch, loss = load_model(model, optimizer,path=MODEL_PATH)
    print("Já foi treinado")
except:
    print("Não foi treinado ainda")

torch.manual_seed(0)
print("epoca {} loss {}".format(i_epoch, loss))
for epoch in range(i_epoch,args.epochs):
    train(model, trainloader, optimizer, epoch,is_show=False)
    test(model, testloader)
    # print(test_accuracy)




