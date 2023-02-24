import torch
import torch.optim as optim
import torchvision
from PIL.Image import Image
from torchvision.transforms import transforms, InterpolationMode

from training import KeyEqGroup, random_augmentation
from utils import save_model,load_model, imshow

from config import args, device
MODEL_PATH = "model_train.pt"

transform = transforms.Compose([
        transforms.Resize((200,200),interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

trainset = torchvision.datasets.Flowers102(root='./data', split='train',
                                           download=True, transform=transform)
testset = torchvision.datasets.Flowers102(root='./data', split='test',
                                          download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)



model = KeyEqGroup(args)
# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("---")
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

running_loss = 0.

class kp_loss:
    def __init__(self) -> None:
        self.basic = torch.nn.MSELoss()

    def __call__(self, im1, im2):
        _std = im1.std() + 0.01
        return self.basic(im1, im2) / _std

from tqdm import tqdm
from skimage.transform import rotate
import torch.optim as optim
import gc

def train(model, train_loader, optimizer, epoch,is_show=True):
    model.train()
    criterion = kp_loss()

    running_loss = 0.
    t = tqdm(train_loader,desc="Train Epoch:{} ".format(epoch))
    for batch_idx, (batch_image, batch_label) in enumerate(t):
        optimizer.zero_grad()
        # prever detector/descritor
        batch_image = batch_image.to(device)
        _kp1, _orie1 = model(batch_image)
        points = torch.randn(batch_image.shape[0], 2, 2)  # BxNx2 [x,y]
        imgs_trans, feature_kp_trans, features_ori_trans, coords_trans, mask_trans = random_augmentation(batch_image,
                                                                                                         _kp1, _orie1,
                                                                                                         points)
        _kp2, _orie2 = model(imgs_trans)

        if is_show and batch_idx%25==0:
            temp = torch.cat([batch_image[3],imgs_trans[3]], dim=-1)
            imshow(temp)
            temp = torch.cat([feature_kp_trans[3], _kp2[3]], dim=-1)
            imshow(temp)

        # print(batch_image[0].min(), batch_image[0].max(), batch_image[0].mean())
        # print(imgs_trans[0].min(), imgs_trans[0].max(), imgs_trans[0].mean())
        loss = criterion(feature_kp_trans, _kp2)
        # loss = criterion(feature_kp_trans, _kp2,mask_trans.to(device))

        loss.backward()
        item = loss.item()
        running_loss += item
        t.set_description("Train Epoch:{} Loss: {:.5f}".format(epoch,running_loss))
        optimizer.step()

    model.eval()
    print('Train Epoch: {} \tLoss: {:.15f}'.format(
        epoch, running_loss))
    save_model(model, optimizer, epoch, running_loss,path=MODEL_PATH)
    del batch_image
    del batch_label
    gc.collect()
    torch.cuda.empty_cache()


def test(model, test_loader):
    model.eval()
    criterion = kp_loss()

    running_loss = 0.
    t = tqdm(test_loader,desc="Test Epoch: {} ".format(epoch))
    for batch_idx, (batch_image, batch_label) in enumerate(t):
        batch_image = batch_image.to(device)
        _kp1, _orie1 = model(batch_image)

        points = torch.randn(batch_image.shape[0], 2, 2)  # BxNx2 [x,y]

        imgs_trans, feature_kp_trans, features_ori_trans, coords_trans, mask_trans = \
            random_augmentation(batch_image,_kp1, _orie1,points)
        _kp2, _orie2 = model(imgs_trans)

        # if batch_idx%25==0:
        #     temp = torch.cat([batch_image[0],imgs_trans[0]], dim=-1)
        #     imshow(temp)
        #     temp = torch.cat([feature_kp_trans[0], _kp2[0]], dim=-1)
        #     imshow(temp)

        loss = criterion(feature_kp_trans, _kp2)
        item = loss.item()
        running_loss += item
        t.set_description("Test Epoch:{} Loss: {:.5f}".format(epoch, running_loss))

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


print("epoca {} loss {}".format(i_epoch, loss))
for epoch in range(i_epoch,args.epochs):
    train(model, trainloader, optimizer, epoch,is_show=True)
    test(model, testloader)
    # print(test_accuracy)




