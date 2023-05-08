import torchvision
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from train_basic import *
from training import KeyEqGroup
from config import args
from utils import load_model,save_model

from torchvision.transforms import transforms, InterpolationMode
from training import KeyEqGroup,triplet_loss as loss_function
from utils.my_dataset import FibersDataset, WoodsDataset

MODEL_PATH = "../data/models/model_woods_ssim.pt"
args.img_size = 180
args.dim_first = 2
args.dim_second = 3
args.dim_third = 4
args.batch_size = 10
args.margin_loss = 2.0
args.is_loss_ssim = True
args.epochs = 50

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = WoodsDataset(transform=transform, train=True, path='../data/datasets/woods/')
    testset = WoodsDataset(transform=transform, train=False, path='../data/datasets/woods/')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)


    model = KeyEqGroup(args).to(device)
    i_epoch = 0
    loss = 0
    # optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    # scheduler = ExponentialLR(optimizer, gamma=0.90)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = ExponentialLR(optimizer, gamma=0.75)
    try:
        model, optimizer, i_epoch, loss = load_model(model, optimizer, path=MODEL_PATH)
        print("Já foi treinado")
    except:
        print("Não foi treinado ainda")

    torch.manual_seed(0)
    print("epoca {} loss {}".format(i_epoch, loss))
    for epoch in range(i_epoch, args.epochs):
        criterion = loss_function(is_ssim=args.is_loss_ssim, margim=args.margin_loss)
        running_loss = train_one_epoch(model, trainloader, optimizer=optimizer, criterion=criterion,epoch=epoch, is_show=False)
        save_model(model, optimizer, epoch, running_loss, path=MODEL_PATH)
        valid_loss = running_loss
        valid_loss = test(model, testloader, criterion=criterion,epoch=epoch)
        if epoch % 5 == 0:
            scheduler.step()
        print(optimizer.param_groups[0]['initial_lr'], optimizer.param_groups[0]['lr'])

        if valid_loss <= 0.3:
            break