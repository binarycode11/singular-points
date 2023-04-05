import torch

from predict.predict_utils import load_model_trained, predict_single_points
from config import args, device
from torchvision.transforms import transforms, InterpolationMode

from utils.my_dataset import FibersDataset
def custom_config(args):
    args.img_size = 250
    args.dim_first = 2
    args.dim_second = 3
    args.dim_third = 4
    args.batch_size = 5
    args.margin_loss = 2.0
    return args

if __name__ == '__main__':
    args = custom_config(args)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    MODEL_PATH = "../data/models/model_fibers.pt"
    testset = FibersDataset(transform=transform, train=False, path='../data/datasets/fibers/')
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    model = load_model_trained(args,device,path=MODEL_PATH)
    predict_single_points(model,testloader)