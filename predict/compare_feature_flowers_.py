import torch
import torchvision

from predict.predict_utils import load_model_trained, predict_single_points,predict_triplets_show_points
from config import args, device
from torchvision.transforms import transforms, InterpolationMode

from utils.my_dataset import FibersDataset
def custom_config(args):
    args.img_size = 180
    args.dim_first = 2
    args.dim_second = 3
    args.dim_third = 4
    args.batch_size = 1
    args.nms_size = 40 # normal é 10, alterei so pra avaliar o match
    args.is_loss_ssim = False
    args.show_feature = True
    return args

if __name__ == '__main__':
    args = custom_config(args)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    MODEL_PATH = "../data/models/model_flowers_ssim.pt"

    testset = torchvision.datasets.Flowers102(root='../data/datasets', split='test',
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    model = load_model_trained(args,device,path=MODEL_PATH)
    batch = next(iter(testloader))
    batch_result, _orie1_summary =predict_single_points(model,batch)
    print(len(batch_result),len(batch_result[0]))
