# singular-points


# Downloads
- !wget https://github.com/kornia/data/raw/main/simba.png
- !wget https://github.com/kornia/data/raw/main/arturito.jpg

## Install
- pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
- pip3 install kornia e2cnn matplotlib scikit-image 


# arquivos

- antigo : sem o kornia
  - file model: 'model_antigo.pt'
- train : esta executando com pessima performance
  - file model: 'model_train.pt'


## Configuração:

### train 1
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = ExponentialLR(optimizer, gamma=0.75)
    triplet_loss

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

     augme_affim =K.augmentation.RandomAffine(degrees=[-90, 90],translate=[0.05,0.05], p=1.,same_on_batch=True,keepdim=True)
      augm_pespec =K.augmentation.RandomPerspective(distortion_scale=0.05, p=1.,same_on_batch=True,keepdim=True)
      aug_compost = K.augmentation.AugmentationSequential(
        augm_pespec,
        augme_affim,
        data_keys=["input","input", "input","keypoints","mask"]
      )

| Dataset | HXW | layer_1 | layer_2 | layer_3 | batch | margin | loss | Optim          | gamma | epoch | loss_rate |
|---------|-----|---------|---------|---------|-------|--------|------|----------------|-------|-------|-----------|
| Fibers  | 250 | 2       | 3       | 4       | 5     | 2      | NMRE | Adam_lr=0.0001 | 0.9   | 37    | 0.02      |
| Flowers | 180 | 2       | 3       | 4       | 10    | 2      | NMRE | Adam_lr=0.0001 | 0.9   | 7     | 4.78      |
| Woods   | 180 | 2       | 3       | 4       | 10    | 2      | NMRE | Adam_lr=0.0001 | 0.9   | 18    | 0.12      |
| Fibers  | 250 | 2       | 3       | 4       | 5     | 2      | SSIM | Adam_lr=0.01   | 0.75  | 35    | 1.62      |
| Flowers | 180 | 2       | 3       | 4       | 10    | 2      | SSIM | Adam_lr=0.01   | 0.75  | 39    | 1.63      |
| Woods   | 180 | 2       | 3       | 4       | 10    | 2      | SSIM | Adam_lr=0.01   | 0.75  | 31    | 1.63      |