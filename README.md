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


## Problemas:

- FLOWERS
  - args.img_size = 180
  - args.dim_first = 2
  - args.dim_second = 3
  - args.dim_third = 4
  - args.batch_size = 10
  - args.is_loss_ssim = True
- FIBERS
  - args.img_size = 250
  - args.dim_first = 2
  - args.dim_second = 3
  - args.dim_third = 4
  - args.batch_size = 5
  - args.margin_loss = 2.0
  - args.is_loss_ssim = False
- WOODS
  - args.img_size = 180
  - args.dim_first = 2
  - args.dim_second = 3
  - args.dim_third = 4
  - args.batch_size = 10
  - args.margin_loss = 2.0
  - args.is_loss_ssim = True
  - args.epochs = 100