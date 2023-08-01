import torch
import kornia as K
from skimage import io
import numpy as np
from matplotlib import pyplot as plt
from config import device
def imread(data_path: str) -> torch.Tensor:
  img = io.imread(data_path)
  img_t = K.image_to_tensor(img)
  return img_t.float() / 255.

def imshow(input:torch.Tensor,coords=None):
    fig, ax = plt.subplots()
    if coords is not None:
      ax.add_patch(plt.Circle((coords[0,0], coords[0,1]), color='r'))
      ax.add_patch(plt.Circle((coords[1,0], coords[1,1]), color='r'))

    out_np: np.array = K.tensor_to_image(input)
    ax.imshow(out_np); ax.axis('off');plt.show()

def imshow2(input:torch.Tensor,coords=None):
    plt.plot(coords[:, 0], coords[:, 1], 'ro')
    out_np: np.array = K.tensor_to_image(input)
    plt.imshow(out_np); plt.axis('off');plt.show()

'''
    plot imagem quaisquer com os seus pontos detectados e 
    os labels para identificar cada ponto em outra foto correspondente
'''
#TODO refatorar
def imshow_points(img,coords):
    labels = np.arange(coords.shape[2])
    plt.imshow(img)#ori_arg_max[0].detach()
    plt.plot(coords[0, 0, :], coords[0, 1, :], 'r+')
    # # plot the points
    # plt.scatter(xs,ys)
    # zip joins x and y coordinates in pairs
    for i, (x, y) in enumerate(zip(coords[0, 0, :], coords[0, 1, :])):
        # label = f"({x},{y})"

        plt.annotate(labels[i],  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     color='white',
                     ha='center')  # horizontal alignment can be left, right or center
    plt.show()


'''
    Plota na janela as imagens originais (ancora, img positiva, img negativa)
                    os mapas de ativação correspondentes
                    e os pontos detectados
'''
def imshow3(originais:torch.Tensor,filtros:torch.Tensor,points=None):
    img_qtd = len(originais)
    fig, axs = plt.subplots(2, img_qtd,figsize=(14,4*img_qtd), sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0})

    m_atv1 = K.tensor_to_image(filtros[0])
    m_atv2 = K.tensor_to_image(filtros[1])

    for i in range(len(originais)):
        img = K.tensor_to_image(originais[i])
        mapa_atv = K.tensor_to_image(filtros[i])
        axs[0, i].imshow(img); axs[0, i].axis('off');
        axs[1, i].imshow(mapa_atv);axs[1, i].axis('off');
        coords = np.array([[x,y] for [x,y] in points[i] if x!=0 or y!=0])
        print('imshow3 ',type(coords),coords.shape)
        axs[1, i].plot(coords[:, 0], coords[:, 1], 'ro');

    plt.show()

def save_model(model,optimizer,epoch,running_loss,path='esqueci.pt'):
    state = {
        'epoch': epoch+1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': running_loss
    }
    torch.save(state,
               path)

def load_model(model_novo,optimizer_novo,path='esqueci.pt'):
    checkpoint = torch.load(path, map_location=torch.device(device))
    model_novo.load_state_dict(checkpoint['state_dict'])
    optimizer_novo.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model_novo,optimizer_novo,epoch,loss
