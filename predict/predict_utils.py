import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt

from config import args,device
from training import KeyEqGroup, KeyPointsSelection, remove_borders, random_augmentation, shifted_batch_tensor
from utils import load_model, imshow, imshow2, imshow3,NMSHead,get_features


import matplotlib
matplotlib.get_backend()
def compute_gradient_direction(orie_img_batch):
    _b,_na,_c,_r=orie_img_batch.shape #bacth,num degree,col,row
    ori_arg_max= torch.argmax(orie_img_batch, dim=1)
    bin_size = 360/_na
    ori_arg_max=ori_arg_max*bin_size # direcao do gradiente
                               # para cada pixel
    return ori_arg_max

def plot_orient_with_labels(ori_arg_max,coords):
    labels = np.arange(coords.shape[2])
    plt.imshow(ori_arg_max, aspect='auto')
    plt.plot(coords[0,0,:],coords[0,1,:], 'r+')

    # # plot the points
    # plt.scatter(xs,ys)

    # zip joins x and y coordinates in pairs
    for i, (x,y) in enumerate(zip(coords[0,0,:],coords[0,1,:])):

        # label = f"({x},{y})"

        plt.annotate(labels[i], # this is the text
                     (x,y), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     color='white',
                     ha='center') # horizontal alignment can be left, right or center

    plt.show()

def load_model_trained(args,device,path):
    model = KeyEqGroup(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    try:
        model, optimizer, i_epoch, loss = load_model(model, optimizer, path=path)
        print("Já foi treinado")
        print("epoca {} loss {}".format(i_epoch, loss))
    except:
        print("Não foi treinado ainda")
    return model

def predict_triplets_show_points(model,testloader):
    model.eval()
    with torch.no_grad():
        for img_batch, labels in testloader:
            img_batch = img_batch.to(device)

            _B, _C, _W, _H = img_batch.shape
            SIZE_BORDER = args.border_size
            batch_mask = torch.zeros(_B, 1, _W, _H).to(device)
            batch_mask[:, :, SIZE_BORDER:_W - SIZE_BORDER, SIZE_BORDER:_H - SIZE_BORDER] = 1

            # img_batch_2 = img_batch*batch_mask
            _kp1, _orie1 = model(img_batch)
            points = torch.randn(img_batch.shape[0], 2,
                                 2).to(device)  # BxNx2 [x,y] pontos sinteticos so pra completar parametros

            batch_image_pos_trans, feature_kp_anchor_trans, features_ori_anchor_trans, coords_trans, mask_trans = random_augmentation(
                img_batch,
                _kp1, _orie1,
                points,
                batch_mask)
            _kp2_pos, _orie2_pos = model(batch_image_pos_trans)
            batch_image_neg_trans, _kp2_neg, _orie2_neg = shifted_batch_tensor(batch_image_pos_trans, _kp2_pos,
                                                                               _orie2_pos)
            # h, w = batch_mask.shape[2:]
            # mask_trans = torch.tensor(create_circular_mask(h, w,radius = (h/2)-5)).to(device)

            feat_kp_trans_masked = feature_kp_anchor_trans * mask_trans
            _kp2_pos_masked = _kp2_pos * mask_trans
            _kp2_neg_masked = _kp2_neg * mask_trans
            if False:
                select = KeyPointsSelection(show=False)
                points1 = select(feat_kp_trans_masked[0][0].detach().cpu(), 5, 100)[:, :2]
                points2 = select(_kp2_pos_masked[0][0].detach().cpu(), 5, 100)[:, :2]
                points3 = select(_kp2_neg_masked[0][0].detach().cpu(), 5, 100)[:, :2]
                print(points1.shape)
                print(points2)
                print(points3)
            else:
                batch_detect = torch.cat(
                    [feat_kp_trans_masked[0][None], _kp2_pos_masked[0][None], _kp2_neg_masked[0][None]])

                nms = NMSHead(nms_size=10, nms_min_val=1, mask=batch_mask[0][0])
                coords = nms.forward(batch_detect.clone().detach()).cpu()
                # recuperar o x,y somente
                subdata = coords[:, :2]
                # mudar de ordem a dimensao 1 com 2
                subdata = subdata.transpose(1, 2)
                points1, points2, points3 = subdata[0], subdata[1], subdata[2]

            '''
            criar um batch de 3 elementos e plotar resultado 
            ancora , positivo e negativo
            '''
            imshow3([img_batch[0], batch_image_pos_trans[0], batch_image_neg_trans[0]],
                    [feat_kp_trans_masked[0][0], _kp2_pos_masked[0][0], _kp2_neg_masked[0][0]],
                    [points1, points2, points3])

            orie_img_batch = compute_gradient_direction(features_ori_anchor_trans)
            plot_orient_with_labels(orie_img_batch[0].cpu().detach(), coords)
            # batch_result = get_features(orie_img_batch[:1, :, :], subdata[:1, :, :], 12)
            batch_result = get_features(orie_img_batch[:1, :, :], subdata[:1, :, :], 12)
            print(features_ori_anchor_trans.shape, orie_img_batch.shape, subdata.shape)

    return batch_result

def predict_single_points(model,batch):
    model.eval()
    with torch.no_grad():

        img_batch, labels = batch
        img_batch = img_batch.to(device)
        _kp1, _orie1 = model(img_batch)
        print('pos predict ',_kp1.shape, _orie1.shape)
        _B, _C, _W, _H = img_batch.shape
        SIZE_BORDER = args.border_size
        batch_mask = torch.zeros(_B, 1, _W, _H).to(device)
        batch_mask[:, :, SIZE_BORDER:_W - SIZE_BORDER, SIZE_BORDER:_H - SIZE_BORDER] = 1


        nms = NMSHead(nms_size=args.nms_size, nms_min_val=1,mask=batch_mask[0][0])
        coords = nms.forward(_kp1.clone().detach()).cpu()
        # recuperar o x,y somente
        subdata = coords[:, :2]
        # mudar de ordem a dimensao 1 com 2
        subdata = subdata.transpose(1, 2)


        orie_img_batch = compute_gradient_direction(_orie1)
        #plot_orient_with_labels(orie_img_batch[0].cpu().detach(), coords)
        #plot_two_images_with_labels(img_batch[0].cpu().detach(),orie_img_batch[0].cpu().detach(),coords)
        batch_result = get_features(_kp1[:, :, :],orie_img_batch[:, :, :], subdata[:, :, :], args.box_size,args.show_feature)
    return batch_result, _kp1,orie_img_batch,coords