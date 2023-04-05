import gc
import torch
from tqdm import tqdm
from utils import imshow, save_model
from config import device
from tensor_augmetation import random_augmentation, shifted_batch_tensor


def train_one_epoch(model, loader, optimizer, criterion, epoch, is_show=True):
    model.train()
    running_loss = 0.
    t = tqdm(loader, desc="Train Epoch:{} ".format(epoch))
    for batch_idx, (batch_image, labels) in enumerate(t):

        # prever detector/descritor
        batch_image = batch_image.to(device)
        _kp1, _orie1 = model(batch_image)
        points = torch.randn(batch_image.shape[0], 2, 2)  # BxNx2 [x,y]
        batch_image_pos_trans, feature_kp_anchor_trans, features_ori_anchor_trans, coords_trans, mask_trans = random_augmentation(
            batch_image,
            _kp1, _orie1,
            points)
        _kp2_pos, _orie2_pos = model(batch_image_pos_trans)
        batch_image_neg_trans, _kp2_neg, _orie2_neg = shifted_batch_tensor(batch_image_pos_trans, _kp2_pos,
                                                                           _orie2_pos)  # faz o shift com o comando roll(x,1,0)

        if is_show and batch_idx % 25 == 0:
            temp = torch.cat([batch_image_pos_trans[3], batch_image_pos_trans[3] * mask_trans[3],
                              batch_image_neg_trans[3] * mask_trans[3]], dim=-1)
            imshow(temp)
            temp = torch.cat([feature_kp_anchor_trans[3] * mask_trans[3][0], _kp2_pos[3] * mask_trans[3][0],
                              _kp2_neg[3] * mask_trans[3][0]], dim=-1)
            imshow(temp)

        loss = criterion(feature_kp_anchor_trans*mask_trans, _kp2_pos*mask_trans, _kp2_neg*mask_trans)
        # loss = criterion(feature_kp_anchor_trans, _kp2_pos, _kp2_neg)
        # loss = criterion(feature_kp_trans, _kp2,mask_trans.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        try:
            running_loss += loss.item()
        except AttributeError:
            print('AttributeError ', type(loss), loss)

        t.set_description("Train Epoch:{} Loss: {:.5f}".format(epoch, running_loss))

        del batch_image
        del batch_image_neg_trans
        del _kp1, _kp2_pos, _kp2_neg, _orie1, _orie2_pos, _orie2_neg
        gc.collect()
        torch.cuda.empty_cache()

    model.eval()
    # print('Train Epoch: {} \tLoss: {:.15f}'.format(
    #     epoch, running_loss))
    return running_loss


def test(model, loader, criterion, epoch):
    model.eval()
    running_loss = 0.
    t = tqdm(loader, desc="Test Epoch:{} ".format(epoch))
    with torch.no_grad():
        for batch_idx, (batch_image, labels) in enumerate(t):
            # prever detector/descritor
            batch_image = batch_image.to(device)
            _kp1, _orie1 = model(batch_image)
            points = torch.randn(batch_image.shape[0], 2,
                                 2)  # BxNx2 [x,y] pontos sinteticos so pra completar parametros
            batch_image_pos_trans, feature_kp_anchor_trans, features_ori_anchor_trans, coords_trans, mask_trans = random_augmentation(
                batch_image,
                _kp1, _orie1,
                points)
            _kp2_pos, _orie2_pos = model(batch_image_pos_trans)
            batch_image_neg_trans, _kp2_neg, _orie2_neg = shifted_batch_tensor(batch_image_pos_trans, _kp2_pos,
                                                                               _orie2_pos)  # faz o shift com o comando roll(x,1,0)
            loss = criterion(feature_kp_anchor_trans * mask_trans, _kp2_pos * mask_trans, _kp2_neg * mask_trans)
            # loss = criterion(feature_kp_anchor_trans, _kp2_pos, _kp2_neg)
            try:
                running_loss += loss.item()
            except AttributeError:
                print('AttributeError ',type(loss),loss)

            t.set_description("Test Epoch:{} Loss: {:.5f}".format(epoch, running_loss))

        del batch_image
        del batch_image_neg_trans
        del _kp1, _kp2_pos, _kp2_neg, _orie1, _orie2_pos, _orie2_neg
        gc.collect()
        torch.cuda.empty_cache()
        # print('Test Epoch: {} \tLoss: {:.15f}'.format(
        #     epoch, running_loss))
        return running_loss
