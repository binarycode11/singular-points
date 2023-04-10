import torch
import torch.optim as optim
from config import device
from training import KeyEqGroup, KeyPointsSelection, remove_borders, random_augmentation, shifted_batch_tensor
from utils import load_model, imshow, imshow2, imshow3




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

def predict_single_points(model,testloader):
    model.eval()
    with torch.no_grad():
        for img_batch, labels in testloader:
            img_batch = img_batch.to(device)
            _kp1, _orie1 = model(img_batch)
            points = torch.randn(img_batch.shape[0], 2,
                                 2).to(device)  # BxNx2 [x,y] pontos sinteticos so pra completar parametros
            batch_image_pos_trans, feature_kp_anchor_trans, features_ori_anchor_trans, coords_trans, mask_trans = random_augmentation(
                img_batch,
                _kp1, _orie1,
                points)
            _kp2_pos, _orie2_pos = model(batch_image_pos_trans)
            batch_image_neg_trans, _kp2_neg, _orie2_neg = shifted_batch_tensor(batch_image_pos_trans, _kp2_pos,
                                                                               _orie2_pos)
            feat_kp_trans_masked = feature_kp_anchor_trans * mask_trans
            _kp2_pos_masked = _kp2_pos * mask_trans
            _kp2_neg_masked = _kp2_neg * mask_trans
            print("masked ",feat_kp_trans_masked.shape,mask_trans.shape)
            select = KeyPointsSelection(show=False)
            points1 = select(_kp1[0][0].detach().cpu(), 5, 100)
            points2 = select(_kp2_pos[0][0].detach().cpu(), 5, 100)
            points3 = select(_kp2_neg[0][0].detach().cpu(), 5, 100)

            imshow3([img_batch[0],batch_image_pos_trans[0],batch_image_neg_trans[0]],[feat_kp_trans_masked[0][0], _kp2_pos_masked[0][0],_kp2_neg_masked[0][0]], [points1, points2,points3])
            print("teste")