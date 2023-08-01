"""
Este arquivo contém a implementação de uma rede siamesa para detecção e correspondência de características em imagens usando PyTorch, Kornia e e2cnn.
A classe 'Feature' define uma arquitetura de rede convolucional com camadas equivariantes para a extração de características.
A classe 'Discriminator' implementa uma rede densa para calcular a distância euclidiana entre as características extraídas.
A classe 'Siamesa' combina a extração de características e o cálculo de correspondências usando o método 'bidirectional_match',
permitindo a obtenção de correspondências confiáveis entre duas imagens. O arquivo também contém um bloco de código para testar a rede em um exemplo de entrada.
"""


from e2cnn import gspaces
from e2cnn import nn as enn    #the equivariant layer we need to build the model
from torch import nn
import torch
import numpy as np
import kornia

class Feature(nn.Module):
    def __init__(self,n_channel=2) -> None:
        super().__init__()
        r2_act = gspaces.Rot2dOnR2(N=18)

        feat_type_in  = enn.FieldType(r2_act,  n_channel*[r2_act.trivial_repr])
        feat_type_out = enn.FieldType(r2_act, 2*n_channel*[r2_act.regular_repr])
        self.input_type = feat_type_in

        self.block1 = enn.SequentialModule(
                enn.R2Conv(feat_type_in, feat_type_out, kernel_size=3, padding=0, bias=False),
                enn.InnerBatchNorm(feat_type_out),
                enn.ReLU(feat_type_out, inplace=True)
                )

        self.pool1 = enn.PointwiseAvgPoolAntialiased(feat_type_out, sigma=0.66, stride=1, padding=0)

        feat_type_in  = self.block1.out_type
        feat_type_out = enn.FieldType(r2_act,  4*n_channel*[r2_act.regular_repr])
        self.block2 = enn.SequentialModule(
                enn.R2Conv(feat_type_in, feat_type_out, kernel_size=3, padding=0, bias=False),
                enn.InnerBatchNorm(feat_type_out),
                enn.ReLU(feat_type_out, inplace=True),
                )
        # self.pool2 = enn.PointwiseAvgPool(feat_type_out, 21)

        feat_type_in  = feat_type_out
        feat_type_out = enn.FieldType(r2_act,  8*n_channel*[r2_act.regular_repr])
        self.block3 = enn.SequentialModule(
                enn.R2Conv(feat_type_in, feat_type_out, kernel_size=3, padding=0, bias=False),
                enn.InnerBatchNorm(feat_type_out),
                enn.ReLU(feat_type_out, inplace=True),
                enn.GroupPooling(feat_type_out),
                )
        self.pool = enn.PointwiseAdaptiveAvgPool(self.block3.out_type,1)

    def forward(self,X1)->torch.Tensor:
        x = enn.GeometricTensor(X1, self.input_type)
        n_dim = X1.shape[-1]
        mask = enn.MaskModule(self.input_type, n_dim, margin=2).to(X1.device)
        x = mask(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.tensor.reshape(x.shape[0],-1)
        return x

class Discriminator(nn.Module):
    def __init__(self, n_classes=10) -> None:
        super().__init__()
        #criar camadas densa a partir de x que é uma cnn
        self.dense1 = nn.Linear(2*1*128, 512)
        self.dense2 = nn.Linear(512, 128)
        self.droupout = nn.Dropout(0.2)
        self.activation = nn.LeakyReLU()
        #self.activation = nn.ELU()
        #função de ativação ideal para retornar um valor entre 0 e 1
        self.activation2 = nn.Tanh()


    def forward(self,X1,X2)->torch.Tensor:
        flatten_x1 = X1.view(X1.size(0), -1)
        flatten_x2 = X2.view(X2.size(0), -1)
        x = torch.cat((flatten_x1,flatten_x2),dim=1)
        x = self.droupout(self.dense1(x))
        x = self.activation(x)
        x = self.droupout(self.dense2(x))
        x = self.activation(x)

        # Calculando a distância euclidiana
        x = torch.norm(x, dim=1)
        # x = self.activation2(x) #retorna um valor entre 0 e 1
        return x

class Siamesa(nn.Module):
    def __init__(self,n_channel=2) -> None:
            super().__init__()
            self.feature = Feature(n_channel=n_channel)
            self.discriminator = Discriminator()

    def forward(self,X1,X2)->torch.Tensor:
        # matches = self.match_bidirecional(X1,X2)
        #---- via kornia
        desc1 = self.feature(X1)
        desc2 = self.feature(X2)
        # print(desc1.shape,desc2.shape)
        matches = self.bidirectional_match(desc1,desc2,threshold=0.75)# via kornia
        return matches

    def matching_siamese_patch(self,data1,data2):
        size = data2.shape[0]
        matches = []
        for i in range(len(data1)):
          temp = data1[i][None]
          temp_n = temp.repeat(size,1)
          distances = self.discriminator(temp_n, data2)
          match_index = distances.argmin().item()
          matches.append((i, match_index))
        matches = np.array(matches)
        return matches

    def match_bidirecional(self,data1,data2):
        f_dt1 = self.feature(data1)
        f_dt2 = self.feature(data2)
        matches1 = self.matching_siamese_patch(f_dt1,f_dt2)
        matches2 = self.matching_siamese_patch(f_dt2,f_dt1)
        matches = []
        for i in range(len(matches1)):
           if matches2[matches1[i][1]][1] == i:
                matches.append(matches1[i])
        return torch.tensor(matches)

    def bidirectional_match(self,feat1, feat2, threshold=1.0):
        feat1 = feat1.float()
        feat2 = feat2.float()

        s1, matches1 = kornia.feature.match_snn(feat1, feat2, threshold)
        s2, matches2 = kornia.feature.match_snn(feat2, feat1, threshold)

        bidirectional_matches = []
        for i, match in enumerate(matches1):
            indices = torch.where(matches2[:, 0] == match[1].item())[0]
            if indices.numel() > 0:
                for index in indices:
                    if matches2[index][1].item() == match[0].item():
                        bidirectional_matches.append((match[0].item(), match[1].item()))
        return torch.tensor(bidirectional_matches)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    n_channel =8
    PS =21
    model =Siamesa(n_channel=n_channel).to(device)
    model.eval()
    X1=torch.rand(50,n_channel,PS,PS).to(device)

    with torch.no_grad():
        dist = model(X1,X1)
        print(dist.shape,dist)
        dist = model(X1,X1)
        print(dist.shape,dist)

