"""
Este arquivo contém a definição da classe 'MeuDataset', uma subclasse do PyTorch Dataset, que representa um conjunto de dados personalizado.
A classe permite armazenar dados de amostras e seus respectivos rótulos, além de fornecer métodos para acessar, salvar e carregar os dados.
O método 'save_to_file' salva o dataset em um arquivo no formato binário do PyTorch (.pt) contendo tensores para as amostras e rótulos.
O método 'load_from_file' permite recuperar o dataset a partir de um arquivo previamente salvo, permitindo reutilizar os dados em futuras execuções do código.
Os dados são armazenados em tensores do PyTorch para eficiência e facilidade de manipulação em tarefas de aprendizado de máquina e processamento de dados.
"""
import torch
from torch.utils.data import Dataset, DataLoader

class MeuDataset(Dataset):
    def __init__(self, summary_pool_list, labels_list):
        self.summary = summary_pool_list
        self.labels = labels_list

    def __len__(self):
        return len(self.summary)

    def __getitem__(self, idx):
        summary = self.summary[idx]
        labels = self.labels[idx]
        # Implemente aqui a lógica para retornar uma amostra do seu conjunto de dados
        return summary, labels

    def save_to_file(self, file_name):
        dataset_dict = {
            'summary': [summary.tolist() for summary in self.summary],
            'labels': [labels.tolist() for labels in self.labels],
        }
        torch.save(dataset_dict, file_name)

    @classmethod
    def load_from_file(cls, file_name):
        dataset_dict = torch.load(file_name)
        summary_pool_list = [torch.tensor(summary) for summary in dataset_dict['summary']]
        labels_list = [torch.tensor(labels) for labels in dataset_dict['labels']]
        return cls(summary_pool_list, labels_list)



import kornia
from tqdm import tqdm

def avaliar_descritor(dataloader, descriptor,th=0.85,only_diferent=False,device='cuda:0'):
    """
    Avalia um descritor de características utilizando um DataLoader contendo pares de imagens.

    Parâmetros:
        dataloader (DataLoader): DataLoader contendo os pares de imagens para avaliar o descritor.
        descriptor (callable): Função descritora que recebe uma imagem e retorna seus descritores.
        th (float, opcional): Valor de threshold para considerar um par como positivo. Padrão é 0.85.
        only_diferent (bool, opcional): Se True, considera apenas pares de imagens diferentes.
                                       Se False, considera todos os pares. Padrão é False.

    Retorna:
        total_acertos (int): O número total de pares corretamente correspondidos (positivos).
        total_erros (int): O número total de pares incorretamente correspondidos (falsos positivos e falsos negativos).
        total_elementos (int): O número total de elementos (pares de imagens) no DataLoader.
    """
    progress_bar = tqdm(dataloader)
    total_acertos = 0
    total_erros = 0
    total_elementos = len(dataloader.dataset)  # Total de elementos no DataLoader
    for idx, data in enumerate(progress_bar):
        # extrair as features e orientações
        batch_in, batch_out = data[0].to(device), data[1].to(device)
        descs_original = descriptor(batch_in)
        descs_transform = descriptor(batch_out)
        match_pos, match_neg = calcular_matching(descs_original, descs_transform,th,only_diferent)
        acertos_pos = torch.sum(match_pos[:, 0] == match_pos[:, 1])
        false_pos = torch.sum(match_pos[:, 0] != match_pos[:, 1])
        false_neg = match_neg.shape[0]
        total_acertos += acertos_pos
        total_erros += false_pos + false_neg

    if only_diferent is False:
        total_elementos = total_elementos*4
    else:
        total_elementos = total_elementos*2

    return total_acertos, total_erros, total_elementos

def calcular_matching(descs_original, descs_transform,th =0.85,only_diferent=False):
    """
    Calcula os correspondentes (matches) entre os descritores de duas imagens.

    Parâmetros:
        descs_original (Tensor): Descritores da imagem original.
        descs_transform (Tensor): Descritores da imagem transformada.
        th (float, opcional): Valor de threshold para considerar um match como positivo. Padrão é 0.85.
        only_diferent (bool, opcional): Se True, considera apenas matches entre imagens diferentes.
                                       Se False, considera todos os matches. Padrão é False.

    Retorna:
        match_pos (Tensor): Tensor contendo os correspondentes positivos (matches verdadeiros).
        match_neg (Tensor): Tensor contendo os correspondentes negativos (matches falsos).
    """

    half_size = descs_original.size(0) // 2
    in1 = descs_original[:half_size]
    in2 = descs_original[half_size:]

    out1 = descs_transform[:half_size]
    out2 = descs_transform[half_size:]

    #match no cenário positivo 2 vezes
    _,op1 = kornia.feature.match_snn(in1, out1,th)
    _,op2 = kornia.feature.match_snn(in1, in1,th)
    _,op3 = kornia.feature.match_snn(in2, out2,th)
    _,op4 = kornia.feature.match_snn(in2, in2,th)


    #match no cenário negativo 2 vezes
    _,on1 = kornia.feature.match_snn(in1, out2,th)
    _,on2 = kornia.feature.match_snn(in1, in2,th)
    _,on3 = kornia.feature.match_snn(in2, out1,th)
    _,on4 = kornia.feature.match_snn(in2, in1,th)

    if(only_diferent is False):
        match_pos = torch.cat([op1, op2, op3, op4], dim=0)
        match_neg = torch.cat([on1, on2, on3, on4], dim=0)
    else:
        match_pos = torch.cat([op1, op3], dim=0)
        match_neg = torch.cat([on1, on3], dim=0)

    return match_pos,match_neg