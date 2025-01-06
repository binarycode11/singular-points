import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--exported', required=False, default=False, type=bool)
parser.add_argument("--num_channels", required=False, default=3, type=int)
parser.add_argument("--pyramid_levels", required=False, default=3, type=int)# min 2
parser.add_argument("--scale_pyramid", required=False, default=1.3, type=int)# min 2
parser.add_argument("--dim_first", required=False, default=3, type=int)
parser.add_argument("--dim_second", required=False, default=3, type=int)
parser.add_argument("--dim_third", required=False, default=3, type=int)
parser.add_argument("--group_size", required=False, default=36, type=int)
parser.add_argument("--epochs", required=False, default=70
                    , type=int)#20
parser.add_argument("--border_size", required=False, default=22, type=int)
parser.add_argument("--box_size", required=False, default=21, type=int)
parser.add_argument("--nms_size", required=False, default=10, type=int)
parser.add_argument("--img_size", required=False, default=250, type=int)
parser.add_argument("--batch_size", required=False, default=5, type=int)
parser.add_argument('--path_data', required=False, default='./data', type=str)
parser.add_argument('--path_model', required=False, default='model.pt', type=str)
parser.add_argument('--is_loss_ssim', required=False, default=True, type=bool)
parser.add_argument('--margin_loss', required=False, default=2.0, type=float)
parser.add_argument('--outlier_rejection', required=False, default=False, type=bool)
parser.add_argument('--show_feature', required=False, default=False, type=bool)
args = parser.parse_args([])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def custom_config(args):
    args.num_channels = 3
    args.img_size = 120
    args.dim_first = 3
    args.dim_second = 5
    args.dim_third = 8
    args.batch_size = 16
    args.margin_loss = 2.0
    args.border_size  = 12
    args.nms_size = 5 # normal é 10, alterei so pra avaliar o match
    args.box_size = 21
    args.exported = False
    return args

def get_config_rekd(jupyter=False):
    parser = argparse.ArgumentParser(description='Train REKD Architecture')

    ## basic configuration
    parser.add_argument('--data_dir', type=str, default='../ImageNet2012/ILSVRC2012_img_val', #default='path-to-ImageNet',
                            help='The root path to the data from which the synthetic dataset will be created.')
    parser.add_argument('--synth_dir', type=str, default='', 
                            help='The path to save the generated sythetic image pairs.')
    parser.add_argument('--log_dir', type=str, default='trained_models/weights',
                            help='The path to save the REKD weights.')
    parser.add_argument('--load_dir', type=str, default='',
                        help='Set saved model parameters if resume training is desired.')                            
    parser.add_argument('--exp_name', type=str, default='REKD',
                            help='The Rotaton-equivaraiant Keypoint Detection (REKD) experiment name')
    ## network architecture
    parser.add_argument('--factor_scaling_pyramid', type=float, default=1.2,
                        help='The scale factor between the multi-scale pyramid levels in the architecture.')
    parser.add_argument('--group_size', type=int, default=36,  
                        help='The number of groups for the group convolution.')
    parser.add_argument('--dim_first', type=int, default=2,
                        help='The number of channels of the first layer')
    parser.add_argument('--dim_second', type=int, default=2,
                        help='The number of channels of the second layer')
    parser.add_argument('--dim_third', type=int, default=2,
                        help='The number of channels of the thrid layer')                       
    ## network training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=20, 
                        help='Number of epochs for training.')
    ## Loss function  
    parser.add_argument('--init_initial_learning_rate', type=float, default=1e-3,
                        help='The init initial learning rate value.')
    parser.add_argument('--MSIP_sizes', type=str, default="8,16,24,32,40",
                        help='MSIP sizes.')
    parser.add_argument('--MSIP_factor_loss', type=str, default="256.0,64.0,16.0,4.0,1.0",
                        help='MSIP loss balancing parameters.')
    parser.add_argument('--ori_loss_balance', type=float, default=100., 
                        help='')
    ## Dataset generation
    parser.add_argument('--patch_size', type=int, default=192,
                        help='The patch size of the generated dataset.')
    parser.add_argument('--max_angle', type=int, default=180,
                        help='The max angle value for generating a synthetic view to train REKD.')
    parser.add_argument('--min_scale', type=float, default=1.0,
                        help='The min scale value for generating a synthetic view to train REKD.')
    parser.add_argument('--max_scale', type=float, default=1.0,
                        help='The max scale value for generating a synthetic view to train REKD.')
    parser.add_argument('--max_shearing', type=float, default=0.0,
                        help='The max shearing value for generating a synthetic view to train REKD.')
    parser.add_argument('--num_training_data', type=int, default=9000,
                        help='The number of the generated dataset.')
    parser.add_argument('--is_debugging', type=bool, default=False,
                        help='Set variable to True if you desire to train network on a smaller dataset.')
    ## For eval/inference
    parser.add_argument('--num_points', type=int, default=1500,
                        help='the number of points at evaluation time.')
    parser.add_argument('--pyramid_levels', type=int, default=5,
                        help='downsampling pyramid levels.')
    parser.add_argument('--upsampled_levels', type=int, default=2,
                        help='upsampling image levels.')
    parser.add_argument('--nms_size', type=int, default=15,
                        help='The NMS size for computing the validation repeatability.')
    parser.add_argument('--border_size', type=int, default=15,
                        help='The number of pixels to remove from the borders to compute the repeatability.')
    ## For HPatches evaluation
    parser.add_argument('--hpatches_path', type=str, default='./datasets/hpatches-sequences-release',
                        help='dataset ')
    parser.add_argument('--eval_split', type=str, default='debug',
                        help='debug, view, illum, full, debug_view, debug_illum ...')      
    parser.add_argument('--descriptor', type=str, default="hardnet",
                        help='hardnet, sosnet, hynet')    

    args = parser.parse_args() if not jupyter else parser.parse_args(args=[])   

    if args.synth_dir == "":
        args.synth_dir = 'datasets/synth_data'

    args.MSIP_sizes = [int(i) for i in args.MSIP_sizes.split(",")]
    args.MSIP_factor_loss =[float(i) for i in args.MSIP_factor_loss.split(",")]

    return args


def get_config_singular(jupyter=False):
    """
    Cria e configura um parser de argumentos para o modelo Singular.
    
    Returns:
        args: Namespace contendo os argumentos configurados.
    """
    parser = argparse.ArgumentParser(description="Configurações do modelo Singular")

    # Configurações gerais
    general_group = parser.add_argument_group("Configurações Gerais")
    general_group.add_argument('--exported', default=False, type=bool, help="Indica se o modelo foi exportado.")
    general_group.add_argument('--path_data', default='./data', type=str, help="Caminho para os dados.")
    general_group.add_argument('--path_model', default='model.pt', type=str, help="Caminho para salvar o modelo.")

    # Hiperparâmetros do modelo
    model_group = parser.add_argument_group("Hiperparâmetros do Modelo")
    model_group.add_argument("--num_channels", default=3, type=int, help="Número de canais na entrada.")
    model_group.add_argument("--pyramid_levels", default=3, type=int, help="Níveis de pirâmide (mínimo 2).")
    model_group.add_argument("--scale_pyramid", default=1.3, type=float, help="Fator de escala da pirâmide (mínimo 2).")
    model_group.add_argument("--dim_first", default=3, type=int, help="Dimensão da primeira camada.")
    model_group.add_argument("--dim_second", default=5, type=int, help="Dimensão da segunda camada.")
    model_group.add_argument("--dim_third", default=8, type=int, help="Dimensão da terceira camada.")
    model_group.add_argument("--group_size", default=36, type=int, help="Tamanho do grupo para o modelo.")

    # Configurações de treinamento
    train_group = parser.add_argument_group("Configurações de Treinamento")
    train_group.add_argument("--epochs", default=70, type=int, help="Número de épocas para treinamento.")
    train_group.add_argument("--batch_size", default=16, type=int, help="Tamanho do batch.")
    train_group.add_argument('--is_loss_ssim', default=True, type=bool, help="Se verdadeiro, utiliza SSIM como loss.")
    train_group.add_argument('--margin_loss', default=2.0, type=float, help="Margem de perda para treinamento.")
    train_group.add_argument('--outlier_rejection', default=False, type=bool, help="Rejeição de outliers no treinamento.")

    # Configurações de entrada/saída
    io_group = parser.add_argument_group("Configurações de Entrada/Saída")
    io_group.add_argument("--img_size", default=120, type=int, help="Tamanho das imagens de entrada.")
    io_group.add_argument("--border_size", default=12, type=int, help="Tamanho da borda de exclusão.")
    io_group.add_argument("--box_size", default=21, type=int, help="Tamanho da caixa de detecção.")
    io_group.add_argument("--nms_size", default=5, type=int, help="Tamanho da área NMS (Non-Max Suppression).")
    io_group.add_argument('--show_feature', default=False, type=bool, help="Exibe as features detectadas.")

    # Parse dos argumentos
    args = parser.parse_args() if not jupyter else parser.parse_args(args=[])  

    # Seleção do dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args
