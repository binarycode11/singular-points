import torch
import kornia
from kornia.feature import (
    LocalFeature, LAFDescriptor, MultiResolutionDetector, SOSNet,
    CornerGFTT, PassLAF, LAFOrienter, LAFAffNetShapeEstimator
)
from kornia.feature.scale_space_detector import get_default_detector_config
from kornia.feature.keynet import KeyNetDetector, keynet_default_config
from external.REKD import REKD
from teste_util import CustomNetDetector, load_model
from config import get_config_rekd, get_config_singular
from best.singular_point import SingularPoints
from typing import Optional, Tuple
from torch import nn, Tensor


class GFTTFeatureSosNet(LocalFeature):
    """Convenience module, which implements GFTT detector + SOSNet descriptor."""

    def __init__(
        self,
        num_features: int = 200,
        upright: bool = False,
        device: torch.device = torch.device("cpu"),
        config: dict = None,
    ) -> None:
        if config is None:
            config = get_default_detector_config()
        detector = MultiResolutionDetector(
            CornerGFTT(),
            num_features,
            config,
            ori_module=PassLAF() if upright else LAFOrienter(19),
            aff_module=LAFAffNetShapeEstimator(preserve_orientation=upright).eval(),
        ).to(device)

        sosnet32 = SOSNet(pretrained=True).to(device).eval()
        descriptor = LAFDescriptor(sosnet32, patch_size=32, grayscale_descriptor=True).to(device)
        super().__init__(detector, descriptor)


class KeyNetFeatureSosNet(LocalFeature):
    """Combines KeyNet detector + SOSNet descriptor."""

    def __init__(
        self,
        num_features: int = 200,
        upright: bool = False,
        device: torch.device = torch.device("cpu"),
        config: dict = None,
    ) -> None:
        if config is None:
            config = {
                'num_filters': 8,
                'num_levels': 3,
                'kernel_size': 5,
                'Detector_conf': {
                    'nms_size': 5, 'pyramid_levels': 2, 'up_levels': 2,
                    'scale_factor_levels': 1.3, 's_mult': 12.0
                },
            }
        detector = KeyNetDetector(
            pretrained=True,
            num_features=num_features,
            ori_module=PassLAF() if upright else LAFOrienter(32),
            aff_module=LAFAffNetShapeEstimator(preserve_orientation=upright).eval(),
            keynet_conf=config,
        ).to(device).eval()

        sosnet32 = SOSNet(pretrained=True).to(device).eval()
        descriptor = LAFDescriptor(sosnet32, patch_size=32, grayscale_descriptor=True).to(device)
        super().__init__(detector, descriptor)


class KeyNetFeatureSIFT(LocalFeature):
    """Combines KeyNet detector + SIFT descriptor."""

    def __init__(
        self,
        num_features: int = 200,
        upright: bool = False,
        device: torch.device = torch.device("cpu"),
        config: dict = None,
    ) -> None:
        if config is None:
            config = {
                'num_filters': 8,
                'num_levels': 3,
                'kernel_size': 5,
                'Detector_conf': {
                    'nms_size': 5, 'pyramid_levels': 2, 'up_levels': 2,
                    'scale_factor_levels': 1.3, 's_mult': 12.0
                },
            }
        detector = KeyNetDetector(
            pretrained=True,
            num_features=num_features,
            ori_module=PassLAF() if upright else LAFOrienter(32),
            aff_module=LAFAffNetShapeEstimator(preserve_orientation=upright).eval(),
            keynet_conf=config,
        ).to(device).eval()

        patch_size = 13
        sift_descriptor = kornia.feature.SIFTDescriptor(patch_size=patch_size, rootsift=True).to(device)
        descriptor = LAFDescriptor(sift_descriptor, patch_size=patch_size, grayscale_descriptor=True).to(device)
        super().__init__(detector, descriptor)


class REKDSosNet(LocalFeature):
    """Combines REKD detector + SOSNet descriptor."""

    def __init__(
        self,
        num_features: int = 200,
        upright: bool = False,
        device: torch.device = torch.device("cpu"),
        config: dict = None,
    ) -> None:
        super().__init__(detector=None, descriptor=None)
        self.device = device
        self.config = config or {
            'num_filters': 8,
            'num_levels': 3,
            'kernel_size': 5,
            'Detector_conf': {
                'nms_size': 5, 'pyramid_levels': 0, 'up_levels': 0,
                'scale_factor_levels': 1.3, 's_mult': 12.0
            },
        }
        self.detector = self._initialize_detector(num_features)
        self.descriptor = self._initialize_descriptor()

    def _initialize_detector(self, num_features,size_laf=19):
        class REKDetector(nn.Module):
            def __init__(self, args, device: torch.device) -> None:
                super().__init__()
                self.model = REKD(args, device).to(device).eval()
                self.model.load_state_dict(torch.load(args.load_dir, weights_only=False))

            def forward(self, x: Tensor) -> Tensor:
                return self.model(x)[0]
        
        args = get_config_rekd(jupyter=True)
        args.load_dir = 'trained_models/release_group36_f2_s2_t2.log/best_model.pt'
        return MultiResolutionDetector(
            REKDetector(args, self.device), num_features=num_features,
            config=self.config["Detector_conf"], ori_module=LAFOrienter(size_laf)
        ).to(self.device)

    def _initialize_descriptor(self) -> LAFDescriptor:
        return LAFDescriptor(SOSNet(pretrained=True).to(self.device).eval(), patch_size=32, grayscale_descriptor=True).to(self.device)


class SingularPointSosNet(LocalFeature):
    def __init__(        self,
        num_features: int = 200,
        upright: bool = False,
        device: torch.device = torch.device("cpu"),
        config: dict = None
        ) -> None:
        super().__init__(detector=None, descriptor=None)
        self.device = device
        self.config = config or {
            'num_filters': 8,
            'num_levels': 3,
            'kernel_size': 5,
            'Detector_conf': {'nms_size': 5, 'pyramid_levels': 0, 'up_levels': 0, 'scale_factor_levels': 1.3, 's_mult': 12.0},
        }

        self.detector = self._initialize_detector(num_features)
        self.descriptor = self._initialize_descriptor()
        self.to(self.device)

    def _initialize_detector(self,num_features,size_laf=19):
        class SingularPointDetector(nn.Module):
            
            def __init__(self,args) -> None:
                super().__init__()
                self.model = SingularPoints(args).to( args.device)
                load_model(self.model, args.load_dir, args.device)
                # self.model.load_state_dict(torch.load(args.load_dir, weights_only=False))

            def forward(self, x):
                return self.model(x)[1]

        args = get_config_singular(jupyter=True)
        args.num_channels = 1
        args.load_dir = './data/models/sp_map_fo_30.pth'
        args.device = self.device
        return MultiResolutionDetector(SingularPointDetector(args), num_features = num_features,config = self.config["Detector_conf"], 
                                       ori_module=LAFOrienter(size_laf))

    def _initialize_descriptor(self) -> LAFDescriptor:
        """Cria o descritor SOSNet"""
        return LAFDescriptor(SOSNet(pretrained=True), patch_size=32, grayscale_descriptor=True).to(self.device) 