import torch
from typing import Optional, Tuple
from torch import nn, Tensor
import kornia
from kornia.feature import (
    LocalFeature, LAFDescriptor, MultiResolutionDetector, SOSNet,HardNet,
    CornerGFTT, PassLAF, LAFOrienter, LAFAffNetShapeEstimator
)
from kornia.feature.scale_space_detector import get_default_detector_config
from kornia.feature.keynet import KeyNetDetector, keynet_default_config
from external.REKD import REKD
from teste_util import CustomNetDetector, load_model
from config import get_config_rekd, get_config_singular
from best.singular_point import SingularPoints



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


class BaseDetector(nn.Module):
    """Classe base para inicialização de detectores."""
    def initialize_detector(self, num_features: int, size_laf: int = 19) -> nn.Module:
        """Método que deve ser implementado para criar o detector."""
        raise NotImplementedError


class BaseDescriptor(nn.Module):
    """Classe base para inicialização de descritores."""
    def initialize_descriptor(self) -> LAFDescriptor:
        """Método que deve ser implementado para criar o descritor."""
        raise NotImplementedError

class KeyNetDetectorMixin(BaseDetector):
    """Mixin para o detector KeyNet."""

    def initialize_detector(self, num_features: int, size_laf: int = 19) -> nn.Module:
        return KeyNetDetector(
            pretrained=True,
            num_features=num_features,
            ori_module=PassLAF() if self.upright else LAFOrienter(size_laf),
            aff_module=LAFAffNetShapeEstimator(preserve_orientation=self.upright).eval(),
            keynet_conf=self.config,
        ).to(self.device).eval()



class REKDetectorMixin(BaseDetector):
    """Mixin para o detector REKD."""

    def initialize_detector(self, num_features: int, size_laf: int = 19) -> nn.Module:
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
            REKDetector(args, self.device),
            num_features=num_features,
            config=self.config["Detector_conf"],
            ori_module=LAFOrienter(size_laf),
        ).to(self.device)


class SingularPointDetectorMixin(BaseDetector):
    """Mixin para o detector SingularPoint."""

    def initialize_detector(self, num_features: int, size_laf: int = 19) -> nn.Module:
        class SingularPointDetector(nn.Module):
            def __init__(self, args) -> None:
                super().__init__()
                self.model = SingularPoints(args).to(args.device)
                load_model(self.model, args.load_dir, args.device)

            def forward(self, x):
                return self.model(x)[1]

        args = get_config_singular(jupyter=True)
        args.num_channels = 1
        args.load_dir = './data/models/sp_map_fo_30.pth'
        args.device = self.device
        return MultiResolutionDetector(
            SingularPointDetector(args),
            num_features=num_features,
            config=self.config["Detector_conf"],
            ori_module=LAFOrienter(size_laf),
        )


class SIFTDescriptorMixin(BaseDescriptor):
    """Mixin para o descritor SIFT."""
    def initialize_descriptor(self) -> LAFDescriptor:
        patch_size = 13  # Tamanho do patch do descritor SIFT
        sift_descriptor = kornia.feature.SIFTDescriptor(patch_size=patch_size, rootsift=True).to(self.device)
        return LAFDescriptor(
            sift_descriptor,
            patch_size=patch_size,
            grayscale_descriptor=True,
        ).to(self.device)
    

class SosNetDescriptorMixin(BaseDescriptor):
    """Mixin para o descritor SOSNet."""

    def initialize_descriptor(self) -> LAFDescriptor:
        return LAFDescriptor(
            SOSNet(pretrained=True).to(self.device).eval(),
            patch_size=32,
            grayscale_descriptor=True,
        ).to(self.device)
    
class HardNetDescriptorMixin(BaseDescriptor):
    """Mixin para o descritor HardNet."""
    
    def initialize_descriptor(self) -> LAFDescriptor:
        
        return LAFDescriptor(
            HardNet(pretrained=True).to(self.device).eval(),  # Inicializa o descritor HardNet
            patch_size=32,  # Tamanho do patch, pode ser ajustado conforme a necessidade
            grayscale_descriptor=True,
        ).to(self.device)

class KeyNetFeatureSIFT(LocalFeature, KeyNetDetectorMixin, SIFTDescriptorMixin):
    """Combina o detector KeyNet com o descritor SOSNet."""

    def __init__(
        self,
        num_features: int = 200,
        upright: bool = False,
        device: torch.device = torch.device("cpu"),
        config: dict = None,
    ) -> None:
        self.upright = upright
        self.device = device
        self.config = config or {
            'num_filters': 8,
            'num_levels': 3,
            'kernel_size': 5,
            'Detector_conf': {
                'nms_size': 5, 'pyramid_levels': 2, 'up_levels': 2,
                'scale_factor_levels': 1.3, 's_mult': 12.0
            },
        }
        super().__init__(detector=None, descriptor=None)
        self.detector = self.initialize_detector(num_features)
        self.descriptor = self.initialize_descriptor()
        self.to(self.device)        

class KeyNetFeatureSosNet(LocalFeature, KeyNetDetectorMixin, SosNetDescriptorMixin):
    """Combina o detector KeyNet com o descritor SOSNet."""

    def __init__(
        self,
        num_features: int = 200,
        upright: bool = False,
        device: torch.device = torch.device("cpu"),
        config: dict = None,
    ) -> None:
        self.upright = upright
        self.device = device
        self.config = config or {
            'num_filters': 8,
            'num_levels': 3,
            'kernel_size': 5,
            'Detector_conf': {
                'nms_size': 5, 'pyramid_levels': 2, 'up_levels': 2,
                'scale_factor_levels': 1.3, 's_mult': 12.0
            },
        }
        super().__init__(detector=None, descriptor=None)
        self.detector = self.initialize_detector(num_features)
        self.descriptor = self.initialize_descriptor()
        self.to(self.device)

class REKDSosNet(LocalFeature, REKDetectorMixin, SosNetDescriptorMixin):
    """Combina o detector REKD com o descritor SOSNet."""

    def __init__(self, num_features: int, upright: bool = False, device: torch.device = torch.device("cpu"), config: dict = None) -> None:
        super().__init__(detector=None, descriptor=None)
        self.device = device
        self.config = config or {
            'num_filters': 8,
            'num_levels': 3,
            'kernel_size': 5,
            'Detector_conf': {
                'nms_size': 5,
                'pyramid_levels': 0,
                'up_levels': 0,
                'scale_factor_levels': 1.3,
                's_mult': 12.0,
            },
        }
        self.detector = self.initialize_detector(num_features)
        self.descriptor = self.initialize_descriptor()
        self.to(self.device)


class SingularPointSosNet(LocalFeature, SingularPointDetectorMixin, SosNetDescriptorMixin):
    """Combina o detector SingularPoint com o descritor SOSNet."""

    def __init__(self, num_features: int, upright: bool = False, device: torch.device = torch.device("cpu"), config: dict = None) -> None:
        super().__init__(detector=None, descriptor=None)
        self.device = device
        self.config = config or {
            'num_filters': 8,
            'num_levels': 3,
            'kernel_size': 5,
            'Detector_conf': {
                'nms_size': 5,
                'pyramid_levels': 0,
                'up_levels': 0,
                'scale_factor_levels': 1.3,
                's_mult': 12.0,
            },
        }
        self.detector = self.initialize_detector(num_features)
        self.descriptor = self.initialize_descriptor()
        self.to(self.device)

class REKDHardNet(LocalFeature, REKDetectorMixin, HardNetDescriptorMixin):
    """Combina o detector REKD com o descritor SOSNet."""

    def __init__(self, num_features: int, upright: bool = False, device: torch.device = torch.device("cpu"), config: dict = None) -> None:
        super().__init__(detector=None, descriptor=None)
        self.device = device
        self.config = config or {
            'num_filters': 8,
            'num_levels': 3,
            'kernel_size': 5,
            'Detector_conf': {
                'nms_size': 5,
                'pyramid_levels': 0,
                'up_levels': 0,
                'scale_factor_levels': 1.3,
                's_mult': 12.0,
            },
        }
        self.detector = self.initialize_detector(num_features)
        self.descriptor = self.initialize_descriptor()
        self.to(self.device)


class SingularPointHardNet(LocalFeature, SingularPointDetectorMixin, HardNetDescriptorMixin):
    """Combina o detector SingularPoint com o descritor SOSNet."""

    def __init__(self, num_features: int, upright: bool = False, device: torch.device = torch.device("cpu"), config: dict = None) -> None:
        super().__init__(detector=None, descriptor=None)
        self.device = device
        self.config = config or {
            'num_filters': 8,
            'num_levels': 3,
            'kernel_size': 5,
            'Detector_conf': {
                'nms_size': 5,
                'pyramid_levels': 0,
                'up_levels': 0,
                'scale_factor_levels': 1.3,
                's_mult': 12.0,
            },
        }
        self.detector = self.initialize_detector(num_features)
        self.descriptor = self.initialize_descriptor()
        self.to(self.device)