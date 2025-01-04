import torch
from torch import nn, Tensor
from kornia.feature import (
    MultiResolutionDetector, PassLAF, LAFOrienter, LAFAffNetShapeEstimator
)
from kornia.feature.keynet import KeyNetDetector
from external.REKD import REKD
from teste_util import load_model
from config import get_config_rekd, get_config_singular
from best.singular_point import SingularPoints

class BaseDetector(nn.Module):
    """Classe base para inicialização de detectores."""
    def initialize_detector(self, num_features: int, size_laf: int = 19) -> nn.Module:
        """Método que deve ser implementado para criar o detector."""
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
