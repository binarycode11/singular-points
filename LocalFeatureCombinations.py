import torch
from kornia.feature import LocalFeature
from DetectorMixins import KeyNetDetectorMixin, REKDetectorMixin, SingularPointDetectorMixin
from DescriptorMixins import SIFTDescriptorMixin, SosNetDescriptorMixin, HardNetDescriptorMixin

class KeyNetFeatureSIFT(LocalFeature, KeyNetDetectorMixin, SIFTDescriptorMixin):
    """Combina o detector KeyNet com o descritor SIFT."""
    def __init__(self, num_features: int = 200, upright: bool = False, device: torch.device = torch.device("cpu"), config: dict = None) -> None:
        super().__init__(detector=None, descriptor=None)
        self.upright = upright
        self.device = device
        self.config = config or { 'num_filters': 8, 'num_levels': 3, 'kernel_size': 5, 'Detector_conf': {'nms_size': 5, 'pyramid_levels': 2, 'up_levels': 2, 'scale_factor_levels': 1.3, 's_mult': 12.0 }}
        self.detector = self.initialize_detector(num_features)
        self.descriptor = self.initialize_descriptor()
        self.to(self.device)

class KeyNetFeatureSosNet(LocalFeature, KeyNetDetectorMixin, SosNetDescriptorMixin):
    """Combina o detector KeyNet com o descritor SOSNet."""
    def __init__(self, num_features: int = 200, upright: bool = False, device: torch.device = torch.device("cpu"), config: dict = None) -> None:
        super().__init__(detector=None, descriptor=None)
        self.upright = upright
        self.device = device
        self.config = config or { 'num_filters': 8, 'num_levels': 3, 'kernel_size': 5, 'Detector_conf': {'nms_size': 5, 'pyramid_levels': 2, 'up_levels': 2, 'scale_factor_levels': 1.3, 's_mult': 12.0 }}
        self.detector = self.initialize_detector(num_features)
        self.descriptor = self.initialize_descriptor()
        self.to(self.device)

class REKDSosNet(LocalFeature, REKDetectorMixin, SosNetDescriptorMixin):
    """Combina o detector REKD com o descritor SOSNet."""
    def __init__(self, num_features: int= 200, upright: bool = False, device: torch.device = torch.device("cpu"), config: dict = None) -> None:
        super().__init__(detector=None, descriptor=None)
        self.device = device
        self.config = config or { 'num_filters': 8, 'num_levels': 3, 'kernel_size': 5, 'Detector_conf': {'nms_size': 5, 'pyramid_levels': 0, 'up_levels': 0, 'scale_factor_levels': 1.3, 's_mult': 12.0 }}
        self.detector = self.initialize_detector(num_features)
        self.descriptor = self.initialize_descriptor()
        self.to(self.device)

class SingularPointSosNet(LocalFeature, SingularPointDetectorMixin, SosNetDescriptorMixin):
    """Combina o detector SingularPoint com o descritor SOSNet."""
    def __init__(self, num_features: int= 200, upright: bool = False, device: torch.device = torch.device("cpu"), config: dict = None) -> None:
        super().__init__(detector=None, descriptor=None)
        self.device = device
        self.config = config or { 'num_filters': 8, 'num_levels': 3, 'kernel_size': 5, 'Detector_conf': {'nms_size': 5, 'pyramid_levels': 0, 'up_levels': 0, 'scale_factor_levels': 1.3, 's_mult': 12.0 }}
        self.detector = self.initialize_detector(num_features)
        self.descriptor = self.initialize_descriptor()
        self.to(self.device)

class REKDHardNet(LocalFeature, REKDetectorMixin, HardNetDescriptorMixin):
    """Combina o detector REKD com o descritor HardNet."""
    def __init__(self, num_features: int= 200, upright: bool = False, device: torch.device = torch.device("cpu"), config: dict = None) -> None:
        super().__init__(detector=None, descriptor=None)
        self.device = device
        self.config = config or { 'num_filters': 8, 'num_levels': 3, 'kernel_size': 5, 'Detector_conf': {'nms_size': 5, 'pyramid_levels': 0, 'up_levels': 0, 'scale_factor_levels': 1.3, 's_mult': 12.0 }}
        self.detector = self.initialize_detector(num_features)
        self.descriptor = self.initialize_descriptor()
        self.to(self.device)

class SingularPointHardNet(LocalFeature, SingularPointDetectorMixin, HardNetDescriptorMixin):
    """Combina o detector SingularPoint com o descritor HardNet."""
    def __init__(self, num_features: int= 200, upright: bool = False, device: torch.device = torch.device("cpu"), config: dict = None) -> None:
        super().__init__(detector=None, descriptor=None)
        self.device = device
        self.config = config or { 'num_filters': 8, 'num_levels': 3, 'kernel_size': 5, 'Detector_conf': {'nms_size': 5, 'pyramid_levels': 0, 'up_levels': 0, 'scale_factor_levels': 1.3, 's_mult': 12.0 }}
        self.detector = self.initialize_detector(num_features)
        self.descriptor = self.initialize_descriptor()
        self.to(self.device)
