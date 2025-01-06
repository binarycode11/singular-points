from torch import nn
from kornia.feature import (
    LAFDescriptor, SOSNet,HardNet, SIFTDescriptor
)

class BaseDescriptor(nn.Module):
    """Classe base para inicialização de descritores."""
    def initialize_descriptor(self) -> LAFDescriptor:
        """Método que deve ser implementado para criar o descritor."""
        raise NotImplementedError

class SIFTDescriptorMixin(BaseDescriptor):
    """Mixin para o descritor SIFT."""
    def initialize_descriptor(self) -> LAFDescriptor:
        patch_size = 13  # Tamanho do patch do descritor SIFT
        sift_descriptor = SIFTDescriptor(patch_size=patch_size, rootsift=True).to(self.device)
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
