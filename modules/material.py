import torch.nn as nn

from .texture import VirtualTextureModule, TextureModule

class MaterialModule(nn.Module):

    def __init__(self, 
                 BaseColor: VirtualTextureModule | TextureModule, 
                 AttenuationRoughnessMetallic: VirtualTextureModule | TextureModule, 
                 Normal: VirtualTextureModule | TextureModule):
        super().__init__()
        self.BaseColor = BaseColor
        self.AttenuationRoughnessMetallic = AttenuationRoughnessMetallic
        self.Normal = Normal
        
    def offload(self):
        self.BaseColor.offload()
        self.AttenuationRoughnessMetallic.offload()
        self.Normal.offload()

    def share_memory_(self):
        self.BaseColor.share_memory_()
        self.AttenuationRoughnessMetallic.share_memory_()
        self.Normal.share_memory_()