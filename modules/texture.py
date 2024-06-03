from collections import deque
import logging
import math
import os
import time
import torch
import torch.nn as nn
from configs import Configuration
import nvdiffrast.torch as dr
import renderutils as ru

from .deffered_cache import CPUOffloadCache

from timer import timers

def is_power_of_two(x):
    return x > 0 and (x & (x-1)) == 0

class TextureFilterMode:
    Nearest = "nearest"
    Linear = "linear"
    LinearMipNearest = "linear-mipmap-nearest"
    LinearMipLinear = "linear-mipmap-linear"

class TextureBoundaryMode:
    Cube = "cube"
    Wrap = "wrap"
    Clamp = "clamp"
    Zero = "zero"

import util
class texture2d_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, texture):
        return util.avg_pool_nhwc(texture, (2,2))

    @staticmethod
    def backward(ctx, dout):
        gy, gx = torch.meshgrid(torch.linspace(0.0 + 0.25 / dout.shape[1], 1.0 - 0.25 / dout.shape[1], dout.shape[1]*2, device="cuda"), 
                                torch.linspace(0.0 + 0.25 / dout.shape[2], 1.0 - 0.25 / dout.shape[2], dout.shape[2]*2, device="cuda"),
                                indexing='ij')
        uv = torch.stack((gx, gy), dim=-1)
        return dr.texture(dout * 0.25, uv[None, ...].contiguous(), filter_mode='linear', boundary_mode='clamp')

class ITextureModule(nn.Module):
    def __init__(self):
        super().__init__()

    def clone(self) -> "ITextureModule":
        raise NotImplementedError
    
    def pin_memory_(self):
        raise NotImplementedError
    
    def share_memory_(self):
        raise NotImplementedError
    
    def requires_grad_(self, requires_grad: bool):
        raise NotImplementedError
    
    def cache_clear(self):
        raise NotImplementedError
    
    def get_trainable_params(self) -> list[torch.Tensor]:
        raise NotImplementedError
    
    def set_max_mip_level(self, max_mip_level):
        raise NotImplementedError
    
    def offload(self):
        raise NotImplementedError
    
    def feedback(self, texc, texc_deriv, filter_mode="linear-mipmap-linear", boundary_mode="wrap", mask=None) -> list[list[torch.Tensor]]:
        raise NotImplementedError
    
    def sample(self, texc, texc_deriv, filter_mode="linear-mipmap-linear", boundary_mode="wrap", mask=None) -> torch.Tensor:
        raise NotImplementedError
    
    def texture_mipmap_fwd(self, max_mip_level=None):
        raise NotImplementedError
    
    def texture_mipmap_bwd(self) -> list[torch.Tensor]:
        raise NotImplementedError

class TextureModule(ITextureModule):

    def __init__(self, data: torch.Tensor,
                 min_max=None, 
                 max_mip_level=None,
                 name="",
                 page_size_x=256,
                 page_size_y=256,
                 fp16_texture=False,
                 lru_cache_max_size=1000) -> None:
        # data must have shape [batch, height, width, channels]
        super().__init__()
        self.height = data.shape[1]
        self.width = data.shape[2]
        self.min_max = min_max
        self.channels = data.shape[3]
        self.set_max_mip_level(max_mip_level)
        self.texture = nn.Parameter(data.clone().detach(), requires_grad=True)
        self.mipmaps = []
        self.is_normal_map = False
        self.name = name

    def clone(self):
        return TextureModule(self.texture.clone().detach(), min_max=self.min_max, max_mip_level=self.max_mip_level, name=self.name + "_clone")

    def pin_memory_(self):
        cudart = torch.cuda.cudart()
        torch.cuda.check_error(cudart.cudaHostRegister(self.texture.data_ptr(), self.texture.numel() * self.texture.element_size(), 0))
        return self
    
    def share_memory_(self):
        self.texture.share_memory_()
        return self
    
    def requires_grad_(self, requires_grad: bool):
        self.texture.requires_grad_(requires_grad)
        return self
    
    def cache_clear(self):
        pass

    def get_trainable_params(self) -> list[torch.Tensor]:
        return [self.texture]

    def set_max_mip_level(self, max_mip_level):
        if max_mip_level is None:
            self.max_mip_level = int(math.log2(min(self.width, self.height)))
        else:
            self.max_mip_level = int(max_mip_level)
            assert(self.max_mip_level >= 0)

    def feedback(self, texc, texc_deriv, filter_mode="linear-mipmap-linear", boundary_mode="wrap", mask=None) -> list[list[torch.Tensor]]:
        return None

    def sample(self, texc, texc_deriv, filter_mode="linear-mipmap-linear", boundary_mode="wrap", mask=None) -> torch.Tensor:
        cuda_texture = self.texture.cuda()
        mipmaps = [cuda_texture]
        while mipmaps[-1].shape[1] > 1 and mipmaps[-1].shape[2] > 1 and len(mipmaps)-1 < self.max_mip_level:
            mipmaps += [texture2d_mip.apply(mipmaps[-1])]
        mipmaps = mipmaps[1:]
                
        out = dr.texture(
            cuda_texture, texc, texc_deriv, None, mipmaps,
            filter_mode, boundary_mode)
        if mask is not None:
            out = out*mask
        return out
    
    @torch.no_grad()
    def texture_mipmap_fwd(self, max_mip_level=None):
        self.set_max_mip_level(max_mip_level)
        if self.min_max is not None:
            for i in range(self.channels):
                self.texture.clamp_(min=self.min_max[0][i], max=self.min_max[1][i])
        if self.is_normal_map:
            util.safe_normalize_(self.texture)

    @torch.no_grad()
    def texture_mipmap_bwd(self) -> list[torch.Tensor]:
        if self.texture.grad is None:
            return [torch.zeros_like(self.texture)]
        return [self.texture.grad.cuda()]
    
    def to_tensor(self) -> torch.Tensor:
        return self.texture.clone()
    
    def offload(self):
        pass

class VirtualTextureModule(nn.Module):

    def __init__(self, data: torch.Tensor, 
                 min_max=None, 
                 max_mip_level=None,
                 name="",
                 page_size_x=256,
                 page_size_y=256,
                 fp16_texture=False,
                 lru_cache_max_size=1000) -> None:
        # data must have shape [batch, height, width, channels]
        super().__init__()
        self.page_size_x = page_size_x
        self.page_size_y = page_size_y
        self.use_fp16 = fp16_texture
        self.min_max = min_max
        self.name = name
        self.pages: list[nn.Parameter] = self._construct_pages(data)
        self.height = data.shape[1]
        self.width = data.shape[2]
        self.channels = data.shape[3]
        assert(is_power_of_two(self.page_size_x))
        assert(is_power_of_two(self.page_size_y))
        assert(is_power_of_two(self.width))
        assert(is_power_of_two(self.height))
        assert(self.width % self.page_size_x == 0)
        assert(self.height % self.page_size_y == 0)
        self.set_max_mip_level(max_mip_level)
        self.is_normal_map = False
        
        self.mipmaps: list[list[torch.Tensor]] = []
        self.all_streaming_time = 0 # in seconds
        self.forward_count = 0
        self.avg_streaming_time = 0 # in seconds
        self.cuda_pages_num = 0

        self.get_page: CPUOffloadCache = None
        self.lru_cache_max_size = lru_cache_max_size
        self.cache_infos = []
        self.total_num_used_pages = 0

        mip_width, mip_height = self.width, self.height
        for mip_level in range(1, self.max_mip_level+1):
            mip_width = max(mip_width // 2, 1)
            mip_height = max(mip_height // 2, 1)
            num_pages_x = math.ceil(mip_width / self.page_size_x)
            num_pages_y = math.ceil(mip_height / self.page_size_y)
            mip_page_size_x = min(mip_width, self.page_size_x)
            mip_page_size_y = min(mip_height, self.page_size_y)
            mipmap = []
            for i in range(num_pages_x * num_pages_y):
                page = torch.zeros([1, mip_page_size_y, mip_page_size_x, self.channels], dtype=torch.float32)
                page = nn.Parameter(page)
                mipmap.append(page)
            self.mipmaps.append(mipmap)

        self.texture_mipmap_fwd(max_mip_level)

    def clone(self) -> "VirtualTextureModule":
        data = ([page.clone().detach() for page in self.pages], self.height, self.width, self.channels)
        return VirtualTextureModule(
            data, 
            min_max=self.min_max, 
            max_mip_level=self.max_mip_level, 
            name=self.name + "_clone",
            page_size_x=self.page_size_x,
            page_size_y=self.page_size_y,
            fp16_texture=self.use_fp16,
            lru_cache_max_size=self.lru_cache_max_size)

    def pin_memory_(self):
        cudart = torch.cuda.cudart()
        mipmaps = [self.pages] + self.mipmaps
        for mipmap in mipmaps:
            for page in mipmap:
                torch.cuda.check_error(cudart.cudaHostRegister(page.data_ptr(), page.numel() * page.element_size(), 0))
        return self
    
    def share_memory_(self):
        mipmaps = [self.pages] + self.mipmaps
        for mipmap in mipmaps:
            for page in mipmap:
                page.share_memory_()
        return self

    def requires_grad_(self, requires_grad: bool):
        mipmaps = [self.pages] + self.mipmaps
        for mipmap in mipmaps:
            for page in mipmap:
                page.requires_grad_(requires_grad)
        return self
    
    def cache_clear(self):
        if self.get_page is not None:
            self.get_page.cache_clear()

    def get_trainable_params(self) -> list[torch.Tensor]:
        return self.pages

    def set_max_mip_level(self, max_mip_level):
        if max_mip_level is None:
            self.max_mip_level = int(math.log2(min(self.width, self.height)))
        else:
            self.max_mip_level = int(max_mip_level)
            assert(self.max_mip_level >= 0)

    def offload(self):
        if self.get_page is None:
            return
        self.get_page.offload()

    def feedback(self, texc, texc_deriv, filter_mode="linear-mipmap-linear", boundary_mode="wrap", mask=None) -> list[list[torch.Tensor]]:
        if self.get_page is None:
            # 由于这个不能被pickle，所以只能lazy init
            self.get_page = CPUOffloadCache(lru_cache_max_size=self.lru_cache_max_size, factory_fn=self._get_page)

        used_pages: dict[tuple[int, int], torch.Tensor] = {}
                
        timers("virtual_texture_feedback").start()
        feedback = dr.virtual_texture_feedback(
            1, self.height, self.width, self.channels, 
            texc, texc_deriv, None, mask,
            filter_mode, boundary_mode, 
            self.page_size_x, self.page_size_y, self.max_mip_level)
        feedback = [torch.where(mipmap)[0].tolist() for mipmap in feedback]
        timers("virtual_texture_feedback").stop()
        
        timers("virtual_texture_streaming").start()
        streaming_start = time.time()
        for mip_level in range(len(feedback)):
            for page_idx in feedback[mip_level]:
                used_pages[(mip_level, page_idx)] = self.get_page(mip_level, page_idx)
        self.total_num_used_pages += len(used_pages)
        streaming_end = time.time()
        streaming_time = streaming_end - streaming_start
        self.all_streaming_time += streaming_time
        self.forward_count += 1
        self.avg_streaming_time = self.all_streaming_time / self.forward_count
        timers("virtual_texture_streaming").stop()
        return used_pages

    def sample(self, texc, texc_deriv, filter_mode="linear-mipmap-linear", boundary_mode="wrap", mask=None) -> torch.Tensor:
        used_pages: dict[tuple[int, int], torch.Tensor] = self.feedback(texc, texc_deriv, filter_mode, boundary_mode, mask)
        timers("virtual_texture sampling").start()
        out = dr.virtual_texture(
            1, self.height, self.width, self.channels, 
            used_pages, texc, texc_deriv, None, mask,
            filter_mode, boundary_mode, 
            self.page_size_x, self.page_size_y, self.max_mip_level)
        timers("virtual_texture sampling").stop()
        return out
    
    @torch.no_grad()
    def texture_mipmap_fwd(self, max_mip_level=None):
        self.set_max_mip_level(max_mip_level)
        cuda_pages: list[torch.Tensor] = []
        for page in self.pages:
            cuda_pages.append(page.to("cuda", dtype=torch.float16 if self.use_fp16 else torch.float32, non_blocking=True))
        for page in cuda_pages:
            if self.min_max is not None:
                for i in range(page.shape[-1]):
                    page[..., i].clamp_(min=self.min_max[0][i], max=self.min_max[1][i])
            if self.is_normal_map:
                util.safe_normalize_(page)
        for page_idx, page in enumerate(cuda_pages):
            self.pages[page_idx].copy_(page, non_blocking=True)
        cuda_mipmaps = dr.virtual_texture_mipmap_fwd(
            1, self.height, self.width, self.channels,
            cuda_pages, self.page_size_x, self.page_size_y, max_mip_level=self.max_mip_level)
        for mip_level, cuda_mipmap in enumerate(cuda_mipmaps):
            for page_idx, page in enumerate(cuda_mipmap):
                self.mipmaps[mip_level][page_idx].copy_(page, non_blocking=True)

    @torch.no_grad()
    def texture_mipmap_bwd(self) -> list[torch.Tensor]:

        self.get_page.wait_for_offload()

        cpu_grads = self.get_page.get_offloaded_grads()
        cuda_pages = self.get_page.get_cached_items()
        cuda_grads = {}
        for key, cpu_grad in cpu_grads:
            cuda_grads[key] = cpu_grad.to("cuda", non_blocking=True)
        for key, cuda_page in cuda_pages:
            if cuda_page.grad is not None:
                if key not in cuda_grads:
                    cuda_grads[key] = cuda_page.grad
                else:
                    cuda_grads[key] += cuda_page.grad
                cuda_page.grad = None
        self.get_page.offload_clear()
        self.get_page.cache_clear()
        
        empty = torch.tensor([])
        all_pages_grads = [[empty] * len(mip) for mip in [self.pages]+self.mipmaps]
        for key, cuda_grad in cuda_grads.items():
            mip_level, page_idx = key
            all_pages_grads[mip_level][page_idx] = cuda_grad
        for page_idx in range(len(self.pages)):
            if all_pages_grads[0][page_idx] is empty:
                all_pages_grads[0][page_idx] = torch.zeros_like(self.pages[page_idx], device="cuda")
        dr.virtual_texture_mipmap_bwd(1, self.height, self.width, self.channels, all_pages_grads, page_size_x=self.page_size_x, page_size_y=self.page_size_y)
        return all_pages_grads[0]

    def to_tensor(self) -> torch.Tensor:
        res = []
        page_num_y = math.ceil(self.height / self.page_size_y)
        page_num_x = math.ceil(self.width / self.page_size_x)
        for x in range(page_num_x):
            col = []
            for y in range(page_num_y):
                col.append(self.pages[y + x*page_num_y].cpu())
            col = torch.cat(col, dim=2)
            res.append(col)
        res = torch.cat(res, dim=1)
        return res

    def _get_page(self, mip: int, i: int):
        all_pages = [self.pages] + self.mipmaps
        dtype = torch.float16 if self.use_fp16 else torch.float32
        cpu_page = all_pages[mip][i]
        cuda_page: torch.Tensor = cpu_page.to(device="cuda", dtype=dtype, non_blocking=True).detach().requires_grad_(cpu_page.requires_grad)
        return cuda_page

    def _construct_pages(self, data: torch.Tensor) -> list[nn.Parameter]:
        shape = data.shape
        page_num_y = math.ceil(shape[1] / self.page_size_y)
        page_num_x = math.ceil(shape[2] / self.page_size_x)
        pages = []
        page_id = 0
        for y in range(page_num_y):
            for x in range(page_num_x):
                start_y = y * self.page_size_y
                start_x = x * self.page_size_x
                end_y = min((1+y) * self.page_size_y, shape[1])
                end_x = min((1+x) * self.page_size_x, shape[2])
                page: nn.Parameter = nn.Parameter(data[:, start_y:end_y, start_x:end_x, :].clone())
                page_id += 1
                pages.append(page)

        return pages
    