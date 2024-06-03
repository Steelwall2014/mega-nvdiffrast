# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch
import nvdiffrast.torch as dr
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
from torch.utils.checkpoint import checkpoint
from configs import Configuration

import util
import renderutils as ru

######################################################################################
# Utility functions
######################################################################################

class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return util.avg_pool_nhwc(cubemap, (2,2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                                    indexing='ij')
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################

class Encoder(torch.nn.Module):
    def __init__(self, n_frequencies, dtype=torch.float16):
        super(Encoder, self).__init__()
        self.freq_bands = 2 ** torch.linespace(0, n_frequencies-1, n_frequencies)
        self.n_frequencies = n_frequencies
        self.dtype = dtype

    def forward(self, x):
        x = x[..., None] * self.freq_bands[None, None, None, :]
        x = torch.cat([torch.sin(x), torch.cos(x)], -1)
        x = x.to(self.dtype)
        return x

class EnvironmentLight(torch.nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base):
        super(EnvironmentLight, self).__init__()
        self.mtx = None      
        self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=True)
        self.register_parameter('env_base', self.base)

    def xfm(self, mtx):
        self.mtx = mtx

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        return torch.where(roughness < self.MAX_ROUGHNESS
                        , (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
                        , (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2)
        
    def build_mips(self, cutoff=0.99):
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 
        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))

    def shade(self, gb_pos, gb_normal, kd, ks, view_pos, specular=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        wo = util.safe_normalize(view_pos - gb_pos)

        if specular:
            roughness = ks[..., 1:2] # y component
            metallic  = ks[..., 2:3] # z component
            spec_col  = (1.0 - metallic)*0.04 + kd * metallic
            diff_col  = kd * (1.0 - metallic)
        else:
            diff_col = kd

        reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
        nrmvec = gb_normal
        if self.mtx is not None: # Rotate lookup
            mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
            # 有时候会backward会产生nan，所以还是用python实现吧
            reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx, use_python=True).view(*reflvec.shape)
            nrmvec  = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx, use_python=True).view(*nrmvec.shape)

        # Diffuse lookup
        diffuse = dr.texture(self.diffuse[None, ...], nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube')
        shaded_col = diffuse * diff_col

        diffuse_r = dr.texture(self.diffuse[None, ...], reflvec.contiguous(), filter_mode='linear', boundary_mode='cube')

        spec = None
        if specular:
            # Lookup FG term from lookup texture
            NdotV = torch.clamp(util.dot(wo, gb_normal), min=1e-4)
            fg_uv = torch.cat((NdotV, roughness), dim=-1)
            if not hasattr(self, '_FG_LUT'):
                self._FG_LUT = torch.as_tensor(np.fromfile('data/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
            fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')

            # Roughness adjusted specular env lookup
            miplevel = self.get_mip(roughness)
            spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')

            # Compute aggregate lighting
            reflectance = spec_col * fg_lookup[...,0:1] + fg_lookup[...,1:2]
            shaded_col += spec * reflectance

        return shaded_col * (1.0 - ks[..., 0:1]) # Modulate by hemisphere visibility
    
    @torch.no_grad()
    def serialize(self):
        return self.base.cpu()
    
    @staticmethod
    def deserialize(state):
        return EnvironmentLight(state)   

######################################################################################
# Load and store
######################################################################################

# Load from latlong .HDR file
def _load_env_hdr(fn, scale=1.0):
    latlong_img = torch.tensor(util.load_image(fn), dtype=torch.float32, device='cuda')*scale
    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])

    l = EnvironmentLight(cubemap)
    l.build_mips()

    return l

def load_env(fn, scale=1.0):
    if os.path.splitext(fn)[1].lower() == ".hdr":
        return _load_env_hdr(fn, scale)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]

def save_env_map(fn, light):
    assert isinstance(light, EnvironmentLight), "Can only save EnvironmentLight currently"
    if isinstance(light, EnvironmentLight):
        color = util.cubemap_to_latlong(light.base, [512, 1024])
    util.save_image_raw(fn, color.detach().cpu().numpy())

######################################################################################
# Create trainable env map with random initialization
######################################################################################

def create_trainable_env_rnd(base_res, scale=0.5, bias=0.25):
    base = torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device='cuda') * scale + bias
    return EnvironmentLight(base)
      
def get_box_points(AABBs_min: torch.Tensor, AABBs_max: torch.Tensor):
    # 获取AABB的8个顶点
    min_x = AABBs_min[..., 0:1] # [num_clusters, 1]
    min_y = AABBs_min[..., 1:2] # [num_clusters, 1]
    min_z = AABBs_min[..., 2:3] # [num_clusters, 1]
    max_x = AABBs_max[..., 0:1] # [num_clusters, 1]
    max_y = AABBs_max[..., 1:2] # [num_clusters, 1]
    max_z = AABBs_max[..., 2:3] # [num_clusters, 1]
    box_points = [torch.cat([min_x, min_y, min_z], dim=1),  # [num_clusters, 3]
                  torch.cat([max_x, min_y, min_z], dim=1),  # [num_clusters, 3]
                  torch.cat([min_x, max_y, min_z], dim=1),  # [num_clusters, 3]
                  torch.cat([max_x, max_y, min_z], dim=1),  # [num_clusters, 3]
                  torch.cat([min_x, min_y, max_z], dim=1),  # [num_clusters, 3]
                  torch.cat([max_x, min_y, max_z], dim=1),  # [num_clusters, 3]
                  torch.cat([min_x, max_y, max_z], dim=1),  # [num_clusters, 3]
                  torch.cat([max_x, max_y, max_z], dim=1)]  # [num_clusters, 3]
    return torch.cat(box_points, dim=0)  # [num_clusters*8, 3]

class DirectionalLight(torch.nn.Module):

    def __init__(self, light_direction, light_color, AABB: tuple[torch.Tensor, torch.Tensor], vsm_resolution=[8192,8192], page_size_x=512, page_size_y=512, first_level_extent=30, cast_shadows=True) -> None:
        super().__init__()
        if isinstance(light_direction, list):
            light_direction = torch.tensor(light_direction, dtype=torch.float32, device="cuda") 
        if isinstance(light_color, list):
            light_color = torch.tensor(light_color, dtype=torch.float32, device="cuda")
        light_direction = util.safe_normalize(light_direction)
        self.light_direction = light_direction.clone().detach().cuda()
        self.light_color = torch.nn.Parameter(light_color)
        self.light_position = torch.zeros_like(light_direction)
        self.vsm_resolution = vsm_resolution
        self.page_size_x = page_size_x
        self.page_size_y = page_size_y
        self.first_level_extent = first_level_extent
        self.cast_shadows = cast_shadows

        self.update_light_position(AABB)

    @torch.no_grad()
    def update_light_position(self, AABB: tuple[torch.Tensor, torch.Tensor]):
        """
        用场景的bounding box更新光源的位置, 让光源的shadow map能正好覆盖场景
        Args:
            AABB: 场景的AABB, 是一个二元组, 分别是AABB的min和max。
                  min和max的shape均为[num_clusters, 3]
        """
        AABB_min, AABB_max = AABB
        up = torch.tensor(util.WORLD_UP, dtype=torch.float32, device="cuda")
        # 如果光源方向和up方向正好相反，那么计算view矩阵的话会出现nan，所以这里加一个微小的偏移
        if torch.abs(self.light_direction + up).max() < 1e-6:  
            self.light_direction += 0.001
        # 给定一个初始的猜测光源位置，这是为了获取场景在光源空间下的包围盒
        guess_light_position = -self.light_direction.view([3])
        at = torch.tensor([0.0, 0.0, 0.0], device="cuda")
        lgt_view_matrix = util.lookAt(guess_light_position, at, up)
        box_points = get_box_points(AABB_min, AABB_max)
        box_points = torch.nn.functional.pad(box_points, (0, 1), value=1.0, mode='constant').t()    # [4, num_clusters*8]
        view_space_pos = lgt_view_matrix @ box_points   # 包围盒的8个顶点在光源空间下的坐标
        min_x = torch.min(view_space_pos[0:1, ...], dim=1).values
        min_y = torch.min(view_space_pos[1:2, ...], dim=1).values
        max_x = torch.max(view_space_pos[0:1, ...], dim=1).values
        max_y = torch.max(view_space_pos[1:2, ...], dim=1).values
        max_z = torch.max(view_space_pos[2:3, ...], dim=1).values
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        lgt_rotation = lgt_view_matrix[0:3, 0:3].t()    # 将光源空间下的向量旋转到世界空间的矩阵是view矩阵左上角的转置
        offset = torch.stack([center_x, center_y, max_z], dim=0)
        offset_world_space = lgt_rotation @ offset
        light_position = guess_light_position + offset_world_space.view([3])
        self.light_position.copy_(light_position)
        half_width = (max_x - min_x) / 2 * 1.05  # 留一点余量
        half_height = (max_y - min_y) / 2 * 1.05 # 留一点余量
        self.half_frustum_width = torch.max(half_width, half_height)