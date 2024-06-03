from collections import defaultdict
import logging
from typing import Iterable
import math
import random
import torch
import torch.nn as nn
from distribute import log_dist
from modules.material import MaterialModule
from modules.mesh import FrustumCullOutput, VirtualGeometryModule
import util
import nvdiffrast.torch as dr
import renderutils as ru

class VertexShaderOutput:
    def __init__(self):
        self.tri_pos: torch.Tensor = None   # index of positions
        self.pos: torch.Tensor = None
        self.clip_pos: torch.Tensor = None
        self.tri_uv: torch.Tensor = None    # index of uv
        self.uv: torch.Tensor = None
        self.nrm: torch.Tensor = None
        self.tng: torch.Tensor = None
        self.ranges: torch.Tensor = None
def vertex_shader(mvp: torch.Tensor, cull_out: FrustumCullOutput) -> VertexShaderOutput:
    device = mvp.device
    clip_pos_per_view = []
    num_views = len(cull_out.pos_per_view)
    num_vertices = 0
    num_uv_vertices = 0
    num_triangles = 0
    ranges = []
    for i in range(num_views):
        cull_out.tri_pos_per_view[i] += num_vertices
        ranges.append((num_triangles, cull_out.tri_pos_per_view[i].shape[0]))
        clip_pos = ru.xfm_points(cull_out.pos_per_view[i][None, ...], mvp[i:i+1, ...], use_python=True)[0]
        clip_pos_per_view.append(clip_pos)
        num_vertices += cull_out.pos_per_view[i].shape[0]
        num_triangles += cull_out.tri_pos_per_view[i].shape[0]
        cull_out.tri_uv_per_view[i] += num_uv_vertices
        num_uv_vertices += cull_out.uv_per_view[i].shape[0]
    
    vs_out = VertexShaderOutput()
    vs_out.tri_pos = torch.cat(cull_out.tri_pos_per_view, dim=0)
    vs_out.pos = torch.cat(cull_out.pos_per_view, dim=0)
    vs_out.clip_pos = torch.cat(clip_pos_per_view, dim=0)
    vs_out.tri_uv = torch.cat(cull_out.tri_uv_per_view, dim=0)
    vs_out.uv = torch.cat(cull_out.uv_per_view, dim=0)
    vs_out.nrm = torch.cat(cull_out.nrm_per_view, dim=0)
    vs_out.tng = torch.cat(cull_out.tng_per_view, dim=0)
    vs_out.ranges = torch.tensor(ranges, device="cpu", dtype=torch.int32)

    return vs_out

class RasterizeOutput:
    def __init__(self):
        self.full_res_rast: torch.Tensor = None
        self.rast: torch.Tensor = None
        self.rast_db: torch.Tensor = None
        self.gb_pos: torch.Tensor = None
        self.gb_clip_pos: torch.Tensor = None
        self.gb_texc: torch.Tensor = None
        self.gb_texd: torch.Tensor = None
        self.gb_normal: torch.Tensor = None
        self.gb_tangent: torch.Tensor = None
def rasterize(ctx, resolution, vs_out: VertexShaderOutput, spp=1, msaa=False) -> RasterizeOutput:
    if vs_out.ranges is None:  # 如果不是range mode的话，要求shape是[num_batches, num_vertices, num_channels]
        assert vs_out.clip_pos.dim() == 3
        assert vs_out.pos is None or vs_out.pos.dim() == 3
        assert vs_out.uv is None or vs_out.uv.dim() == 3
        assert vs_out.nrm is None or vs_out.nrm.dim() == 3
        assert vs_out.tng is None or vs_out.tng.dim() == 3
    else:               # 如果是range mode的话，要求shape是[num_vertices, num_channels]
        assert vs_out.clip_pos.dim() == 2
        assert vs_out.pos is None or vs_out.pos.dim() == 2
        assert vs_out.uv is None or vs_out.uv.dim() == 2
        assert vs_out.nrm is None or vs_out.nrm.dim() == 2
        assert vs_out.tng is None or vs_out.tng.dim() == 2

    full_resolution = (resolution[0] * spp, resolution[1] * spp)
    rast_out = RasterizeOutput()
    rast, rast_db = dr.rasterize(ctx, vs_out.clip_pos, vs_out.tri_pos, full_resolution, ranges=vs_out.ranges, grad_db=True)
    if spp > 1 and msaa:
        rast_out.rast = util.scale_img_nhwc(rast, resolution, mag='nearest', min='nearest')
        rast_out.rast_db = util.scale_img_nhwc(rast_db, resolution, mag='nearest', min='nearest') * spp
    else:
        rast_out.rast = rast
        rast_out.rast_db = rast_db
    rast_out.full_res_rast = rast

    rast_out.gb_clip_pos, _ = dr.interpolate(vs_out.clip_pos, rast_out.rast, vs_out.tri_pos)
    rast_out.gb_pos, _ = dr.interpolate(vs_out.pos, rast_out.rast, vs_out.tri_pos)
    rast_out.gb_texc, rast_out.gb_texd = dr.interpolate(vs_out.uv, rast_out.rast, vs_out.tri_uv, rast_db=rast_out.rast_db, diff_attrs="all")
    rast_out.gb_normal, _ = dr.interpolate(vs_out.nrm, rast_out.rast, vs_out.tri_pos)
    rast_out.gb_tangent, _ = dr.interpolate(vs_out.tng, rast_out.rast, vs_out.tri_pos)

    return rast_out
