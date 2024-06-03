from collections import deque
import logging
import os
from typing import Iterable
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from configs import Configuration
from distribute import get_local_rank, log_dist
import util
import nvdiffrast.torch as dr
import renderutils as ru

from .deffered_cache import CPUOffloadCache

from timer import timers

def auto_normals(positions, faces):
    faces = faces.long()

    i0 = faces[:, 0]
    i1 = faces[:, 1]
    i2 = faces[:, 2]

    v0 = positions[i0, :]
    v1 = positions[i1, :]
    v2 = positions[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(positions)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(util.dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_nrm = util.safe_normalize(v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))

    return v_nrm

def compute_tangents(positions, normals, faces, texcoords, uv_faces):
    faces = faces.long()
    uv_faces = uv_faces.long()

    vn_idx = [None] * 3
    pos = [None] * 3
    tex = [None] * 3
    for i in range(0,3):
        pos[i] = positions[faces[:, i]]
        tex[i] = texcoords[uv_faces[:, i]]
        vn_idx[i] = faces[:, i]

    tangents = torch.zeros_like(normals)

    # Compute tangent space for each triangle
    uve1 = tex[1] - tex[0]
    uve2 = tex[2] - tex[0]
    pe1  = pos[1] - pos[0]
    pe2  = pos[2] - pos[0]
    
    nom   = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
    denom = (uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1])
    
    # Avoid division by zero for degenerated texture coordinates
    tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

    # Update all 3 vertices
    for i in range(0,3):
        idx = vn_idx[i][:, None].repeat(1,3)
        tangents.scatter_add_(0, idx, tang)                # tangents[n_i] = tangents[n_i] + tang

    # Normalize and make sure tangent is perpendicular to normal
    tangents = util.safe_normalize(tangents)
    tangents = util.safe_normalize(tangents - util.dot(tangents, normals) * normals)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(tangents))

    return tangents

def _cluster_normal_tangent_forward(positions: torch.Tensor, faces: torch.Tensor, texcoords: torch.Tensor, tfaces: torch.Tensor, new_to_old: list[torch.Tensor]):
    with torch.no_grad():
        positions = positions.cuda()
        faces = faces.cuda()
        texcoords = texcoords.cuda()
        tfaces = tfaces.cuda()
        normals = auto_normals(positions, faces)
        tangents = compute_tangents(positions, normals, faces, texcoords, tfaces)
        num_clusters = len(new_to_old)
        ClusterNormals = []
        ClusterTangents = []
        for cluster_idx in range(num_clusters):
            old_indices = new_to_old[cluster_idx]
            ClusterNormals.append(normals[old_indices])
            ClusterTangents.append(tangents[old_indices])
        return ClusterNormals, ClusterTangents
    
def _cluster_normal_tangent_backward(positions: torch.Tensor, faces: torch.Tensor, texcoords: torch.Tensor, tfaces: torch.Tensor, new_to_old: list[torch.Tensor], 
                                               grad_cluster_normals: list[torch.Tensor], grad_cluster_tangents: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        timers("_cluster_normal_tangent_backward_fwd").start()
        positions = positions.cuda().detach().requires_grad_(True)
        faces = faces.cuda()
        texcoords = texcoords.cuda().detach().requires_grad_(True)
        tfaces = tfaces.cuda()
        normals = auto_normals(positions, faces)
        tangents = compute_tangents(positions, normals, faces, texcoords, tfaces)
        num_clusters = len(new_to_old)
        ClusterNormals = []
        ClusterTangents = []
        for cluster_idx in range(num_clusters):
            old_indices = new_to_old[cluster_idx]
            ClusterNormals.append(normals[old_indices])
            ClusterTangents.append(tangents[old_indices])
        timers("_cluster_normal_tangent_backward_fwd").stop()
        timers("_cluster_normal_tangent_backward_bwd").start()
        torch.autograd.backward(ClusterNormals + ClusterTangents, grad_cluster_normals + grad_cluster_tangents)
        timers("_cluster_normal_tangent_backward_bwd").stop()
        return positions.grad, texcoords.grad

def extract_frustum(mvp: torch.Tensor):

    def normalize_plane(plane: torch.Tensor):
        length_of_normal = plane[:, 0:3].square().sum(1, keepdim=True).sqrt()
        return plane / length_of_normal
    batch = mvp.shape[0]
    left = normalize_plane(mvp.select(1, 3) + mvp.select(1, 0))
    right = normalize_plane(mvp.select(1, 3) - mvp.select(1, 0))
    top = normalize_plane(mvp.select(1, 3) - mvp.select(1, 1))
    bottom = normalize_plane(mvp.select(1, 3) + mvp.select(1, 1))
    near = normalize_plane(mvp.select(1, 3) + mvp.select(1, 2))
    far = normalize_plane(mvp.select(1, 3) - mvp.select(1, 2))
    frustums = []
    for b in range(batch):
        frustum = torch.cat([left[b:b+1, ...], right[b:b+1, ...], top[b:b+1, ...], bottom[b:b+1, ...], near[b:b+1, ...], far[b:b+1, ...]])
        frustums.append(frustum[None, ...])
    frustums = torch.cat(frustums)

    return frustums

class FrustumCullOutput:
    def __init__(self):
        self.tri_pos_per_view: list[torch.Tensor] = []
        self.tri_uv_per_view: list[torch.Tensor] = []
        self.pos_per_view: list[torch.Tensor] = []
        self.uv_per_view: list[torch.Tensor] = []
        self.nrm_per_view: list[torch.Tensor] = []
        self.tng_per_view: list[torch.Tensor] = []

class IGeometryModule(nn.Module):
    def __init__(self):
        super().__init__()

    def pin_memory_(self):
        raise NotImplementedError

    def share_memory_(self):
        raise NotImplementedError

    def requires_grad_(self, requires_grad: bool):
        raise NotImplementedError

    def cache_clear(self):
        raise NotImplementedError

    def get_trainable_params(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        raise NotImplementedError
    
    def frustum_cull(self, mvp: torch.Tensor) -> FrustumCullOutput:
        raise NotImplementedError

    def offload(self):
        raise NotImplementedError

    def get_AABB(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def normal_tangent_fwd(self):
        raise NotImplementedError

    def normal_tangent_bwd(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        raise NotImplementedError

    def update_bounding_box(self):
        raise NotImplementedError

class GeometryModule(IGeometryModule):

    def __init__(self, Name: str, 
                 Positions: torch.Tensor, 
                 TexCoords: torch.Tensor, 
                 PosIndexes: torch.Tensor, 
                 UVIndexes: torch.Tensor):
        super().__init__()
        self.Name = Name
        self.Positions = nn.Parameter(Positions.clone().detach(), requires_grad=True)
        self.TexCoords = nn.Parameter(TexCoords.clone().detach(), requires_grad=False)
        self.PosIndexes = nn.Parameter(PosIndexes.int().clone().detach(), requires_grad=False)
        self.UVIndexes = nn.Parameter(UVIndexes.int().clone().detach(), requires_grad=False)
        self.update_bounding_box()

    def pin_memory_(self):
        cudart = torch.cuda.cudart()
        torch.cuda.check_error(cudart.cudaHostRegister(self.Positions.data_ptr(), self.Positions.numel() * self.Positions.element_size(), 0))
        torch.cuda.check_error(cudart.cudaHostRegister(self.TexCoords.data_ptr(), self.TexCoords.numel() * self.TexCoords.element_size(), 0))
        torch.cuda.check_error(cudart.cudaHostRegister(self.PosIndexes.data_ptr(), self.PosIndexes.numel() * self.PosIndexes.element_size(), 0))
        torch.cuda.check_error(cudart.cudaHostRegister(self.UVIndexes.data_ptr(), self.UVIndexes.numel() * self.UVIndexes.element_size(), 0))
        return self

    def share_memory_(self):
        self.Positions.share_memory_()
        return self
    
    def requires_grad_(self, requires_grad: bool):
        self.Positions.requires_grad_(requires_grad)
        return self
    
    def cache_clear():
        pass

    def get_trainable_params(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return [self.Positions], [self.TexCoords]
    
    def frustum_cull(self, mvp: torch.Tensor) -> FrustumCullOutput:
        cull_out = FrustumCullOutput()
        num_views = mvp.shape[0]
        pos = self.Positions.cuda()
        tri_pos = self.PosIndexes.cuda()
        uv = self.TexCoords.cuda()
        tri_uv = self.UVIndexes.cuda()
        nrm = auto_normals(pos, tri_pos)
        tng = compute_tangents(pos, nrm, tri_pos, uv, tri_uv)
        for view_idx in range(num_views):
            cull_out.pos_per_view.append(pos)
            cull_out.tri_pos_per_view.append(tri_pos)
            cull_out.uv_per_view.append(uv)
            cull_out.tri_uv_per_view.append(tri_uv)
            cull_out.nrm_per_view.append(nrm)
            cull_out.tng_per_view.append(tng)
        return cull_out

    def offload(self):
        pass
    
    def get_AABB(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.AABB = self.AABB.cuda()
        return self.AABB[:, 0:3], self.AABB[:, 3:6]
    
    @torch.no_grad()
    def normal_tangent_bwd(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pos_grad = self.Positions.grad
        if pos_grad is None:
            pos_grad = torch.zeros_like(self.Positions)
        uv_grad = self.TexCoords.grad
        if uv_grad is None:
            uv_grad = torch.zeros_like(self.TexCoords)
        return [pos_grad.cuda()], [uv_grad.cuda()]
    
    @torch.no_grad()
    def normal_tangent_fwd(self):
        pass

    @torch.no_grad()
    def update_bounding_box(self, use_cuda=True):
        Min = self.Positions.min(0, True)[0]
        Max = self.Positions.max(0, True)[0]
        self.AABB = torch.cat([Min, Max], dim=1)
    
def construct_clusters(cluster_triangles: list[torch.Tensor], attribute: torch.Tensor, attr_faces: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    old_to_new_mapping = torch.zeros(attribute.size(0), dtype=torch.int64)
    new_to_old_mapping = []
    cluster_attributes = []
    cluster_indices = []
    for tri_idx in cluster_triangles:
        vert_index: torch.Tensor = attr_faces[tri_idx].long().reshape(-1)
        sorted_unique_vert_index: torch.Tensor = torch.unique(vert_index, sorted=True)
        old_to_new_mapping[sorted_unique_vert_index] = torch.arange(sorted_unique_vert_index.size(0))
        attr: torch.Tensor = attribute[sorted_unique_vert_index].clone()
        tri: torch.Tensor = old_to_new_mapping[vert_index].reshape(-1, 3)
        new_to_old_mapping.append(sorted_unique_vert_index)
        cluster_attributes.append(attr)
        cluster_indices.append(tri)
    return cluster_attributes, cluster_indices, new_to_old_mapping

class VirtualGeometryModule(IGeometryModule):

    def __init__(self, Name: str, 
                 cluster_positions: list[torch.Tensor], cluster_pos_indices: list[torch.Tensor], pos_new_to_old: torch.Tensor, 
                 cluster_texcoords: list[torch.Tensor], cluster_uv_indices: list[torch.Tensor], uv_new_to_old: torch.Tensor,
                 shared_verts: torch.Tensor, shared_verts_offsets: torch.Tensor,
                 initial_positions: torch.Tensor, initial_pos_indices: torch.Tensor,
                 initial_texcoords: torch.Tensor, initial_uv_indices: torch.Tensor,
                 lru_cache_max_size=200):
        super().__init__()
        self.Name = Name

        self.pos_new_to_old = pos_new_to_old
        self.uv_new_to_old = uv_new_to_old
        self.NumClusters = len(cluster_positions)
        self.shared_verts: torch.Tensor = shared_verts
        self.shared_verts_offsets: torch.Tensor = shared_verts_offsets
        self.lru_cache_max_size = lru_cache_max_size
        
        self.ClusterPositions: list[nn.Parameter] = []
        self.ClusterNormals: list[nn.Parameter] = []
        self.ClusterTangents: list[nn.Parameter] = []
        for i, positions in enumerate(cluster_positions):
            positions = nn.Parameter(positions.clone())
            self.ClusterPositions.append(positions)
            Normals = nn.Parameter(torch.zeros_like(positions))
            self.ClusterNormals.append(Normals)
            Tangents = nn.Parameter(torch.zeros_like(positions))
            self.ClusterTangents.append(Tangents)
        self.update_bounding_box(use_cuda=False)

        self.ClusterTexCoords: list[nn.Parameter] = []
        for i, texCoords in enumerate(cluster_texcoords):
            texCoords = nn.Parameter(texCoords.clone())
            self.ClusterTexCoords.append(texCoords)

        self.ClusterPosIndexes: list[nn.Parameter] = []
        self.ClusterUVIndexes: list[nn.Parameter] = []
        for i, (pos_indices, uv_indices) in enumerate(zip(cluster_pos_indices, cluster_uv_indices)):
            assert pos_indices.shape[0] == uv_indices.shape[0]
            pos_indices = nn.Parameter(pos_indices.clone().int(), requires_grad=False)
            self.ClusterPosIndexes.append(pos_indices)
            uv_indices = nn.Parameter(uv_indices.clone().int(), requires_grad=False)
            self.ClusterUVIndexes.append(uv_indices)
        
        self._initial_positions: torch.Tensor = initial_positions
        self._initial_pos_indices: torch.Tensor = initial_pos_indices
        self._initial_texcoords: torch.Tensor = initial_texcoords
        self._initial_uv_indices: torch.Tensor = initial_uv_indices

        self.get_position: CPUOffloadCache = None
        self.get_pos_indices: CPUOffloadCache = None
        self.get_uv_indices: CPUOffloadCache = None
        self.get_texcoord: CPUOffloadCache = None
        self.get_normal: CPUOffloadCache = None
        self.get_tangent: CPUOffloadCache = None
        self.caches: list[CPUOffloadCache] = []

        self.cache_infos = []
        self.total_num_used_clusters = 0
        self.forward_count = 0
        self.cluster_usage_count = torch.zeros(len(self.ClusterPositions), dtype=torch.int32, device="cpu")

        self._cluster_attributes: list[list[torch.Tensor]] = [self.ClusterPositions, self.ClusterTexCoords, self.ClusterPosIndexes, self.ClusterUVIndexes, self.ClusterNormals, self.ClusterTangents]

        with torch.no_grad():
            CudaNormals, CudaTangents = _cluster_normal_tangent_forward(
                self._initial_positions, self._initial_pos_indices, 
                self._initial_texcoords, self._initial_uv_indices, 
                self.pos_new_to_old)

            for i, (normal, tangent) in enumerate(zip(CudaNormals, CudaTangents)):
                self.ClusterNormals[i].copy_(normal, non_blocking=True)
                self.ClusterTangents[i].copy_(tangent, non_blocking=True)

    def pin_memory_(self):
        # 由于pytorch没有inplace的pin_memory，所以这里只能用cuda的api来注册
        cudart = torch.cuda.cudart()
        for cluster in self.ClusterPositions:
            torch.cuda.check_error(cudart.cudaHostRegister(cluster.data_ptr(), cluster.numel() * cluster.element_size(), 0))
        for cluster in self.ClusterTexCoords:
            torch.cuda.check_error(cudart.cudaHostRegister(cluster.data_ptr(), cluster.numel() * cluster.element_size(), 0))
        for cluster in self.ClusterPosIndexes:
            torch.cuda.check_error(cudart.cudaHostRegister(cluster.data_ptr(), cluster.numel() * cluster.element_size(), 0))
        for cluster in self.ClusterUVIndexes:
            torch.cuda.check_error(cudart.cudaHostRegister(cluster.data_ptr(), cluster.numel() * cluster.element_size(), 0))
        for cluster in self.ClusterNormals:
            torch.cuda.check_error(cudart.cudaHostRegister(cluster.data_ptr(), cluster.numel() * cluster.element_size(), 0))
        for cluster in self.ClusterTangents:
            torch.cuda.check_error(cudart.cudaHostRegister(cluster.data_ptr(), cluster.numel() * cluster.element_size(), 0))
        return self

    def share_memory_(self):
        for cluster in self.ClusterPositions:
            cluster.share_memory_()
        for cluster in self.ClusterTexCoords:
            cluster.share_memory_()
        for cluster in self.ClusterPosIndexes:
            cluster.share_memory_()
        for cluster in self.ClusterUVIndexes:
            cluster.share_memory_()
        for cluster in self.ClusterNormals:
            cluster.share_memory_()
        for cluster in self.ClusterTangents:
            cluster.share_memory_()
        return self
    
    def requires_grad_(self, requires_grad: bool):
        for cluster in self.ClusterPositions:
            cluster.requires_grad_(requires_grad)
        for cluster in self.ClusterNormals:
            cluster.requires_grad_(requires_grad)
        for cluster in self.ClusterTangents:
            cluster.requires_grad_(requires_grad)
        for cluster in self.ClusterTexCoords:
            cluster.requires_grad_(requires_grad)
        return self
    
    def cache_clear(self):
        self.get_position.cache_clear()
        self.get_pos_indices.cache_clear()
        self.get_uv_indices.cache_clear()
        self.get_texcoord.cache_clear()
        self.get_normal.cache_clear()
        self.get_tangent.cache_clear()

    def get_trainable_params(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self.ClusterPositions, self.ClusterTexCoords

    def offload(self):
        if self.get_position is None:
            return
        self.get_position.offload()
        self.get_pos_indices.offload()
        self.get_uv_indices.offload()
        self.get_texcoord.offload()
        self.get_normal.offload()
        self.get_tangent.offload()

    def frustum_cull(self, mvp) -> FrustumCullOutput | None:
        if self.get_position is None:
            # lazy init
            self.get_position = CPUOffloadCache(lru_cache_max_size=self.lru_cache_max_size, factory_fn=self._get_position)
            self.get_pos_indices = CPUOffloadCache(lru_cache_max_size=self.lru_cache_max_size, factory_fn=self._get_pos_indices)
            self.get_uv_indices = CPUOffloadCache(lru_cache_max_size=self.lru_cache_max_size, factory_fn=self._get_uv_indices)
            self.get_texcoord = CPUOffloadCache(lru_cache_max_size=self.lru_cache_max_size, factory_fn=self._get_texcoord)
            self.get_normal = CPUOffloadCache(lru_cache_max_size=self.lru_cache_max_size, factory_fn=self._get_normal)
            self.get_tangent = CPUOffloadCache(lru_cache_max_size=self.lru_cache_max_size, factory_fn=self._get_tangent)
            self.caches = [self.get_position, self.get_pos_indices, self.get_uv_indices, self.get_texcoord, self.get_normal, self.get_tangent]

        device = mvp.device
        timers("virtual_geometry_frustum_cull").start()
        with torch.no_grad():
            frustums = extract_frustum(mvp)
            res = dr.virtual_geometry_frustum_cull(self.AABBs.to(device), frustums)
            Culled = torch.all(res, 0)
            self.cluster_usage_count += torch.where(Culled == False, 1, 0).cpu()
            NotCulled = torch.where(Culled == False)[0].tolist()
            Culled = torch.where(Culled == True)[0].tolist()
            if len(NotCulled) == 0:
                return None
            NotCulledPerView = []
            for i in range(mvp.shape[0]):
                NotCulledPerView.append(torch.where(res[i] == False)[0].tolist())
            num_used_clusters = len(NotCulled)
            self.total_num_used_clusters += num_used_clusters
            self.forward_count += 1
        timers("virtual_geometry_frustum_cull").stop()

        timers("virtual_geometry_streaming").start()
        Positions = {}
        TexCoords = {}
        Normals = {}
        Tangents = {}
        PosIndexes = {}
        UVIndexes = {}
        for idx in NotCulled:
            Positions[idx] = self.get_position(idx)
            TexCoords[idx] = self.get_texcoord(idx)
            PosIndexes[idx] = self.get_pos_indices(idx)
            UVIndexes[idx] = self.get_uv_indices(idx)
            Normals[idx] = self.get_normal(idx)
            Tangents[idx] = self.get_tangent(idx)
        timers("virtual_geometry_streaming").stop()

        timers("virtual_geometry_merge").start()
        cull_out = FrustumCullOutput()
        for view_idx in range(mvp.shape[0]):
            tri_pos = []
            tri_uv = []
            pos = []
            uv = []
            nrm = []
            tng = []
            num_pos_vertices = 0
            num_uv_vertices = 0
            for idx in NotCulledPerView[view_idx]:
                tri_pos.append(PosIndexes[idx] + num_pos_vertices)
                tri_uv.append(UVIndexes[idx] + num_uv_vertices)
                pos.append(Positions[idx])
                uv.append(TexCoords[idx])
                nrm.append(Normals[idx])
                tng.append(Tangents[idx])
                num_pos_vertices += Positions[idx].shape[0]
                num_uv_vertices += TexCoords[idx].shape[0]

            if len(tri_pos) > 0:
                tri_pos = torch.cat(tri_pos)
                tri_uv = torch.cat(tri_uv)
                pos = torch.cat(pos)
                uv = torch.cat(uv)
                nrm = torch.cat(nrm)
                tng = torch.cat(tng)
            else:
                tri_pos = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
                tri_uv = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
                pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
                uv = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], device=device, dtype=torch.float32)
                nrm = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
                tng = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=torch.float32)

            cull_out.tri_pos_per_view.append(tri_pos)
            cull_out.tri_uv_per_view.append(tri_uv)
            cull_out.pos_per_view.append(pos)
            cull_out.uv_per_view.append(uv)
            cull_out.nrm_per_view.append(nrm)
            cull_out.tng_per_view.append(tng)
        timers("virtual_geometry_merge").stop()

        return cull_out

        # CUDA光栅器实现中，会将三角形的索引加1后存储成float，而float能够精确表示的整数上限是16777216，
        # TODO: 三角形的数量超过16777216-1的时候，每个minibatch分别光栅化
        
    def get_AABB(self):
        self.AABBs = self.AABBs.cuda()
        return torch.min(self.AABBs[:, 0:3], dim=0, keepdim=True).values, torch.max(self.AABBs[:, 3:6], dim=0, keepdim=True).values

    @torch.no_grad()
    def update_bounding_box(self, use_cuda=True):
        AABBs = []
        for cluster in self.ClusterPositions:
            if use_cuda:
                cluster = cluster.cuda()
            Min = cluster.min(0, True)[0]
            Max = cluster.max(0, True)[0]
            AABB = torch.cat([Min, Max], dim=1)
            AABBs.append(AABB)
        self.AABBs: torch.Tensor = torch.cat(AABBs, 0)

    @torch.no_grad()
    def normal_tangent_fwd(self):
        self.shared_verts = self.shared_verts.cuda()
        self.shared_verts_offsets = self.shared_verts_offsets.cuda()

        positions = torch.zeros_like(self._initial_positions, device="cuda")
        for pos, new_to_old in zip(self.ClusterPositions, self.pos_new_to_old):
            positions[new_to_old] = pos.cuda()

        texcoords = torch.zeros_like(self._initial_texcoords, device="cuda")
        for uv, new_to_old in zip(self.ClusterTexCoords, self.uv_new_to_old):
            texcoords[new_to_old] = uv.cuda()

        CudaNormals, CudaTangents = _cluster_normal_tangent_forward(positions, self._initial_pos_indices, texcoords, self._initial_uv_indices, self.pos_new_to_old)

        for i, (normal, tangent) in enumerate(zip(CudaNormals, CudaTangents)):
            self.ClusterNormals[i].copy_(normal, non_blocking=True)
            self.ClusterTangents[i].copy_(tangent, non_blocking=True)

    @torch.no_grad()
    def normal_tangent_bwd(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

        for cache in self.caches:
            cache.wait_for_offload()

        timers("cuda grad add").start()
        grad_cluster_positions = [torch.zeros_like(pos, device="cuda") for pos in self.ClusterPositions]
        grad_cluster_texcoords = [torch.zeros_like(uv, device="cuda") for uv in self.ClusterTexCoords]
        grad_cluster_normals = [torch.zeros_like(pos, device="cuda") for pos in self.ClusterPositions]
        grad_cluster_tangents = [torch.zeros_like(pos, device="cuda") for pos in self.ClusterPositions]
        for cluster_idx, position in self.get_position.get_cached_items():
            if position.grad is not None:
                grad_cluster_positions[cluster_idx].add_(position.grad)
        for cluster_idx, texcoord in self.get_texcoord.get_cached_items():
            if texcoord.grad is not None:
                grad_cluster_texcoords[cluster_idx].add_(texcoord.grad)
        for cluster_idx, normal in self.get_normal.get_cached_items():
            if normal.grad is not None:
                grad_cluster_normals[cluster_idx].add_(normal.grad)
        for cluster_idx, tangent in self.get_tangent.get_cached_items():
            if tangent.grad is not None:
                grad_cluster_tangents[cluster_idx].add_(tangent.grad)
        # 清除cuda上的grads
        for cache in self.caches:
            cache.cache_clear()
        timers("cuda grad add").stop()

        timers("cpu grad add").start()
        for cluster_idx, cpu_grad in self.get_position.get_offloaded_grads():
            if cpu_grad is not None:
                grad_cluster_positions[cluster_idx].add_(cpu_grad.cuda())
        for cluster_idx, cpu_grad in self.get_texcoord.get_offloaded_grads():
            if cpu_grad is not None:
                grad_cluster_texcoords[cluster_idx].add_(cpu_grad.cuda())
        for cluster_idx, cpu_grad in self.get_normal.get_offloaded_grads():
            if cpu_grad is not None:
                grad_cluster_normals[cluster_idx].add_(cpu_grad.cuda())
        for cluster_idx, cpu_grad in self.get_tangent.get_offloaded_grads():
            if cpu_grad is not None:
                grad_cluster_tangents[cluster_idx].add_(cpu_grad.cuda())
        for cache in self.caches:
            cache.offload_clear()
        timers("cpu grad add").stop()

        timers("grad merge").start()
        positions = torch.zeros_like(self._initial_positions, device="cuda")
        texcoords = torch.zeros_like(self._initial_texcoords, device="cuda")
        positions_grad = torch.zeros_like(positions, device="cuda")
        for cluster_idx in range(self.NumClusters):
            old_indices = self.pos_new_to_old[cluster_idx]
            positions[old_indices] = self.ClusterPositions[cluster_idx].cuda()
            positions_grad[old_indices] += grad_cluster_positions[cluster_idx]
            old_indices = self.uv_new_to_old[cluster_idx]
            texcoords[old_indices] = self.ClusterTexCoords[cluster_idx].cuda()
        timers("grad merge").stop()

        timers("_cluster_normal_tangent_backward").start()
        pos_grad, uv_grad = _cluster_normal_tangent_backward(positions, self._initial_pos_indices, texcoords, self._initial_uv_indices, self.pos_new_to_old, grad_cluster_normals, grad_cluster_tangents)
        positions_grad += pos_grad
        timers("_cluster_normal_tangent_backward").stop()

        for cluster_idx in range(self.NumClusters):
            old_indices = self.pos_new_to_old[cluster_idx]
            grad_cluster_positions[cluster_idx] = positions_grad[old_indices]
            old_indices = self.uv_new_to_old[cluster_idx]
            grad_cluster_texcoords[cluster_idx] += uv_grad[old_indices]
        return grad_cluster_positions, grad_cluster_texcoords

    def _get_position(self, cluster_idx: int) -> torch.Tensor:
        pos = self.ClusterPositions[cluster_idx]
        pos = pos.to("cuda", non_blocking=True).detach().requires_grad_(pos.requires_grad)
        return pos
    def _get_texcoord(self, cluster_idx: int) -> torch.Tensor:
        uv = self.ClusterTexCoords[cluster_idx]
        uv = uv.to("cuda", non_blocking=True).detach().requires_grad_(uv.requires_grad)
        return uv
    def _get_pos_indices(self, cluster_idx: int) -> torch.Tensor:
        pos_idx = self.ClusterPosIndexes[cluster_idx]
        pos_idx = pos_idx.to("cuda", non_blocking=True)
        return pos_idx
    def _get_uv_indices(self, cluster_idx: int) -> torch.Tensor:
        uv_idx = self.ClusterUVIndexes[cluster_idx]
        uv_idx = uv_idx.to("cuda", non_blocking=True)
        return uv_idx
    def _get_normal(self, cluster_idx: int) -> torch.Tensor:
        nrm = self.ClusterNormals[cluster_idx]
        nrm = nrm.to("cuda", non_blocking=True).detach().requires_grad_(nrm.requires_grad)
        return nrm
    def _get_tangent(self, cluster_idx: int) -> torch.Tensor:
        tng = self.ClusterTangents[cluster_idx]
        tng = tng.to("cuda", non_blocking=True).detach().requires_grad_(tng.requires_grad)
        return tng
