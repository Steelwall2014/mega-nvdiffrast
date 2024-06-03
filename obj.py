import io
import os
import time
import numpy as np
import torch
import nvdiffrast.torch as dr
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = 268435456

from modules.material import MaterialModule
from modules.mesh import GeometryModule, IGeometryModule, VirtualGeometryModule
from modules.texture import TextureModule, VirtualTextureModule
import util

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

def _load_obj_mesh_to_module_vg(filename, vertices, texcoords, faces, tfaces, tri_material_idx, max_partition_size=8192, lru_cache_max_size=200):

    vg_cache_path = f"./cache/{filename}_{max_partition_size}.pt"
    if os.path.exists(vg_cache_path):
        ClusterOldTriIdx, cluster_positions, cluster_pos_indices, pos_new_to_old, cluster_texcoords, cluster_uv_indices, uv_new_to_old, shared_verts, shared_verts_offsets = torch.load(vg_cache_path, map_location="cpu")
    else:
        ClusterOldTriIdx = dr.virtual_geometry_construct(vertices, faces, max_partition_size=max_partition_size)
        cluster_positions, cluster_pos_indices, pos_new_to_old = construct_clusters(ClusterOldTriIdx, vertices, faces)
        cluster_texcoords, cluster_uv_indices, uv_new_to_old = construct_clusters(ClusterOldTriIdx, texcoords, tfaces)
        if len(ClusterOldTriIdx) == 1:
            shared_verts, shared_verts_offsets = torch.tensor([], dtype=torch.int32), torch.tensor([0], dtype=torch.int32)
        else:
            shared_verts, shared_verts_offsets = dr.virtual_geometry_shared_vertices(cluster_positions)
        torch.save((ClusterOldTriIdx, cluster_positions, cluster_pos_indices, pos_new_to_old, cluster_texcoords, cluster_uv_indices, uv_new_to_old, shared_verts, shared_verts_offsets), vg_cache_path)

    cluster_tri_mat_idx = []
    for i, tri_idx in enumerate(ClusterOldTriIdx):
        cluster_tri_mat_idx.append(tri_material_idx[tri_idx])

    return VirtualGeometryModule(filename, 
                                 cluster_positions=cluster_positions, 
                                 cluster_pos_indices=cluster_pos_indices, 
                                 pos_new_to_old=pos_new_to_old, 
                                 cluster_texcoords=cluster_texcoords, 
                                 cluster_uv_indices=cluster_uv_indices, 
                                 uv_new_to_old=uv_new_to_old,
                                 shared_verts=shared_verts, 
                                 shared_verts_offsets=shared_verts_offsets, 
                                 initial_positions=vertices,
                                 initial_pos_indices=faces,
                                 initial_texcoords=texcoords,
                                 initial_uv_indices=tfaces,
                                 lru_cache_max_size=lru_cache_max_size)

def _load_obj_mesh_to_module(filename, vertices, texcoords, faces, tfaces, tri_material_idx) -> GeometryModule:
    return GeometryModule(filename, vertices, texcoords, faces, tfaces)

def load_obj_mesh_to_module(fp, use_vg=True, max_partition_size=8192, lru_cache_max_size=200) -> IGeometryModule:
    # TODO: 现在暂时只读取只有一个material的obj文件
    with open(fp, 'r') as f:
        lines = f.readlines()

    filename = os.path.basename(fp)
    cache_path = f"./cache/{filename}.pt"
    if not os.path.exists("./cache"):
        os.mkdir("./cache")

    if os.path.exists(cache_path):
        # print("Cache found")
        vertices, texcoords, faces, tfaces, tri_material_idx = torch.load(cache_path, map_location="cpu")
    else:
        # python解析obj文件极慢
        # TODO: 用C++
        vertices, texcoords  = [], []
        for i, line in enumerate(lines):
            line_split = line.split()
            if len(line_split) == 0:
                continue
            
            prefix = line_split[0].lower()
            if prefix == 'v':
                vertices.append([float(v) for v in line_split[1:]])
            elif prefix == 'vt':
                val = [float(v) for v in line_split[1:]]
                texcoords.append([val[0], 1.0 - val[1]])

        faces, tfaces = [], []
        for i, line in enumerate(lines):
            line_split = line.split()
            if len(line_split) == 0:
                continue

            prefix = line_split[0].lower()
            if prefix == 'f': # Parse face
                vs = line_split[1:]
                nv = len(vs)
                vv = vs[0].split('/')
                v0 = int(vv[0]) - 1
                t0 = int(vv[1]) - 1 if vv[1] != "" else -1
                for i in range(nv - 2): # Triangulate polygons
                    vv = vs[i + 1].split('/')
                    v1 = int(vv[0]) - 1
                    t1 = int(vv[1]) - 1 if vv[1] != "" else -1
                    vv = vs[i + 2].split('/')
                    v2 = int(vv[0]) - 1
                    t2 = int(vv[1]) - 1 if vv[1] != "" else -1
                    faces.append([v0, v1, v2])
                    tfaces.append([t0, t1, t2])
        assert len(tfaces) == len(faces)

        vertices = torch.tensor(vertices, dtype=torch.float32, device='cpu')
        texcoords = torch.tensor(texcoords, dtype=torch.float32, device='cpu')
        faces = torch.tensor(faces, dtype=torch.int32, device='cpu')
        tfaces = torch.tensor(tfaces, dtype=torch.int32, device='cpu')
        tri_material_idx = torch.zeros(faces.size(0), dtype=torch.int32, device='cpu')  # tri_material_idx是每个三角形属于哪个材质，目前没有使用，只支持单个材质
        torch.save((vertices, texcoords, faces, tfaces, tri_material_idx), cache_path)

    if use_vg:
        return _load_obj_mesh_to_module_vg(filename, vertices, texcoords, faces, tfaces, tri_material_idx, max_partition_size=max_partition_size, lru_cache_max_size=lru_cache_max_size)
    else:
        return _load_obj_mesh_to_module(filename, vertices, texcoords, faces, tfaces, tri_material_idx)

def _load_obj_materials(fp, texture_resolution, use_vt=True,
                        default_roughness=0.8,
                        default_metallic=0.0,
                        page_size_x=256,
                        page_size_y=256,
                        fp16_texture=False,
                        lru_cache_max_size=1000) -> list[MaterialModule]:
    import re
    mtl_path = os.path.dirname(fp)
    tex_resolution = texture_resolution

    # Read file
    with open(fp, 'r') as f:
        lines = f.readlines()

    # Parse materials
    materials: list[dict] = []
    for line in lines:
        split_line = re.split(' +|\t+|\n+', line.strip())
        prefix = split_line[0].lower()
        data = split_line[1:]
        if 'newmtl' in prefix:
            material = {'name' : data[0]}
            materials += [material]
        elif materials:
            if 'bsdf' in prefix or 'map_kd' in prefix or 'map_ks' in prefix or 'bump' in prefix:
                material[prefix] = data[0]
            else:
                material[prefix] = torch.tensor(tuple(float(d) for d in data), dtype=torch.float32)

    # Convert everything to textures. Our code expects 'kd' and 'ks' to be texture maps. So replace constants with 1x1 maps
    Materials = []
    for mat in materials:
        if not 'bsdf' in mat:
            mat['bsdf'] = 'pbr'

        if 'map_kd' in mat:
            path = os.path.join(mtl_path, mat['map_kd'])
            kd = torch.from_numpy(util.load_image(path))[..., :3].contiguous()
        elif hasattr(mat, 'kd'):
            kd = torch.zeros((tex_resolution, tex_resolution, 3), dtype=torch.float32)
            kd[..., :] = mat['kd']
        else:
            kd = torch.rand((tex_resolution, tex_resolution, 3), dtype=torch.float32)
        if tex_resolution is not None:
            kd = util.scale_img_hwc(kd, [tex_resolution, tex_resolution])
        kd = util.srgb_to_rgb(kd)
        
        if 'map_ks' in mat:
            path = os.path.join(mtl_path, mat['map_ks'])
            ks = torch.from_numpy(util.load_image(path))
        elif hasattr(mat, 'ks'):
            ks = torch.zeros((tex_resolution, tex_resolution, 3), dtype=torch.float32)
            ks[..., :] = mat['ks']
        else:
            ks = torch.zeros((tex_resolution, tex_resolution, 3), dtype=torch.float32)
            if isinstance(default_roughness, list):
                ks[..., 1] = torch.normal(mean=default_roughness[0], std=default_roughness[1], size=(tex_resolution, tex_resolution))
            else:
                ks[..., 1] = default_roughness
            if isinstance(default_metallic, list):
                ks[..., 2] = torch.normal(mean=default_metallic[0], std=default_metallic[1], size=(tex_resolution, tex_resolution))
            else:
                ks[..., 2] = default_metallic
        if tex_resolution is not None:
            ks = util.scale_img_hwc(ks, [tex_resolution, tex_resolution])

        if 'bump' in mat:
            path = os.path.join(mtl_path, mat['bump'])
            normal = torch.from_numpy(util.load_image(path))
            normal = normal * 2 - 1
        else:
            normal = torch.zeros((tex_resolution, tex_resolution, 3), dtype=torch.float32)
            normal[..., 2] = 1
        if tex_resolution is not None:
            normal = util.scale_img_hwc(normal, [tex_resolution, tex_resolution])

        filename = os.path.basename(fp).split(".")[0]
        TextureClass = VirtualTextureModule if use_vt else TextureModule
        BaseColor = TextureClass(kd[None, ...], min_max=[[0, 0, 0], [1, 1, 1]], name=f"{filename}_{mat['name']}_baseColor", page_size_x=page_size_x, page_size_y=page_size_y, fp16_texture=fp16_texture, lru_cache_max_size=lru_cache_max_size)
        MetallicRoughness = TextureClass(ks[None, ...], min_max=[[0, 0.08, 0], [1, 1, 1]], name=f"{filename}_{material['name']}_metallic_roughness", page_size_x=page_size_x, page_size_y=page_size_y, fp16_texture=fp16_texture, lru_cache_max_size=lru_cache_max_size)
        Normal = TextureClass(normal[None, ...], min_max=[[-1, -1, 0], [1, 1, 1]], name=f"{filename}_{material['name']}_normal", page_size_x=page_size_x, page_size_y=page_size_y, fp16_texture=fp16_texture, lru_cache_max_size=lru_cache_max_size)
        Material = MaterialModule(BaseColor, MetallicRoughness, Normal)
        Materials.append(Material)

    return Materials

def load_obj_material_to_module(fp, texture_resolution, use_vt=True,
                                default_roughness=0.8,
                                default_metallic=0.0,
                                page_size_x=256,
                                page_size_y=256,
                                fp16_texture=False,
                                lru_cache_max_size=1000) -> list[MaterialModule]:
    with open(fp, 'r') as f:
        obj_path = os.path.dirname(fp)
        lines = f.readlines()
    
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'mtllib':
                return _load_obj_materials(os.path.join(obj_path, line.split()[1]), 
                                           texture_resolution, use_vt,
                                           default_roughness, default_metallic,
                                           page_size_x, page_size_y,
                                           fp16_texture, lru_cache_max_size)