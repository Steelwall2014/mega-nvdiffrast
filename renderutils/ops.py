# Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import slangpy
import os
import sys
import torch
import torch.utils.cpp_extension
from .loss import *
import nvdiffrast.torch as dr

#----------------------------------------------------------------------------
# C++/Cuda plugin compiler/loader.

_cached_plugin = None
def _get_plugin():
    # Return cached plugin if already loaded.
    global _cached_plugin
    if _cached_plugin is not None:
        return _cached_plugin

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == 'nt':
        def find_cl_path():
            import glob
            for edition in ['Enterprise', 'Professional', 'BuildTools', 'Community']:
                paths = sorted(glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64" % edition), reverse=True)
                if paths:
                    return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ['PATH'] += ';' + cl_path

    # Compiler options.
    opts = ['-DNVDR_TORCH']

    # Linker options.
    if os.name == 'posix':
        ldflags = ['-lnvrtc']
    elif os.name == 'nt':
        ldflags = ['cuda.lib', 'advapi32.lib', 'nvrtc.lib']

    # List of sources.
    source_files = [
        'c_src/normal_tangent.cu',
        "c_src/virtual_shadow_mapping.cu",
        'c_src/torch_bindings.cpp',
        'c_src/common.cpp'
    ]

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''

    # Try to detect if a stray lock file is left in cache directory and show a warning. This sometimes happens on Windows if the build is interrupted at just the right moment.
    try:
        lock_fn = os.path.join(torch.utils.cpp_extension._get_build_directory('renderutils_plugin', False), 'lock')
        if os.path.exists(lock_fn):
            print("Warning: Lock file exists in build directory: '%s'" % lock_fn)
    except:
        pass

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(name='renderutils_plugin', sources=source_paths, extra_cflags=opts,
         extra_cuda_cflags=opts, extra_ldflags=ldflags, with_cuda=True, verbose=True)

    # Import, cache, and return the compiled module.
    import renderutils_plugin
    _cached_plugin = renderutils_plugin
    return _cached_plugin

from .loss import *
from .bsdf import *

#----------------------------------------------------------------------------
# Shading normal setup (bump mapping + bent normals)

slang_normal = slangpy.loadModule(os.path.join(os.path.dirname(__file__), "normal.slang"))

class _prepare_shading_normal_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl):
        ctx.two_sided_shading, ctx.opengl = two_sided_shading, opengl
        out = slang_normal.prepare_shading_normal_fwd(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
        ctx.save_for_backward(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm)
        return out

    @staticmethod
    def backward(ctx, dout):
        pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm = ctx.saved_variables
        dout = dout.contiguous()
        res = slang_normal.prepare_shading_normal_bwd(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, dout, ctx.two_sided_shading, ctx.opengl) + (None, None, None)
        return res

def prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading=True, opengl=True, use_python=False):
    '''Takes care of all corner cases and produces a final normal used for shading:
        - Constructs tangent space
        - Flips normal direction based on geometric normal for two sided Shading
        - Perturbs shading normal by normal map
        - Bends backfacing normals towards the camera to avoid shading artifacts

        All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        pos: World space g-buffer position.
        view_pos: Camera position in world space (typically using broadcasting).
        perturbed_nrm: Trangent-space normal perturbation from normal map lookup.
        smooth_nrm: Interpolated vertex normals.
        smooth_tng: Interpolated vertex tangents.
        geom_nrm: Geometric (face) normals.
        two_sided_shading: Use one/two sided shading
        opengl: Use OpenGL/DirectX normal map conventions 
        use_python: Use PyTorch implementation (for validation)
    Returns:
        Final shading normal
    '''    

    if perturbed_nrm is None:
        perturbed_nrm = torch.tensor([0, 0, 1], dtype=torch.float32, device='cuda', requires_grad=False)[None, None, None, ...]
    
    if use_python:
        out = bsdf_prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
    else:
        out = _prepare_shading_normal_func.apply(pos, view_pos, perturbed_nrm.contiguous(), smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of prepare_shading_normal contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# Slang pbr shader

slang_module = slangpy.loadModule(os.path.join(os.path.dirname(__file__), "pbr.slang")) 

class _pbr_bsdf_slang_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kd, arm, pos, nrm, view_pos, light_pos, min_roughness):
        ctx.save_for_backward(kd, arm, pos, nrm, view_pos, light_pos)
        ctx.min_roughness = min_roughness
        out = slang_module.pbr_fwd(kd, arm, pos, nrm, view_pos, light_pos, min_roughness)
        return out

    @staticmethod
    def backward(ctx, dout):
        kd, arm, pos, nrm, view_pos, light_pos = ctx.saved_variables
        res = slang_module.pbr_bwd(kd, arm, pos, nrm, view_pos, light_pos, ctx.min_roughness, dout) + (None, None)
        return res

def pbr_bsdf(kd, arm, pos, nrm, view_pos, light_pos, min_roughness=0.08, use_python=False):
    '''Physically-based bsdf, both diffuse & specular lobes
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        kd: Diffuse albedo.
        arm: Specular parameters (attenuation, linear roughness, metalness).
        pos: World space position.
        nrm: World space shading normal.
        view_pos: Camera position in world space, typically using broadcasting.
        light_pos: Light position in world space, typically using broadcasting.
        min_roughness: Scalar roughness clamping threshold
    
    Returns:
        Shaded color.
    '''    

    if use_python:
        out = bsdf_pbr(kd, arm, pos, nrm, view_pos, light_pos, min_roughness)
    else:
        out = _pbr_bsdf_slang_func.apply(kd, arm, pos, nrm, view_pos, light_pos, min_roughness)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of pbr_bsdf contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# cubemap filter with filtering across edges

slang_cubemap = slangpy.loadModule(os.path.join(os.path.dirname(__file__), "cubemap.slang"))

class _diffuse_cubemap_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        out = slang_cubemap.diffuse_cubemap_fwd(cubemap)
        ctx.save_for_backward(cubemap)
        return out

    @staticmethod
    def backward(ctx, dout):
        cubemap, = ctx.saved_variables
        cubemap_grad = slang_cubemap.diffuse_cubemap_bwd(cubemap, dout)
        return cubemap_grad, None

def diffuse_cubemap(cubemap, use_python=False):
    if use_python:
        assert False
    else:
        out = _diffuse_cubemap_func.apply(cubemap)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of diffuse_cubemap contains inf or NaN"
    return out

class _specular_cubemap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap, roughness, costheta_cutoff, bounds):
        out = slang_cubemap.specular_cubemap_fwd(cubemap, bounds, roughness, costheta_cutoff)
        ctx.save_for_backward(cubemap, bounds)
        ctx.roughness, ctx.theta_cutoff = roughness, costheta_cutoff
        return out

    @staticmethod
    def backward(ctx, dout):
        cubemap, bounds = ctx.saved_variables
        cubemap_grad = slang_cubemap.specular_cubemap_bwd(cubemap, bounds, ctx.roughness, ctx.theta_cutoff, dout)
        return cubemap_grad, None, None, None

# Compute the bounds of the GGX NDF lobe to retain "cutoff" percent of the energy
def __ndfBounds(res, roughness, cutoff):
    def ndfGGX(alphaSqr, costheta):
        costheta = np.clip(costheta, 0.0, 1.0)
        d = (costheta * alphaSqr - costheta) * costheta + 1.0
        return alphaSqr / (d * d * np.pi)

    # Sample out cutoff angle
    nSamples = 1000000
    costheta = np.cos(np.linspace(0, np.pi/2.0, nSamples))
    D = np.cumsum(ndfGGX(roughness**4, costheta))
    idx = np.argmax(D >= D[..., -1] * cutoff)

    # Brute force compute lookup table with bounds
    bounds = slang_cubemap.specular_bounds(res, costheta[idx])

    return costheta[idx], bounds
__ndfBoundsDict = {}

def specular_cubemap(cubemap, roughness, cutoff=0.99, use_python=False):
    assert cubemap.shape[0] == 6 and cubemap.shape[1] == cubemap.shape[2], "Bad shape for cubemap tensor: %s" % str(cubemap.shape)

    if use_python:
        assert False
    else:
        key = (cubemap.shape[1], roughness, cutoff)
        if key not in __ndfBoundsDict:
            __ndfBoundsDict[key] = __ndfBounds(*key)
        out = _specular_cubemap.apply(cubemap, roughness, *__ndfBoundsDict[key])
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of specular_cubemap contains inf or NaN"
    return out[..., 0:3] / out[..., 3:]

#----------------------------------------------------------------------------
# Fast image loss function

def strToLoss(s):
    if s == "mse":
        return 1 #LOSS_MSE;
    elif s == "relmse":
        return 2 #LOSS_RELMSE;
    elif s == "smape":
        return 3 #LOSS_SMAPE;
    else:
        return 0 #LOSS_L1;

def strToTonemapper(s):
    if s == "log_srgb":
        return 1 # TONEMAPPER_LOG_SRGB
    else:
        return 0 # TONEMAPPER_NONE

slang_loss = slangpy.loadModule(os.path.join(os.path.dirname(__file__), "loss.slang"))

class _image_loss_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img, target, loss, tonemapper):
        ctx.loss, ctx.tonemapper = loss, tonemapper
        ctx.save_for_backward(img, target)
        out = slang_loss.loss_fwd(img, target, strToLoss(loss), strToTonemapper(tonemapper))
        return out

    @staticmethod
    def backward(ctx, dout):
        img, target = ctx.saved_variables
        return slang_loss.loss_bwd(img, target, strToLoss(ctx.loss), strToTonemapper(ctx.tonemapper), dout.contiguous()) + (None, None, None)

def image_loss(img, target, loss='l1', tonemapper='none', use_python=False):
    '''Compute HDR image loss. Combines tonemapping and loss into a single kernel for better perf.
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        img: Input image.
        target: Target (reference) image. 
        loss: Type of loss. Valid options are ['l1', 'mse', 'smape', 'relmse']
        tonemapper: Tonemapping operations. Valid options are ['none', 'log_srgb']
        use_python: Use PyTorch implementation (for validation)

    Returns:
        Image space loss (scalar value).
    '''
    if use_python:
        out = image_loss_fn(img, target, loss, tonemapper)
    else:
        out = _image_loss_func.apply(img, target, loss, tonemapper)
        out = torch.sum(out) / (img.shape[0]*img.shape[1]*img.shape[2])

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of image_loss contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# Transform points function

slang_mesh = slangpy.loadModule(os.path.join(os.path.dirname(__file__), "mesh.slang"))
class _xfm_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, matrix, isPoints):
        ctx.save_for_backward(points, matrix)
        ctx.isPoints = isPoints
        return slang_mesh.xfm_fwd(points, matrix, isPoints)

    @staticmethod
    def backward(ctx, dout):
        points, matrix = ctx.saved_variables
        return (slang_mesh.xfm_bwd(points, matrix, dout, ctx.isPoints),) + (None, None, None)

def xfm_points(points, matrix, use_python=False):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''    
    if use_python:
        out = torch.matmul(torch.nn.functional.pad(points, pad=(0,1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))
    else:
        out = _xfm_func.apply(points, matrix, True)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out

def xfm_vectors(vectors, matrix, use_python=False):
    '''Transform vectors.
    Args:
        vectors: Tensor containing 3D vectors with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)

    Returns:
        Transformed vectors in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''    

    if use_python:
        out = torch.matmul(torch.nn.functional.pad(vectors, pad=(0,1), mode='constant', value=0.0), torch.transpose(matrix, 1, 2))[..., 0:3].contiguous()
    else:
        out = _xfm_func.apply(vectors, matrix, False)[..., 0:3].contiguous()

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_vectors contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# Virtual geometry calculate normal and tangent

class _calculate_normal_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, MatchingVertices, ClusterIndexes, *ClusterPositions):
        ClusterNormals = _get_plugin().calculate_normal_fwd(ClusterPositions, ClusterIndexes, MatchingVertices)
        ctx.save_for_backward(*ClusterPositions)
        ctx.saved_misc = MatchingVertices, ClusterIndexes
        return tuple(ClusterNormals)

    @staticmethod
    def backward(ctx, *ClusterNormalGrads):
        *ClusterPositions, = ctx.saved_variables
        MatchingVertices, ClusterIndexes = ctx.saved_misc
        ClusterPositionGrads = _get_plugin().calculate_normal_bwd(ClusterPositions, ClusterIndexes, MatchingVertices, ClusterNormalGrads)
        return (None, None) + tuple(ClusterPositionGrads)
    
def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)
def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN
def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def calculate_normal(ClusterPositions: list[torch.Tensor], ClusterIndexes: list[torch.Tensor], MatchingVertices: list[list[tuple[int, int]]]) -> tuple[torch.Tensor]:
    '''Calculate normal and tangent for virtual geometry.
    Args:
        ClusterPositions: List of tensors containing 3D positions with shape [num_vertices, 3]
        ClusterIndexes: List of tensors containing cluster indexes with shape [num_triangles, 3]
        MatchingVertices: List of list of tuples containing (cluster index, vertex index) pairs
    Returns:
        ClusterNormals: List of tensors containing normals with shape [num_vertices, 3]
    '''
    ClusterNormals = list(_calculate_normal_func.apply(MatchingVertices, ClusterIndexes, *ClusterPositions))
    for i in range(len(ClusterNormals)):
        ClusterNormals[i] = safe_normalize(ClusterNormals[i])
    return ClusterNormals

def calculate_normal_fwd(ClusterPositions: list[torch.Tensor], ClusterIndexes: list[torch.Tensor]) -> list[torch.Tensor]:
    ClusterNormals = _get_plugin().calculate_normal_fwd(ClusterPositions, ClusterIndexes)
    return list(ClusterNormals)
def calculate_normal_bwd(ClusterPositions: list[torch.Tensor], ClusterIndexes: list[torch.Tensor], ClusterNormalGrads: list[torch.Tensor]) -> list[torch.Tensor]:
    ClusterPositionGrads = _get_plugin().calculate_normal_bwd(ClusterPositions, ClusterIndexes, ClusterNormalGrads)
    return list(ClusterPositionGrads)

class _calculate_tangent_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, MatchingVertices, ClusterIndexes, *ClusterPositionTexCoords):
        num_clusters = len(ClusterPositionTexCoords) // 2
        ClusterPositions = ClusterPositionTexCoords[0:num_clusters]
        ClusterTexCoords = ClusterPositionTexCoords[num_clusters:]
        ClusterTangents = _get_plugin().calculate_tangent_fwd(ClusterPositions, ClusterTexCoords, ClusterIndexes, MatchingVertices)
        ctx.save_for_backward(*ClusterPositionTexCoords)
        ctx.saved_misc = MatchingVertices, ClusterIndexes
        return tuple(ClusterTangents)

    @staticmethod
    def backward(ctx, *ClusterTangentGrads):
        *ClusterPositionTexCoords, = ctx.saved_variables
        MatchingVertices, ClusterIndexes = ctx.saved_misc
        num_clusters = len(ClusterPositionTexCoords) // 2
        ClusterPositions = ClusterPositionTexCoords[0:num_clusters]
        ClusterTexCoords = ClusterPositionTexCoords[num_clusters:]
        ClusterPositionGrads, ClusterTexCoordGrads = _get_plugin().calculate_tangent_bwd(ClusterPositions, ClusterTexCoords, ClusterIndexes, MatchingVertices, ClusterTangentGrads)
        return (None, None) + tuple(list(ClusterPositionGrads) + list(ClusterTexCoordGrads))
    
def calculate_tangent(ClusterPositions: list[torch.Tensor], ClusterTexCoords: list[torch.Tensor], ClusterNormals: list[torch.Tensor], ClusterIndexes: list[torch.Tensor], MatchingVertices: list[list[tuple[int, int]]]) -> tuple[torch.Tensor]:
    '''Calculate tangent for virtual geometry.
    Args:
        ClusterPositions: List of tensors containing 3D positions with shape [num_vertices, 3]
        ClusterTexCoords: List of tensors containing 2D texture coordinates with shape [num_vertices, 2]
        ClusterIndexes: List of tensors containing cluster indexes with shape [num_triangles, 3]
        MatchingVertices: List of list of tuples containing (cluster index, vertex index) pairs
    Returns:
        ClusterTangents: List of tensors containing tangents with shape [num_vertices, 3]
    '''
    ClusterTangents = list(_calculate_tangent_func.apply(MatchingVertices, ClusterIndexes, *ClusterPositions, *ClusterTexCoords))
    for i in range(len(ClusterTangents)):
        ClusterTangents[i] = safe_normalize(ClusterTangents[i])
        ClusterTangents[i] = safe_normalize(ClusterTangents[i] - dot(ClusterTangents[i], ClusterNormals[i]) * ClusterNormals[i])
    return ClusterTangents

def calculate_tangent_fwd(ClusterPositions: list[torch.Tensor], ClusterTexCoords: list[torch.Tensor], ClusterPosIndexes: list[torch.Tensor], ClusterUVIndexes: list[torch.Tensor]) -> tuple[torch.Tensor]:
    ClusterTangents = _get_plugin().calculate_tangent_fwd(ClusterPositions, ClusterTexCoords, ClusterPosIndexes, ClusterUVIndexes)
    return ClusterTangents
def calculate_tangent_bwd(ClusterPositions: list[torch.Tensor], ClusterTexCoords: list[torch.Tensor], ClusterPosIndexes: list[torch.Tensor], ClusterUVIndexes: list[torch.Tensor], ClusterTangentGrads: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    ClusterPositionGrads, ClusterTexCoordGrads = _get_plugin().calculate_tangent_bwd(ClusterPositions, ClusterTexCoords, ClusterPosIndexes, ClusterUVIndexes, ClusterTangentGrads)
    return ClusterPositionGrads, ClusterTexCoordGrads


def virtual_shadow_map_feedback(camera_pos, gb_pos, shadow_map_uv, vsm_height, vsm_width, filter_mode="linear", page_size_x=256, page_size_y=256, max_mip_level=None, mask=None, first_level_extent=10):
    if max_mip_level is None:
        max_mip_level = -1
    else:
        max_mip_level = int(max_mip_level)
        assert max_mip_level >= 0
        
    if mask is None:
        mask = torch.tensor([])

    if filter_mode == "nearest":
        filter_mode = 0
    elif filter_mode == "linear":
        filter_mode = 1
    else:
        raise ValueError("filter_mode must be 'nearest' or 'linear'")
    
    return _get_plugin().virtual_shadow_map_feedback(camera_pos, gb_pos, shadow_map_uv, filter_mode, vsm_height, vsm_width, page_size_x, page_size_y, max_mip_level, mask, first_level_extent)

def async_add_(input: list[torch.Tensor], other: list[torch.Tensor]):
    return _get_plugin().async_add_(input, other)

def async_copy_(input: list[torch.Tensor], other: list[torch.Tensor]):
    return _get_plugin().async_copy_(input, other)

from torch.optim.optimizer import _get_value

@torch.no_grad()
def async_multi_tensor_adam(
        params: list[torch.Tensor], 
        grads: list[torch.Tensor], 
        exp_avgs: list[torch.Tensor], 
        exp_avg_sqs: list[torch.Tensor], 
        steps: list[torch.Tensor], 
        beta1: float, beta2: float, lr: float, eps=1e-8):
    for i in range(len(params)):
        assert params[i].is_cpu
        assert grads[i].is_cpu
        assert exp_avgs[i].is_cpu
        assert exp_avg_sqs[i].is_cpu
    steps_i = []
    for step_t in steps:
        step_t.add_(1)
        step = _get_value(step_t)
        steps_i.append(int(step))
    return _get_plugin().async_multi_tensor_adam(
        params, 
        grads, 
        exp_avgs, 
        exp_avg_sqs, 
        steps_i, 
        beta1, beta2, lr, eps)

def async_to_cpu(input: list[torch.Tensor], pin_memory=False):
    return _get_plugin().async_to_cpu(input, pin_memory)
