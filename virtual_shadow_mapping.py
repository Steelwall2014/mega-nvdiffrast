# Virtual Shadow Mapping
# 使用虚拟纹理实现虚拟阴影
# feedback有两种方式，一种是根据相机到像素的距离，还有一种是让一个场景像素对应一个阴影图像素
# 前者性能好一点，后者效果好一点（其实也差不多，所以建议还是用第一种）
from collections import defaultdict
import logging
import math
from typing import Callable, Optional
import torch
from distribute import log_dist
from modules.renderer import VertexShaderOutput
import util
import nvdiffrast.torch as dr
import renderutils as ru
from enum import Enum
from timer import timers

class KernelType(Enum):
    Box      = 0
    Gaussian = 1
    Wendland = 2

def is_pow_of_two(x: int) -> bool:
    return (x & (x - 1)) == 0

def make_grid(sizes, limits=(-1, 1), device: torch.device = None):
    # Check if limits are intended for all dimensions
    if len(limits) == 2 and not hasattr(limits[0], '__len__'):
        limits = [limits]*len(sizes)

    # Flip the y-axis for images
    if len(sizes) == 2:
        limits[1] = (limits[1][1], limits[1][0])

    xs = []
    for size_x, limits_x in zip(sizes, limits):
        xs += [ torch.linspace(*limits_x, size_x, device=device) ]

    return torch.stack(torch.meshgrid(*xs[::-1], indexing='ij')[::-1], dim=-1)

def get_box_filter_2d(size: int, device: torch.device = None):
    kernel = torch.ones((1, 1, size, size), dtype=torch.float32, device=device)
    kernel /= kernel.shape[2]*kernel.shape[3]
    return kernel

def get_gaussian_filter_2d(size: int, device: torch.device = None):
    xy = make_grid((size, size), device=device)
    kernel = torch.exp(-(xy[..., 0]**2 + xy[..., 1]**2)/(2*0.1))
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, size, size)
    return kernel

def get_wendland_filter_2d(size: int, device: torch.device = None):
    xy = make_grid((size, size), device=device)
    h = 1 #math.sqrt(2)
    d = torch.linalg.norm(xy, dim=-1).clamp(max=h)
    kernel = (1-d/h)**4 * (4*d/h + 1)
    kernel /= kernel.sum()
    return kernel.view(1, 1, size, size)

def apply_filter_2d(image: torch.Tensor, kernel: torch.Tensor, padding_mode: str) -> torch.Tensor:
    """ Convolve an all channels of an image (or batch of images) with a filter kernel
    
    Args:
        image: The set of images  ((B,)H,W,C)
        kernel: The filter kernel ((1, 1,)KH,KW)
        padding_mode: Padding mode (see `torch.nn.functional.pad`)
    """

    # Convert inputs to the required shape
    is_batched = len(image.shape) == 4
    image  = image if is_batched else image[None]
    kernel = kernel if len(kernel.shape) == 4 else kernel[None, None]

    assert len(image.shape) == 4, "image must have shape [>0, >0, >0, >0]"
    assert len(kernel.shape) == 4, "kernel must have shape [>0, >0, >0, >0]"

    padding      = (kernel.shape[2]//2, kernel.shape[2]//2, kernel.shape[2]//2, kernel.shape[2]//2)
    image_padded = torch.nn.functional.pad(image.permute(0, 3, 1, 2), padding, mode=padding_mode)

    image_filtered = torch.nn.functional.conv2d(image_padded, kernel).permute(0, 2, 3, 1)

    return image_filtered if is_batched else image_filtered[0]

def transform_points(points, matrix):
    """
    Args:
        points: torch.Tensor, shape [..., 3], ...表示任意维度
        matrix: torch.Tensor, shape [4, 4]
    Returns:
        out: torch.Tensor, shape [..., 4]
    """
    points = torch.nn.functional.pad(points, (0, 1), value=1.0, mode='constant')
    out = points @ matrix.t()
    return out

def vsm_feedback_pass_distance(
        camera_pos: torch.Tensor, gb_pos: torch.Tensor, 
        light_view_matrix: torch.Tensor, half_frustum_width: float, filter_mode="linear",
        near=0.0, far=10000.0, vsm_resolution=(16384, 16384), 
        page_size_x=512, page_size_y=512, mask: Optional[torch.Tensor]=None, first_level_extent=10) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    virtual shadow map feedback for *single* light and *multiple* cameras
    使用相机到像素的距离来计算feedback
    Args:
        camera_pos: torch.Tensor, shape [minibatch_size, 3]
        gb_pos: torch.Tensor, shape [minibatch_size, height, width, 3]
        light_view_matrix: torch.Tensor, shape [4, 4]
        half_frustum_width: float, half of the width of the light's frustum
        vsm_resolution: the vsm_resolution of the virtual shadow map, [height, width]
        page_size_x, page_size_y: int, the size of a page in the virtual shadow map
        mask: torch.Tensor, mask of gb_pos, shape [minibatch_size, height, width], dtype=torch.bool
    Returns:
        vsm_uv: torch.Tensor, the uv for sampling the virtual shadow map, shape [minibatch_size, height, width, 2]
        vsm_mip_level_bias: torch.Tensor, the mip level bias of uv coordinates, shape [minibatch_size, height, width]
        feedback: list of torch.Tensor, feedback of the virtual shadow map, len(feedback) == num_mipmaps
    """

    assert len(camera_pos.shape) == 2 and camera_pos.shape[1] == 3
    assert len(gb_pos.shape) == 4 and gb_pos.shape[3] == 3
    assert gb_pos.shape[0] == camera_pos.shape[0]
    assert len(light_view_matrix.shape) == 2 and light_view_matrix.shape[0] == 4 and light_view_matrix.shape[1] == 4
    assert len(vsm_resolution) == 2 and is_pow_of_two(vsm_resolution[0]) and is_pow_of_two(vsm_resolution[1])
    assert page_size_x > 0 and page_size_y > 0
    assert vsm_resolution[0] % page_size_y == 0 and vsm_resolution[1] % page_size_x == 0

    # 把gb_pos变换到光源空间下，获取用来到vsm中采样的uv。用阴影贴图的专用feedback（距离）
    half_frustum_height = vsm_resolution[0] / vsm_resolution[1] * half_frustum_width
    light_proj_matrix = util.ortho(-half_frustum_width, half_frustum_width, -half_frustum_height, half_frustum_height, near, far, device=gb_pos.device)
    light_mvp = light_proj_matrix @ light_view_matrix
    light_clip_pos = transform_points(gb_pos, light_mvp)
    vsm_uv = light_clip_pos[..., 0:2] / light_clip_pos[..., 3:4]
    vsm_uv = (vsm_uv + 1) / 2

    max_mip_level = int(math.log2(max(vsm_resolution) / 8)) # 光栅器限制最小分辨率为8*8, 所以这里需要限制
    feedback, vsm_mip_level_bias = ru.virtual_shadow_map_feedback(camera_pos, gb_pos, vsm_uv, vsm_resolution[0], vsm_resolution[1], 
                                                                  filter_mode=filter_mode, page_size_x=page_size_x, page_size_y=page_size_y, 
                                                                  mask=mask, max_mip_level=max_mip_level, first_level_extent=first_level_extent)
    for i, feedback_mip in enumerate(feedback):
        feedback_mip = torch.any(feedback_mip, dim=0).cpu()
        feedback[i] = torch.where(feedback_mip == True)[0].tolist()
    return vsm_uv, vsm_mip_level_bias, feedback 

def vsm_feedback_pass(
        pos: torch.Tensor, rast: torch.Tensor, tri: torch.Tensor, 
        light_view_matrix: torch.Tensor, half_frustum_width: float, rast_db=None,
        near=0.0, far=10000.0, vsm_resolution=(16384, 16384), 
        page_size_x=512, page_size_y=512, mask: Optional[torch.Tensor]=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    virtual shadow map feedback for *single* light and *multiple* cameras
    让GBuffer一个像素对应一个阴影图像素来做feedback
    Args:
        camera_pos: torch.Tensor, shape [minibatch_size, 3]
        pos, rast, tri, rast_db: torch.Tensor, same as dr.interpolate
        light_view_matrix: torch.Tensor, shape [4, 4]
        half_frustum_width: float, half of the width of the light's frustum
        vsm_resolution: the vsm_resolution of the virtual shadow map, [height, width]
        page_size_x, page_size_y: int, the size of a page in the virtual shadow map
        mask: torch.Tensor, mask of gb_pos, shape [minibatch_size, height, width], dtype=torch.bool
    Returns:
        vsm_uv: torch.Tensor, the uv for sampling the virtual shadow map, shape [minibatch_size, height, width, 2]
        vsm_mip_level_bias: torch.Tensor, the mip level bias of uv coordinates, shape [minibatch_size, height, width]
        feedback: list of torch.Tensor, feedback of the virtual shadow map, len(feedback) == num_mipmaps
    """

    assert len(light_view_matrix.shape) == 2 and light_view_matrix.shape[0] == 4 and light_view_matrix.shape[1] == 4
    assert len(vsm_resolution) == 2 and is_pow_of_two(vsm_resolution[0]) and is_pow_of_two(vsm_resolution[1])
    assert page_size_x > 0 and page_size_y > 0
    assert vsm_resolution[0] % page_size_y == 0 and vsm_resolution[1] % page_size_x == 0

    # 把pos变换到光源空间下，获取用来到vsm中采样的uv。用虚拟纹理的feedback（1场景像素对应1阴影图像素）
    half_frustum_height = vsm_resolution[0] / vsm_resolution[1] * half_frustum_width
    light_proj_matrix = util.ortho(-half_frustum_width, half_frustum_width, -half_frustum_height, half_frustum_height, near, far, device=pos.device)
    light_mvp = light_proj_matrix @ light_view_matrix
    light_clip_pos = ru.xfm_points(pos[None, ...], light_mvp[None, ...], use_python=True)[0]
    vsm_uv = light_clip_pos[..., 0:2] / light_clip_pos[..., 3:4]
    vsm_uv = (vsm_uv + 1) / 2
    vsm_uv, vsm_uv_da = dr.interpolate(vsm_uv, rast, tri, rast_db, diff_attrs="all")

    max_mip_level = int(math.log2(max(vsm_resolution) / 8)) # 光栅器限制最小分辨率为8*8, 所以这里需要限制
    feedback = dr.virtual_texture_feedback(1, vsm_resolution[0], vsm_resolution[1], 1, 
                                vsm_uv, vsm_uv_da, mask=mask, 
                                filter_mode="linear-mipmap-nearest", boundary_mode="clamp", 
                                page_size_x=page_size_x, page_size_y=page_size_y, max_mip_level=max_mip_level)
    feedback = [torch.where(mipmap)[0].tolist() for mipmap in feedback]
    return vsm_uv, vsm_uv_da, feedback 

def vsm_rendering_pass(ctx, 
                       feedback: list[list[int]], 
                       light_view_matrix: torch.Tensor, 
                       half_frustum_width: float, 
                       MeshCollector: Callable[[torch.Tensor], VertexShaderOutput],
                       vsm_resolution=(16384, 16384), page_size_x=512, page_size_y=512, near=0.0, far=10000.0,
                       antialias=True, filter_kernel_type=KernelType.Box, filter_kernel_width=3):
    """
    virtual shadow map rendering for *single* light and *multiple* cameras
    Args:
        feedback: from vsm_feedback_pass() or vsm_feedback_pass_distance()
        light_view_matrix: torch.Tensor, shape [4, 4]
        half_frustum_width: float, half of the width of the light's frustum
        MeshCollector: Callable, a function to get the vertices. Some times we need to use mvp matrix to do frustum culling
        vsm_resolution: the resolution of the virtual shadow map, [height, width]
        page_size_x, page_size_y: int, the size of a page in the virtual shadow map
        near, far: float, the near and far plane of the light's frustum
        antialias: bool, whether to use antialiasing
        filter_kernel_type: KernelType, the type of the filter kernel
        filter_kernel_width: int, the width of the filter kernel
    Returns:
        m1_vsm: dict[tuple[int,int], torch.Tensor], the depth of the rendered virtual shadow map
        m2_vsm: dict[tuple[int,int], torch.Tensor], m1*m1
        The m1_vsm and m2_vsm will be used as the input of dr.virtual_texture()
    """
    
    assert len(light_view_matrix.shape) == 2 and light_view_matrix.shape[0] == 4 and light_view_matrix.shape[1] == 4
    assert len(vsm_resolution) == 2 and is_pow_of_two(vsm_resolution[0]) and is_pow_of_two(vsm_resolution[1])
    assert page_size_x > 0 and page_size_y > 0
    assert vsm_resolution[0] % page_size_y == 0 and vsm_resolution[1] % page_size_x == 0

    device = light_view_matrix.device
    empty = torch.tensor([])
    m1_vsm: dict[tuple[int,int], torch.Tensor] = {}
    m2_vsm: dict[tuple[int,int], torch.Tensor] = {}
    half_frustum_height = vsm_resolution[0] / vsm_resolution[1] * half_frustum_width
    page_size_to_page_mvps = defaultdict(list)
    for mip_level, feedback_mip in enumerate(feedback):
        w = vsm_resolution[1] >> mip_level
        h = vsm_resolution[0] >> mip_level
        page_size_x = page_size_x if w > page_size_x else w
        page_size_y = page_size_y if h > page_size_y else h
        num_pages_x = w // page_size_x
        num_pages_y = h // page_size_y
        page_width = 2 * half_frustum_width / num_pages_x       # 各个page的frustum的宽度
        page_height = 2 * half_frustum_height / num_pages_y     # 各个page的frustum的高度

        for page_idx in feedback_mip:
            page_idx_x = page_idx % num_pages_x
            page_idx_y = page_idx // num_pages_x
            offset_x = page_idx_x * page_width
            offset_y = page_idx_y * page_height
            l = -half_frustum_width + offset_x
            r = l + page_width
            b = -half_frustum_height + offset_y
            t = b + page_height
            page_proj_matrix = util.ortho(l, r, b, t, near, far, device=device)
            page_mvp = page_proj_matrix @ light_view_matrix
            page_size_to_page_mvps[(page_size_x, page_size_y)].append((page_mvp, mip_level, page_idx))
    
    # 同样分辨率的vsm page可以合并到同一个批次中
    for (page_size_x, page_size_y), values in page_size_to_page_mvps.items():
        num_pages = len(values)
        wave_length = 8
        for w in range(0, num_pages, wave_length):   # 一次最多只渲染wave_length张page，太多了会崩
            wave = values[w:w+wave_length]
            num_pages_per_mip = defaultdict(lambda: 0)
            for _, mip_level, _ in wave:
                num_pages_per_mip[mip_level] += 1
            page_mvps = torch.stack([page_mvp for page_mvp, _, _ in wave], dim=0)

            vs_out = MeshCollector(page_mvps)
            tri, pos, clip_pos, ranges = vs_out.tri_pos, vs_out.pos, vs_out.clip_pos, vs_out.ranges
            # 在光源视角光栅化一遍网格
            rast, rast_db = dr.rasterize(ctx, clip_pos, tri, [page_size_y, page_size_x], ranges=ranges)

            # 然后插值得到每个像素的在光源坐标系下的坐标
            view_pos = transform_points(pos, light_view_matrix)
            position_mv, _ = dr.interpolate(view_pos, rast, tri, rast_db)

            # 根据不同的光源类型，得到深度图
            # 深度图的单位直接就是世界坐标的单位
            # depth = torch.norm(position_mv, dim=-1, keepdim=True) # 这是对于spot light的写法 TODO: 实现一下spot light的vsm?
            depth = -position_mv[..., 2:3]   # 对于directional light应该这么写，z轴正方向朝后，所以要加个负号

            # 处理一下没有被光栅化到的像素
            mask = rast[..., 3:4] > 0
            depth = torch.where(mask, depth, far)

            # variance shadow map
            m1 = depth
            m2 = depth * depth

            # Filter shadow maps
            kernel: Optional[torch.Tensor] = None
            if filter_kernel_width > 0:
                if filter_kernel_type == KernelType.Box:
                    kernel = get_box_filter_2d(filter_kernel_width).to(device)
                elif filter_kernel_type == KernelType.Gaussian:
                    kernel = get_gaussian_filter_2d(filter_kernel_width).to(device)
                elif filter_kernel_type == KernelType.Wendland:
                    kernel = get_wendland_filter_2d(filter_kernel_width).to(device)
                else:
                    raise RuntimeError(f"Unknown smoothing kernel {filter_kernel_type}")
            
            if antialias:
                m1 = dr.antialias(m1, rast, clip_pos, tri)
                m2 = dr.antialias(m2, rast, clip_pos, tri)

            if kernel is not None:
                m1 = apply_filter_2d(m1, kernel, padding_mode='replicate')
                m2 = apply_filter_2d(m2, kernel, padding_mode='replicate')
                
            for i, (_, mip_level, page_idx) in enumerate(wave):
                m1_vsm[(mip_level, page_idx)] = m1[i:i+1, ...]
                m2_vsm[(mip_level, page_idx)] = m2[i:i+1, ...]

    return m1_vsm, m2_vsm

def chebyshev_one_sided(variance, mean, x):
    return variance / (variance + (x - mean)**2)

def compute_visibility(
        vsm_uv: torch.Tensor, depth_actual: torch.Tensor, 
        m1_vsm: dict[tuple[int,int], torch.Tensor], m2_vsm: dict[tuple[int,int], torch.Tensor], vsm_uv_da=None, vsm_mip_level_bias:torch.Tensor=None, 
        vsm_resolution=(16384, 16384), page_size_x=512, page_size_y=512, variance_min=0.0001, mask: Optional[torch.Tensor]=None):
    """
    Compute the visibility from the virtual shadow map
    Args:
        vsm_uv: torch.Tensor, the uv for sampling the virtual shadow map, shape [minibatch_size, height, width, 2]
        vsm_mip_level_bias: torch.Tensor, the mip level bias of uv coordinates, shape [minibatch_size, height, width]
        depth_actual: torch.Tensor, the depth in light space, shape [minibatch_size, height, width, 1]
        m1_vsm: dict[tuple[int,int], torch.Tensor], the depth of the rendered virtual shadow map
        m2_vsm: dict[tuple[int,int], torch.Tensor], m1*m1
        vsm_resolution: the resolution of the virtual shadow map, [height, width]
        page_size_x, page_size_y: int, the size of a page in the virtual shadow map
        variance_min: float, the minimum variance
    Returns:
        visibility: torch.Tensor, the visibility with range [0, 1], shape [minibatch_size, height, width, 1]
    """
    assert len(vsm_uv.shape) == 4 and vsm_uv.shape[3] == 2
    assert vsm_mip_level_bias is None or len(vsm_mip_level_bias.shape) == 3
    assert len(depth_actual.shape) == 4 and depth_actual.shape[3] == 1
    assert vsm_mip_level_bias is None or vsm_mip_level_bias.shape[0] == vsm_uv.shape[0]
    assert depth_actual.shape[0] == vsm_uv.shape[0]

    timers("m1 vsm sampling").start()
    m1_sampled = dr.virtual_texture(
        1, vsm_resolution[0], vsm_resolution[1], 1, 
        m1_vsm, vsm_uv, uv_da=vsm_uv_da, 
        mip_level_bias=vsm_mip_level_bias, mask=mask, 
        filter_mode="linear-mipmap-nearest", boundary_mode="clamp", # 要用mip_level_bias的话需要用linear-mipmap-xxxx
        page_size_x=page_size_x, page_size_y=page_size_y)
    timers("m1 vsm sampling").stop()
    
    timers("m2 vsm sampling").start()
    m2_sampled = dr.virtual_texture(
        1, vsm_resolution[0], vsm_resolution[1], 1, 
        m2_vsm, vsm_uv, uv_da=vsm_uv_da, 
        mip_level_bias=vsm_mip_level_bias, mask=mask, 
        filter_mode="linear-mipmap-nearest", boundary_mode="clamp", 
        page_size_x=page_size_x, page_size_y=page_size_y)
    timers("m2 vsm sampling").stop()
    
    mean_sampled     = m1_sampled
    variance_sampled = m2_sampled - m1_sampled*m1_sampled
    variance_sampled = variance_sampled.clamp(min=variance_min)
    
    # 这是vairance shadow map, 来自"Differentiable Shadow Mapping for Efficient Inverse Graphics"
    timers("visibility torch.where").start()
    visibility = torch.where(depth_actual <= mean_sampled, 
        torch.tensor(1, dtype=torch.float32, device=depth_actual.device), 
        chebyshev_one_sided(variance_sampled, mean_sampled, depth_actual)
    ).clamp(0, 1)
    timers("visibility torch.where").stop()

    return visibility
