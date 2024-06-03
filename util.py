# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import logging
import math
import os
import random
import time

import psutil
import renderutils as ru
import numpy as np
import torch
import nvdiffrast.torch as dr
import imageio
import io
import zlib

#----------------------------------------------------------------------------
# Vector operations
#----------------------------------------------------------------------------

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def safe_normalize_(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:    # inplace safe_normalize
    return x.div_(length(x, eps))

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

#----------------------------------------------------------------------------
# sRGB color transforms
#----------------------------------------------------------------------------

def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)

def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _rgb_to_srgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))

def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _srgb_to_rgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

def reinhard(f: torch.Tensor) -> torch.Tensor:
    return f/(1+f)

#-----------------------------------------------------------------------------------
# Metrics (taken from jaxNerf source code, in order to replicate their measurements)
#
# https://github.com/google-research/google-research/blob/301451a62102b046bbeebff49a760ebeec9707b8/jaxnerf/nerf/utils.py#L266
#
#-----------------------------------------------------------------------------------

def mse_to_psnr(mse):
  """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
  return -10. / np.log(10.) * np.log(mse)

def psnr_to_mse(psnr):
  """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
  return np.exp(-0.1 * np.log(10.) * psnr)

#----------------------------------------------------------------------------
# Displacement texture lookup
#----------------------------------------------------------------------------

def get_miplevels(texture: np.ndarray) -> float:
    minDim = min(texture.shape[0], texture.shape[1])
    return np.floor(np.log2(minDim))

def tex_2d(tex_map : torch.Tensor, coords : torch.Tensor, filter='nearest') -> torch.Tensor:
    tex_map = tex_map[None, ...]    # Add batch dimension
    tex_map = tex_map.permute(0, 3, 1, 2) # NHWC -> NCHW
    tex = torch.nn.functional.grid_sample(tex_map, coords[None, None, ...] * 2 - 1, mode=filter, align_corners=False)
    tex = tex.permute(0, 2, 3, 1) # NCHW -> NHWC
    return tex[0, 0, ...]

#----------------------------------------------------------------------------
# Cubemap utility functions
#----------------------------------------------------------------------------

def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map, res):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device='cuda')
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                indexing='ij')
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap

def cubemap_to_latlong(cubemap, res):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                            indexing='ij')
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return dr.texture(cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]

#----------------------------------------------------------------------------
# Image scaling
#----------------------------------------------------------------------------

def scale_img_hwc(x : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhwc(x  : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def avg_pool_nhwc(x  : torch.Tensor, size) -> torch.Tensor:
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    y = torch.nn.functional.avg_pool2d(y, size)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

#----------------------------------------------------------------------------
# Behaves similar to tf.segment_sum
#----------------------------------------------------------------------------

def segment_sum(data: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    num_segments = torch.unique_consecutive(segment_ids).shape[0]

    # Repeats ids until same dimension as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:], dtype=torch.int64, device='cuda')).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    result = torch.zeros(*shape, dtype=torch.float32, device='cuda')
    result = result.scatter_add(0, segment_ids, data)
    return result

#----------------------------------------------------------------------------
# Matrix helpers.
#----------------------------------------------------------------------------

def fovx_to_fovy(fovx, aspect):
    return np.arctan(np.tan(fovx / 2) / aspect) * 2.0

def focal_length_to_fovy(focal_length, sensor_height):
    return 2 * np.arctan(0.5 * sensor_height / focal_length)

# Reworked so this matches gluPerspective / glm::perspective, using fovy
def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)
    out = torch.zeros([4, 4], dtype=torch.float32, device=device)
    out[0][0] = 1/(y*aspect)
    out[1][1] = 1/-y
    out[2][2] = -(f+n)/(f-n)
    out[2][3] = -(2*f*n)/(f-n)
    out[3][2] = -1
    return out
    # return torch.tensor([[1/(y*aspect),    0,            0,              0], 
    #                      [           0, 1/-y,            0,              0], 
    #                      [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
    #                      [           0,    0,           -1,              0]], dtype=torch.float32, device=device)

# Reworked so this matches gluPerspective / glm::perspective, using fovy
def perspective_offcenter(fovy, fraction, rx, ry, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)

    # Full frustum
    R, L = aspect*y, -aspect*y
    T, B = y, -y

    # Create a randomized sub-frustum
    width  = (R-L)*fraction
    height = (T-B)*fraction
    xstart = (R-L)*rx
    ystart = (T-B)*ry

    l = L + xstart
    r = l + width
    b = B + ystart
    t = b + height
    
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix
    out = torch.zeros([4, 4], dtype=torch.float32, device=device)
    out[0][0] = 2/(r-l)
    out[0][2] = (r+l)/(r-l)
    out[1][1] = -2/(t-b)
    out[1][2] = (t+b)/(t-b)
    out[2][2] = -(f+n)/(f-n)
    out[2][3] = -(2*f*n)/(f-n)
    out[3][2] = -1
    return out
    # return torch.tensor([[2/(r-l),        0,  (r+l)/(r-l),              0], 
    #                      [      0, -2/(t-b),  (t+b)/(t-b),              0], 
    #                      [      0,        0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
    #                      [      0,        0,           -1,              0]], dtype=torch.float32, device=device)

def ortho(l, r, b, t, n, f, device=None):
    out = torch.eye(4, dtype=torch.float32, device=device)
    out[0][0] = 2/(r-l)
    out[1][1] = 2/(t-b)
    out[2][2] = -2/(f-n)
    out[0][3] = -(r+l)/(r-l)
    out[1][3] = -(t+b)/(t-b)
    out[2][3] = -(f+n)/(f-n)
    return out
    # return torch.tensor([[2/(r-l),        0,        0, -(r+l)/(r-l)], 
    #                      [      0,  2/(t-b),        0, -(t+b)/(t-b)], 
    #                      [      0,        0, -2/(f-n), -(f+n)/(f-n)], 
    #                      [      0,        0,        0,            1]], dtype=torch.float32, device=device)

def translate(x, y, z, device=None):
    return torch.tensor([[1, 0, 0, x], 
                         [0, 1, 0, y], 
                         [0, 0, 1, z], 
                         [0, 0, 0, 1]], dtype=torch.float32, device=device)

def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1,  0, 0, 0], 
                         [0,  c, s, 0], 
                         [0, -s, c, 0], 
                         [0,  0, 0, 1]], dtype=torch.float32, device=device)

def rotate_y(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0, s, 0], 
                         [ 0, 1, 0, 0], 
                         [-s, 0, c, 0], 
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)

def scale(s, device=None):
    return torch.tensor([[ s, 0, 0, 0], 
                         [ 0, s, 0, 0], 
                         [ 0, 0, s, 0], 
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)

def lookAt(eye, at, up):
    a = eye - at
    w = a / torch.linalg.norm(a)
    u = torch.cross(up, w)
    u = u / torch.linalg.norm(u)
    v = torch.cross(w, u)
    translate = torch.eye(4, dtype=eye.dtype, device=eye.device)
    translate[0][3] = -eye[0]
    translate[1][3] = -eye[1]
    translate[2][3] = -eye[2]
    rotate = torch.eye(4, dtype=eye.dtype, device=eye.device)
    rotate[0][0] = u[0]
    rotate[0][1] = u[1]
    rotate[0][2] = u[2]
    rotate[1][0] = v[0]
    rotate[1][1] = v[1]
    rotate[1][2] = v[2]
    rotate[2][0] = w[0]
    rotate[2][1] = w[1]
    rotate[2][2] = w[2]
    # translate = torch.tensor([[1, 0, 0, -eye[0]], 
    #                           [0, 1, 0, -eye[1]], 
    #                           [0, 0, 1, -eye[2]], 
    #                           [0, 0, 0, 1]], dtype=eye.dtype, device=eye.device)
    # rotate = torch.tensor([[u[0], u[1], u[2], 0], 
    #                        [v[0], v[1], v[2], 0], 
    #                        [w[0], w[1], w[2], 0], 
    #                        [0, 0, 0, 1]], dtype=eye.dtype, device=eye.device)
    return rotate @ translate

@torch.no_grad()
def random_rotation_translation(t, device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return torch.tensor(m, dtype=torch.float32, device=device)

@torch.no_grad()
def random_rotation(device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.array([0,0,0]).astype(np.float32)
    return torch.tensor(m, dtype=torch.float32, device=device)

#----------------------------------------------------------------------------
# Compute focal points of a set of lines using least squares. 
# handy for poorly centered datasets
#----------------------------------------------------------------------------

def lines_focal(o, d):
    d = safe_normalize(d)
    I = torch.eye(3, dtype=o.dtype, device=o.device)
    S = torch.sum(d[..., None] @ torch.transpose(d[..., None], 1, 2) - I[None, ...], dim=0)
    C = torch.sum((d[..., None] @ torch.transpose(d[..., None], 1, 2) - I[None, ...]) @ o[..., None], dim=0).squeeze(1)
    return torch.linalg.pinv(S) @ C

#----------------------------------------------------------------------------
# Cosine sample around a vector N
#----------------------------------------------------------------------------
@torch.no_grad()
def cosine_sample(N, size=None):
    # construct local frame
    N = N/torch.linalg.norm(N)

    dx0 = torch.tensor([0, N[2], -N[1]], dtype=N.dtype, device=N.device)
    dx1 = torch.tensor([-N[2], 0, N[0]], dtype=N.dtype, device=N.device)

    dx = torch.where(dot(dx0, dx0) > dot(dx1, dx1), dx0, dx1)
    #dx = dx0 if np.dot(dx0,dx0) > np.dot(dx1,dx1) else dx1
    dx = dx / torch.linalg.norm(dx)
    dy = torch.cross(N,dx)
    dy = dy / torch.linalg.norm(dy)

    # cosine sampling in local frame
    if size is None:
        phi = 2.0 * np.pi * np.random.uniform()
        s = np.random.uniform()
    else:
        phi = 2.0 * np.pi * torch.rand(*size, 1, dtype=N.dtype, device=N.device)
        s = torch.rand(*size, 1, dtype=N.dtype, device=N.device)
    costheta = np.sqrt(s)
    sintheta = np.sqrt(1.0 - s)

    # cartesian vector in local space
    x = np.cos(phi)*sintheta
    y = np.sin(phi)*sintheta
    z = costheta

    # local to world
    return dx*x + dy*y + N*z

#----------------------------------------------------------------------------
# Bilinear downsample by 2x.
#----------------------------------------------------------------------------

def bilinear_downsample(x : torch.tensor) -> torch.Tensor:
    w = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=torch.float32, device=x.device) / 64.0
    w = w.expand(x.shape[-1], 1, 4, 4) 
    x = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, padding=1, stride=2, groups=x.shape[-1])
    return x.permute(0, 2, 3, 1)

#----------------------------------------------------------------------------
# Bilinear downsample log(spp) steps
#----------------------------------------------------------------------------

def bilinear_downsample(x : torch.tensor, spp) -> torch.Tensor:
    w = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=torch.float32, device=x.device) / 64.0
    g = x.shape[-1]
    w = w.expand(g, 1, 4, 4) 
    x = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    steps = int(np.log2(spp))
    for _ in range(steps):
        xp = torch.nn.functional.pad(x, (1,1,1,1), mode='replicate')
        x = torch.nn.functional.conv2d(xp, w, padding=0, stride=2, groups=g)
    return x.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

#----------------------------------------------------------------------------
# Gaussian blur
#----------------------------------------------------------------------------

def _gaussian_blur_kernel1d(sigma: float, size: int) -> torch.Tensor:
    kernel = torch.exp(-torch.arange(-(size//2), size//2+1, dtype=torch.float32)**2 / (2 * sigma**2))
    return kernel / kernel.sum()

def _gaussian_blur_kernel2d(sigma: float, size: int) -> torch.Tensor:
    kernel1d = _gaussian_blur_kernel1d(sigma, size)
    kernel2d = kernel1d[:, None] * kernel1d[None, :]
    return kernel2d

def gaussian_blur_nhwc(x: torch.Tensor, kernel_size: int, sigma: float):
    if kernel_size % 2 == 0:
        kernel_size += 1
    num_channels = x.shape[-1]
    y = x.permute(0, 3, 1, 2)
    kernel = _gaussian_blur_kernel2d(sigma, kernel_size).to(y.device)
    kernel = kernel[None, None, ...].repeat(num_channels, 1, 1, 1)
    y = torch.nn.functional.pad(y, [kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2], mode="replicate")
    y = torch.nn.functional.conv2d(y, kernel, groups=num_channels)
    y = y.permute(0, 2, 3, 1).contiguous()
    return y

#----------------------------------------------------------------------------
# Singleton initialize GLFW
#----------------------------------------------------------------------------

_glfw_initialized = False
def init_glfw():
    global _glfw_initialized
    try:
        import glfw
        glfw.ERROR_REPORTING = 'raise'
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        test = glfw.create_window(8, 8, "Test", None, None) # Create a window and see if not initialized yet
    except glfw.GLFWError as e:
        if e.error_code == glfw.NOT_INITIALIZED:
            glfw.init()
            _glfw_initialized = True

#----------------------------------------------------------------------------
# Image display function using OpenGL.
#----------------------------------------------------------------------------

_glfw_window = None
def display_image(image, title=None):
    # Import OpenGL
    import OpenGL.GL as gl
    import glfw

    # Zoom image if requested.
    image = np.asarray(image[..., 0:3]) if image.shape[-1] == 4 else np.asarray(image)
    height, width, channels = image.shape

    # Initialize window.
    init_glfw()
    if title is None:
        title = 'Debug window'
    global _glfw_window
    if _glfw_window is None:
        glfw.default_window_hints()
        _glfw_window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(_glfw_window)
        glfw.show_window(_glfw_window)
        glfw.swap_interval(0)
    else:
        glfw.make_context_current(_glfw_window)
        glfw.set_window_title(_glfw_window, title)
        glfw.set_window_size(_glfw_window, width, height)

    # Update window.
    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glWindowPos2f(0, 0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_format = {3: gl.GL_RGB, 2: gl.GL_RG, 1: gl.GL_LUMINANCE}[channels]
    gl_dtype = {'uint8': gl.GL_UNSIGNED_BYTE, 'float32': gl.GL_FLOAT}[image.dtype.name]
    gl.glDrawPixels(width, height, gl_format, gl_dtype, image[::-1])
    glfw.swap_buffers(_glfw_window)
    if glfw.window_should_close(_glfw_window):
        return False
    return True

#----------------------------------------------------------------------------
# Image save/load helper.
#----------------------------------------------------------------------------

def save_image(fn, x : np.ndarray):
    try:
        if os.path.splitext(fn)[1] == ".png":
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8), compress_level=3) # Low compression for faster saving
        else:
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))
    except:
       print("WARNING: FAILED to save image %s" % fn)

def save_image_raw(fn, x : np.ndarray):
    try:
        imageio.imwrite(fn, x)
    except:
        print("WARNING: FAILED to save image %s" % fn)


def load_image_raw(fn) -> np.ndarray:
    return imageio.v2.imread(fn)

def load_image(fn) -> np.ndarray:
    img = load_image_raw(fn)
    if img.dtype == np.float32: # HDR image
        return img
    else: # LDR image
        return img.astype(np.float32) / 255

#----------------------------------------------------------------------------

def time_to_text(x):
    if x > 3600:
        return "%.2f h" % (x / 3600)
    elif x > 60:
        return "%.2f m" % (x / 60)
    else:
        return "%.2f s" % x

#----------------------------------------------------------------------------

def checkerboard(res, checker_size) -> np.ndarray:
    tiles_y = (res[0] + (checker_size*2) - 1) // (checker_size*2)
    tiles_x = (res[1] + (checker_size*2) - 1) // (checker_size*2)
    check = np.kron([[1, 0] * tiles_x, [0, 1] * tiles_x] * tiles_y, np.ones((checker_size, checker_size)))*0.33 + 0.33
    check = check[:res[0], :res[1]]
    return np.stack((check, check, check), axis=-1)

def quat2mat(quat, device=None):
    # from glm::mat3_cast in glm/gtc/quaterion.inl
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    qxx = x * x
    qyy = y * y
    qzz = z * z
    qxz = x * z
    qxy = x * y
    qyz = y * z
    qwx = w * x
    qwy = w * y
    qwz = w * z

    # 由于glm是列优先存储，因此所有矩阵的索引对调了
    Result = torch.eye(3, dtype=torch.float32, device=device)
    Result[0, 0] = 1 - 2 * (qyy + qzz)
    Result[1, 0] = 2 * (qxy - qwz)
    Result[2, 0] = 2 * (qxz + qwy)
    Result[0, 1] = 2 * (qxy + qwz)
    Result[1, 1] = 1 - 2 * (qxx + qzz)
    Result[2, 1] = 2 * (qyz - qwx)
    Result[0, 2] = 2 * (qxz - qwy)
    Result[1, 2] = 2 * (qyz + qwx)
    Result[2, 2] = 1 - 2 * (qxx + qyy)

    return Result

def mat2quat(m: torch.Tensor) -> torch.Tensor:
    # from glm::quat_cast in glm/gtc/quaterion.inl
    fourXSquaredMinus1 = m[0][0] - m[1][1] - m[2][2]
    fourYSquaredMinus1 = m[1][1] - m[0][0] - m[2][2]
    fourZSquaredMinus1 = m[2][2] - m[0][0] - m[1][1]
    fourWSquaredMinus1 = m[0][0] + m[1][1] + m[2][2]

    biggestIndex = 0
    fourBiggestSquaredMinus1 = fourWSquaredMinus1
    if fourXSquaredMinus1 > fourBiggestSquaredMinus1:
        fourBiggestSquaredMinus1 = fourXSquaredMinus1
        biggestIndex = 1
    if fourYSquaredMinus1 > fourBiggestSquaredMinus1:
        fourBiggestSquaredMinus1 = fourYSquaredMinus1
        biggestIndex = 2
    if fourZSquaredMinus1 > fourBiggestSquaredMinus1:
        fourBiggestSquaredMinus1 = fourZSquaredMinus1
        biggestIndex = 3

    biggestVal = math.sqrt(fourBiggestSquaredMinus1 + 1) * 0.5
    mult = 0.25 / biggestVal
    w, x, y, z = 0, 0, 0, 0
    # 由于glm是列优先存储，因此所有矩阵的索引对调了
    if biggestIndex == 0:
        w = biggestVal
        x = (m[2][1] - m[1][2]) * mult
        y = (m[0][2] - m[2][0]) * mult
        z = (m[1][0] - m[0][1]) * mult
    elif biggestIndex == 1:
        w = (m[2][1] - m[1][2]) * mult
        x = biggestVal
        y = (m[1][0] + m[0][1]) * mult
        z = (m[0][2] + m[2][0]) * mult
    elif biggestIndex == 2:
        w = (m[0][2] - m[2][0]) * mult
        x = (m[1][0] + m[0][1]) * mult
        y = biggestVal
        z = (m[2][1] + m[1][2]) * mult
    elif biggestIndex == 3:
        w = (m[1][0] - m[0][1]) * mult
        x = (m[0][2] + m[2][0]) * mult
        y = (m[2][1] + m[1][2]) * mult
        z = biggestVal
    return torch.tensor([w, x, y, z], device="cuda", dtype=torch.float32)

def lookAt_quat(camera_pos: torch.Tensor, camera_quat: torch.Tensor) -> torch.Tensor : 
    """
    camera_pos: shape [3, 1]
    camera_quat: shape [4], wxyz format
    """
    device = camera_pos.device
    R = quat2mat(camera_quat, device).T
    T = -R @ camera_pos
    RT = torch.concatenate([R, T], axis=1)
    RT = torch.concatenate([RT, torch.tensor([[0, 0, 0, 1]], device=device)], axis=0)
    return RT

def crop_image(image_height, image_width, tile_height=2048, tile_width=2048, method="random"):
    # method="random" or "grid"
    stride_x = int(tile_width * (1-0.01)) # overlap 1%
    stride_y = int(tile_height * (1-0.01)) # overlap 1%
    num_tiles_x = math.ceil((image_width-tile_width) / stride_x) + 1
    num_tiles_y = math.ceil((image_height-tile_height) / stride_y) + 1
    slices = []
    gap_height = int(0.01 * image_height)
    gap_width = int(0.01 * image_width)
    if method == "random":
        for x in range(num_tiles_x):
            for y in range(num_tiles_y):
                start_y = random.randint(gap_height, image_height-gap_height-tile_height)
                end_y = tile_height + start_y
                start_x = random.randint(gap_width, image_width-gap_width-tile_width)
                end_x = tile_width + start_x
                slices.append((start_x, end_x, start_y, end_y))
    elif method == "grid":
        for x in range(num_tiles_x):
            for y in range(num_tiles_y):
                start_y = gap_height + y*stride_y
                end_y = tile_height + start_y
                start_x = gap_width + x*stride_x
                end_x = tile_width + start_x
                if end_x > image_width-gap_width:
                    start_x -= end_x - (image_width-gap_width)
                    end_x = image_width-gap_width
                if end_y > image_height-gap_height:
                    start_y -= end_y - (image_height-gap_height)
                    end_y = image_height-gap_height
                slices.append((start_x, end_x, start_y, end_y))
    return slices

def perspective_focal(width, height, focal_35mm, device=None):
    far = 10000
    near = 0.1
    projection = torch.zeros([4, 4], dtype=torch.float32, device=device)
    n = width / 36 * focal_35mm
    r = width / 2
    l = r - width
    t = height / 2
    b = t - height
    projection[0, 0] = 2*n/(r-l)
    projection[0, 1] = 0
    projection[0, 2] = (r+l) / (r-l)
    projection[1, 1] = -2*n/(t-b)
    projection[1, 2] = (t+b) / (t-b)
    projection[2, 2] = -(far+near) / (far-near)
    projection[2, 3] = -(2*near*far) / (far-near)
    projection[3, 2] = -1
    return projection

def prepare_projections(slices, width, height, focal_35mm, device=None):
    far = 10000
    near = 0.1
    n = width / 36 * focal_35mm   # focal length in unit of pixels
    r = width / 2
    l = r - width
    t = height / 2
    b = t - height
    projections = []
    for start_x, end_x, start_y, end_y in slices:
        rr = l + end_x
        ll = l + start_x
        tt = b + end_y
        bb = b + start_y

        """
        [[2*n/(r-l),          0,  (r+l)/(r-l),              0],
         [        0, -2*n/(t-b),  (t+b)/(t-b),              0],
         [        0,          0, -(f+n)/(f-n), -(2*n*f)/(f-n)],
         [        0,          0,           -1,              0]]
        """
        projection = torch.zeros([4, 4], dtype=torch.float32, device=device)
        projection[0, 0] = 2*n/(rr-ll)
        projection[0, 1] = 0
        projection[0, 2] = (rr+ll) / (rr-ll)
        projection[1, 1] = -2*n/(tt-bb)
        projection[1, 2] = (tt+bb) / (tt-bb)
        projection[2, 2] = -(far+near) / (far-near)
        projection[2, 3] = -(2*near*far) / (far-near)
        projection[3, 2] = -1

        projections.append(projection)

    return projections
    
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _all_reduce(tensors: list[torch.Tensor]):
    if len(tensors) == 0:
        return []
    t = torch._utils._flatten_dense_tensors(tensors)
    torch.distributed.all_reduce(t)
    return torch._utils._unflatten_dense_tensors(t, tensors)

def all_reduce(tensors: list[torch.Tensor], bucket_size=128*1024*1024) -> list[torch.Tensor]:
    try:
        bucket = []
        res = []
        num_bytes = 0
        for tensor in tensors:
            bucket.append(tensor)
            num_bytes += tensor.numel() * tensor.element_size()
            if num_bytes >= bucket_size:
                res += _all_reduce(bucket)
                bucket = []
                num_bytes = 0
        res += _all_reduce(bucket)
        return res
    except:
        for tensor in tensors:
            torch.distributed.all_reduce(tensor, async_op=True)
        torch.cuda.synchronize()
        return tensors