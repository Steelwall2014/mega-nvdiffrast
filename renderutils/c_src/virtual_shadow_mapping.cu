#include <cuda.h>
#include "common.h"
#include "virtual_shadow_mapping.h"

#define mipLevelSize(p, i) make_int2(((p).vsmWidth >> (i)) > 1 ? ((p).vsmWidth >> (i)) : 1, ((p).vsmHeight >> (i)) > 1 ? ((p).vsmHeight >> (i)) : 1)

static __device__ __forceinline__ float length(float3 v)
{
    return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

static __device__ __forceinline__ int2 indexTextureNearest_vsm(const VirtualShadowMapFeedbackKernalParams& p, float3 uv)
{
    int w = p.vsmWidth;
    int h = p.vsmHeight;
    float u = uv.x;
    float v = uv.y;
    int mipmap_level = int(uv.z);
    w = w >> mipmap_level;
    h = h >> mipmap_level;

    u = u * (float)w;
    v = v * (float)h;

    int iu = __float2int_rd(u);
    int iv = __float2int_rd(v);

    // In zero boundary mode, return texture address -1.
    if (iu < 0 || iu >= w || iv < 0 || iv >= h)
        return make_int2(0, -1);

    // Otherwise clamp and calculate the coordinate properly.
    iu = min(max(iu, 0), w-1);
    iv = min(max(iv, 0), h-1);

    // Because sometimes the width and height may be smaller than the page size,
    // so we need to decrease the page_size_x and page_size_y to width and height
    int page_size_x = w >= p.page_size_x ? p.page_size_x : w;
    int page_size_y = h >= p.page_size_y ? p.page_size_y : h;
    int pix = iu / page_size_x;     // page index x
    int piy = iv / page_size_y;     // page index y

    iu = iu % page_size_x;
    iv = iv % page_size_y;

    int page_num_x = w / page_size_x;
    int tc = iu + page_size_x * iv;
    int indexPage = pix + piy * page_num_x;
    return make_int2(indexPage, tc);
}

static __device__ __forceinline__ float2 indexTextureLinear_vsm(const VirtualShadowMapFeedbackKernalParams& p, float2 uv/*, int tz*/, int4& indexPageOut, int4& tcOut, int level)
{
    // Mip level size.
    int2 sz = mipLevelSize(p, level);
    int w = sz.x;
    int h = sz.y;

    // Compute texture-space u, v.
    float u = uv.x;
    float v = uv.y;
    bool clampU = false;
    bool clampV = false;

    // Move to texel space.
    u = u * (float)w - 0.5f;
    v = v * (float)h - 0.5f;

    // Compute texel coordinates and weights.
    int iu0 = __float2int_rd(u);
    int iv0 = __float2int_rd(v);
    int iu1 = iu0 + (clampU ? 0 : 1); // Ensure zero u/v gradients with clamped.
    int iv1 = iv0 + (clampV ? 0 : 1);
    u -= (float)iu0;
    v -= (float)iv0;

    bool iu0_out = (iu0 < 0 || iu0 >= w);
    bool iu1_out = (iu1 < 0 || iu1 >= w);
    bool iv0_out = (iv0 < 0 || iv0 >= h);
    bool iv1_out = (iv1 < 0 || iv1 >= h);
    if (iu0_out || iv0_out) tcOut.x = -1;
    if (iu1_out || iv0_out) tcOut.y = -1;
    if (iu0_out || iv1_out) tcOut.z = -1;
    if (iu1_out || iv1_out) tcOut.w = -1;
    if (iu0_out || iu1_out || iv0_out || iv1_out) 
        return make_float2(0, 0);

    // Because sometimes the width and height may be smaller than the page size,
    // so we need to decrease the page_size_x and page_size_y to width and height
    int page_size_x = w >= p.page_size_x ? p.page_size_x : w;
    int page_size_y = h >= p.page_size_y ? p.page_size_y : h;
    int page_num_x = w / page_size_x;

    int ipx0 = iu0 / page_size_x;
    int ipx1 = iu1 / page_size_x;
    int ipy0 = iv0 / page_size_y;
    int ipy1 = iv1 / page_size_y;

    iu0 = iu0 % page_size_x;
    iu1 = iu1 % page_size_x;
    iv0 = iv0 % page_size_y;
    iv1 = iv1 % page_size_y;

    // Coordinates with tz folded in.
    int iu0z = iu0/* + tz * p.page_size_x * p.page_size_y*/;
    int iu1z = iu1/* + tz * p.page_size_x * p.page_size_y*/;
    tcOut.x = iu0z + page_size_x * iv0;
    tcOut.y = iu1z + page_size_x * iv0;
    tcOut.z = iu0z + page_size_x * iv1;
    tcOut.w = iu1z + page_size_x * iv1;

    indexPageOut.x = ipx0 + ipy0 * page_num_x;
    indexPageOut.y = ipx1 + ipy0 * page_num_x;
    indexPageOut.z = ipx0 + ipy1 * page_num_x;
    indexPageOut.w = ipx1 + ipy1 * page_num_x;

    // All done.
    return make_float2(u, v);
}

// VSM feedback for *single* light and *multiple* cameras
template <int FILTER_MODE>
static __device__ __device__ void VirtualShadowMappingFeedbackTemplate(const VirtualShadowMapFeedbackKernalParams p)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.imgWidth || py >= p.imgHeight || pz >= p.n)
        return;

    // Pixel index.
    int pidx = px + p.imgWidth * (py + p.imgHeight * pz);

    if (p.mask && p.mask[pidx] == false)
        return;

    float3 position = ((const float3*)p.gb_pos)[pidx];  // world position
    float3 camera_pos = ((const float3*)p.camera_pos)[pz];
    float distance = length(position - camera_pos);
    int mipmap_level = int(floor(log2f(distance / p.first_level_extent))) + 1;
    mipmap_level = min(mipmap_level, p.max_mipmap_level);
    mipmap_level = max(mipmap_level, 0);

    float2 uv = ((float2*)p.shadow_map_uv)[pidx];
    p.vsm_mip_levels[pidx] = mipmap_level;

    bool* feedback_mipmap = p.feedback[mipmap_level];
    int num_pages = p.num_pages[mipmap_level];
    int minibatch_offset = num_pages * pz;
    bool* pOut = feedback_mipmap + minibatch_offset;

    if (FILTER_MODE == VSM_FILTER_MODE_NEAREST)
    {
        int2 pi_tc = indexTextureNearest_vsm(p, make_float3(uv, mipmap_level));
        if (pi_tc.y != -1)
            pOut[pi_tc.x] = true; 
        return;
    }

    int4 tc = make_int4(0, 0, 0, 0);
    int4 pi = make_int4(0, 0, 0, 0);    // page index
    float2 uv_ = indexTextureLinear_vsm(p, uv, pi, tc, mipmap_level);
    if (tc.x == -1 || tc.y == -1 || tc.z == -1 || tc.w == -1)   // out of boundary
        return;
    pOut[pi.x] = true;
    pOut[pi.y] = true;
    pOut[pi.z] = true;
    pOut[pi.w] = true;
}

__global__ void VirtualShadowMappingFeedbackNearest(const VirtualShadowMapFeedbackKernalParams p)
{
    VirtualShadowMappingFeedbackTemplate<VSM_FILTER_MODE_NEAREST>(p);
}

__global__ void VirtualShadowMappingFeedbackLinear(const VirtualShadowMapFeedbackKernalParams p)
{
    VirtualShadowMappingFeedbackTemplate<VSM_FILTER_MODE_LINEAR>(p);
}