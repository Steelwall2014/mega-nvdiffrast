#include "common.h"

struct DiffPair_float_0
{
    float primal_0;
    float differential_0;
};
struct DiffPair_float3_0
{
    float3  primal_0;
    float3  differential_0;
};
struct DiffPair_float2_0
{
    float2  primal_0;
    float2  differential_0;
};

inline __device__ float F32_min(float a, float b) { return ::fminf(a, b); }
inline __device__ float F32_max(float a, float b) { return ::fmaxf(a, b); }

__device__ void _d_max_0(DiffPair_float_0 * dpx_0, DiffPair_float_0 * dpy_0, float dOut_0)
{
    DiffPair_float_0 _S38 = *dpx_0;
    float _S39;
    if((*dpx_0).primal_0 > (*dpy_0).primal_0)
    {
        _S39 = dOut_0;
    }
    else
    {
        _S39 = 0.0f;
    }
    dpx_0->primal_0 = _S38.primal_0;
    dpx_0->differential_0 = _S39;
    DiffPair_float_0 _S40 = *dpy_0;
    if((*dpy_0).primal_0 > _S38.primal_0)
    {
        _S39 = dOut_0;
    }
    else
    {
        _S39 = 0.0f;
    }
    dpy_0->primal_0 = _S40.primal_0;
    dpy_0->differential_0 = _S39;
    return;
}

__device__ float s_primal_ctx_max_0(float _S41, float _S42)
{
    return (F32_max((_S41), (_S42)));
}

__device__ void s_bwd_prop_max_0(DiffPair_float_0 * _S43, DiffPair_float_0 * _S44, float _S45)
{
    _d_max_0(_S43, _S44, _S45);
    return;
}

__device__ void _d_min_0(DiffPair_float_0 * dpx_1, DiffPair_float_0 * dpy_1, float dOut_1)
{
    DiffPair_float_0 _S46 = *dpx_1;
    float _S47;
    if((*dpx_1).primal_0 < (*dpy_1).primal_0)
    {
        _S47 = dOut_1;
    }
    else
    {
        _S47 = 0.0f;
    }
    dpx_1->primal_0 = _S46.primal_0;
    dpx_1->differential_0 = _S47;
    DiffPair_float_0 _S48 = *dpy_1;
    if((*dpy_1).primal_0 < _S46.primal_0)
    {
        _S47 = dOut_1;
    }
    else
    {
        _S47 = 0.0f;
    }
    dpy_1->primal_0 = _S48.primal_0;
    dpy_1->differential_0 = _S47;
    return;
}

__device__ float s_primal_ctx_min_0(float _S49, float _S50)
{
    return (F32_min((_S49), (_S50)));
}

__device__ void s_bwd_prop_min_0(DiffPair_float_0 * _S51, DiffPair_float_0 * _S52, float _S53)
{
    _d_min_0(_S51, _S52, _S53);
    return;
}

__device__ void s_bwd_prop_face_tangent_0(DiffPair_float3_0 * dpv0_0, DiffPair_float3_0 * dpv1_0, DiffPair_float3_0 * dpv2_0, DiffPair_float2_0 * dpt0_0, DiffPair_float2_0 * dpt1_0, DiffPair_float2_0 * dpt2_0, float3  s_diff_tangent_T_0)
{
    float2  t1t0_1 = (*dpt1_0).primal_0 - (*dpt0_0).primal_0;
    float2  t2t0_1 = (*dpt2_0).primal_0 - (*dpt0_0).primal_0;
    float _S64 = t2t0_1.y;
    float _S65 = t1t0_1.y;
    float _S66 = t1t0_1.x;
    float _S67 = t2t0_1.x;
    float denom_2 = _S66 * _S64 - _S65 * _S67;
    bool _S68 = denom_2 > 0.0f;
    float3  v1v0_1 = (*dpv1_0).primal_0 - (*dpv0_0).primal_0;
    float3  v2v0_0 = (*dpv2_0).primal_0 - (*dpv0_0).primal_0;
    float3  _S69 = make_float3 (_S64);
    float3  _S70 = make_float3 (_S65);
    float3  nom_1 = v1v0_1 * make_float3 (_S64) - v2v0_0 * make_float3 (_S65);
    float denom_3;
    if(_S68)
    {
        denom_3 = s_primal_ctx_max_0(0.00000099999999747524f, denom_2);
    }
    else
    {
        denom_3 = s_primal_ctx_min_0(-0.00000099999999747524f, denom_2);
    }
    float3  _S71 = make_float3 (denom_3);
    float3  _S72 = make_float3 (denom_3 * denom_3);
    float3  _S73 = s_diff_tangent_T_0 / _S72;
    float3  _S74 = nom_1 * - _S73;
    float3  _S75 = _S71 * _S73;
    float _S76 = _S74.x + _S74.y + _S74.z;
    if(_S68)
    {
        DiffPair_float_0 _S77;
        (&_S77)->primal_0 = 0.00000099999999747524f;
        (&_S77)->differential_0 = 0.0f;
        DiffPair_float_0 _S78;
        (&_S78)->primal_0 = denom_2;
        (&_S78)->differential_0 = 0.0f;
        s_bwd_prop_max_0(&_S77, &_S78, _S76);
        denom_3 = _S78.differential_0;
    }
    else
    {
        DiffPair_float_0 _S79;
        (&_S79)->primal_0 = -0.00000099999999747524f;
        (&_S79)->differential_0 = 0.0f;
        DiffPair_float_0 _S80;
        (&_S80)->primal_0 = denom_2;
        (&_S80)->differential_0 = 0.0f;
        s_bwd_prop_min_0(&_S79, &_S80, _S76);
        denom_3 = _S80.differential_0;
    }
    float _S81 = - denom_3;
    float3  _S82 = - _S75;
    float3  _S83 = v2v0_0 * _S82;
    float3  s_diff_v2v0_T_0 = _S70 * _S82;
    float3  _S84 = v1v0_1 * _S75;
    float3  s_diff_v1v0_T_0 = _S69 * _S75;
    float3  _S85 = - s_diff_v2v0_T_0;
    float3  _S86 = - s_diff_v1v0_T_0;
    float2  s_diff_t2t0_T_0 = make_float2 (_S65 * _S81, _S66 * denom_3 + _S84.x + _S84.y + _S84.z);
    float2  _S87 = - s_diff_t2t0_T_0;
    float2  s_diff_t1t0_T_0 = make_float2 (_S64 * denom_3, _S67 * _S81 + _S83.x + _S83.y + _S83.z);
    float2  _S88 = - s_diff_t1t0_T_0;
    dpt2_0->primal_0 = (*dpt2_0).primal_0;
    dpt2_0->differential_0 = s_diff_t2t0_T_0;
    dpt1_0->primal_0 = (*dpt1_0).primal_0;
    dpt1_0->differential_0 = s_diff_t1t0_T_0;
    float2  _S89 = _S87 + _S88;
    dpt0_0->primal_0 = (*dpt0_0).primal_0;
    dpt0_0->differential_0 = _S89;
    dpv2_0->primal_0 = (*dpv2_0).primal_0;
    dpv2_0->differential_0 = s_diff_v2v0_T_0;
    dpv1_0->primal_0 = (*dpv1_0).primal_0;
    dpv1_0->differential_0 = s_diff_v1v0_T_0;
    float3  _S90 = _S85 + _S86;
    dpv0_0->primal_0 = (*dpv0_0).primal_0;
    dpv0_0->differential_0 = _S90;
    return;
}

__device__ void bwd_face_tangent(DiffPair_float3_0 * v0, DiffPair_float3_0 * v1, DiffPair_float3_0 * v2, DiffPair_float2_0 * t0, DiffPair_float2_0 * t1, DiffPair_float2_0 * t2, float3 diff_tangent)
{
    s_bwd_prop_face_tangent_0(v0, v1, v2, t0, t1, t2, diff_tangent);
    return;
}