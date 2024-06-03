#include <cuda.h>
#include "normal_tangent.h"
#include "slang_generated.inl"

__device__ static __forceinline__ float3 load3(const float* Tensor, int index)
{
    float3 result;
    result.x = Tensor[index*3 + 0];
    result.y = Tensor[index*3 + 1];
    result.z = Tensor[index*3 + 2];
    return result;
}

__device__ static __forceinline__ float2 load2(const float* Tensor, int index)
{
    float2 result;
    result.x = Tensor[index*2 + 0];
    result.y = Tensor[index*2 + 1];
    return result;
}

__device__ static __forceinline__ float3 cross(float3 a, float3 b)
{
    float3 out;
    out.x = a.y * b.z - a.z * b.y;
    out.y = a.z * b.x - a.x * b.z;
    out.z = a.x * b.y - a.y * b.x;
    return out;
}

__device__ static __forceinline__ void atomicAdd_xyz(float3* address, float3 val)
{
    atomicAdd(&address->x, val.x);
    atomicAdd(&address->y, val.y);
    atomicAdd(&address->z, val.z);
}

__device__ static __forceinline__ void atomicAdd_xy(float2* address, float2 val)
{
    atomicAdd(&address->x, val.x);
    atomicAdd(&address->y, val.y);
}

__global__ void CalculateNormalKernel(const NormalKernelParams p)
{
    int TriangleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (TriangleIdx >= p.NumTriangles)
        return;

    int vid0 = p.Indexes[TriangleIdx*3 + 0];
    int vid1 = p.Indexes[TriangleIdx*3 + 1];
    int vid2 = p.Indexes[TriangleIdx*3 + 2];
    float3 v0 = load3(p.Positions, vid0);
    float3 v1 = load3(p.Positions, vid1);
    float3 v2 = load3(p.Positions, vid2);

    float3 face_normals = cross(v1 - v0, v2 - v0);

    atomicAdd_xyz((float3*)&p.Normals[vid0*3], face_normals);
    atomicAdd_xyz((float3*)&p.Normals[vid1*3], face_normals);
    atomicAdd_xyz((float3*)&p.Normals[vid2*3], face_normals);
}

__global__ void CalculateNormalGradKernel(const NormalKernelParams p)
{
    int TriangleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (TriangleIdx >= p.NumTriangles)
        return;

    int vid0 = p.Indexes[TriangleIdx*3 + 0];
    int vid1 = p.Indexes[TriangleIdx*3 + 1];
    int vid2 = p.Indexes[TriangleIdx*3 + 2];
    float3 v0 = load3(p.Positions, vid0);
    float3 v1 = load3(p.Positions, vid1);
    float3 v2 = load3(p.Positions, vid2);
    float3 dLdn0 = load3(p.NormalsGrad, vid0);  // the derivative of the loss with respect to the normal of v0
    float3 dLdn1 = load3(p.NormalsGrad, vid1);  // the derivative of the loss with respect to the normal of v1
    float3 dLdn2 = load3(p.NormalsGrad, vid2);  // the derivative of the loss with respect to the normal of v2
    float3 dLdn = dLdn0 + dLdn1 + dLdn2;

    /*
    *  dn0dv0 = dn1dv0 = dn2dv0 = [0, -v2v1.z, v2v1.y, 
    *                              v2v1.z, 0, -v2v1.x, 
    *                              -v2v1.y, v2v1.x, 0]
    *  dn0dv1 = dn1dv1 = dn2dv1 = [0, -v0v2.z, v0v2.y,
    *                              v0v2.z, 0, -v0v2.x,
    *                              -v0v2.y, v0v2.x, 0]
    *  dn0dv2 = dn1dv2 = dn2dv2 = [0, -v1v0.z, v1v0.y,
    *                              v1v0.z, 0, -v1v0.x,
    *                              -v1v0.y, v1v0.x, 0]
    */ 
    float3 v2v1 = v2 - v1;
    float3 v0v2 = v0 - v2;
    float3 v1v0 = v1 - v0;

    // dLdv0 += dLdn0 * dn0dv0 + dLdn1 * dn1dv0 + dLdn2 * dn2dv0
    float3 dLdv0;
    dLdv0.x = dLdn.x * 0 + dLdn.y * -v2v1.z + dLdn.z * v2v1.y;
    dLdv0.y = dLdn.x * v2v1.z + dLdn.y * 0 + dLdn.z * -v2v1.x;
    dLdv0.z = dLdn.x * -v2v1.y + dLdn.y * v2v1.x + dLdn.z * 0;

    // dLdv1 = dLdn0 * dn0dv1 + dLdn1 * dn1dv1 + dLdn2 * dn2dv1
    float3 dLdv1;
    dLdv1.x = dLdn.x * 0 + dLdn.y * -v0v2.z + dLdn.z * v0v2.y;
    dLdv1.y = dLdn.x * v0v2.z + dLdn.y * 0 + dLdn.z * -v0v2.x;
    dLdv1.z = dLdn.x * -v0v2.y + dLdn.y * v0v2.x + dLdn.z * 0;

    // dLdv2 = dLdn0 * dn0dv2 + dLdn1 * dn1dv2 + dLdn2 * dn2dv2
    float3 dLdv2;
    dLdv2.x = dLdn.x * 0 + dLdn.y * -v1v0.z + dLdn.z * v1v0.y;
    dLdv2.y = dLdn.x * v1v0.z + dLdn.y * 0 + dLdn.z * -v1v0.x;
    dLdv2.z = dLdn.x * -v1v0.y + dLdn.y * v1v0.x + dLdn.z * 0;

    atomicAdd_xyz((float3*)&p.PositionsGrad[vid0*3], dLdv0);
    atomicAdd_xyz((float3*)&p.PositionsGrad[vid1*3], dLdv1);
    atomicAdd_xyz((float3*)&p.PositionsGrad[vid2*3], dLdv2);
}

__device__ float3 face_tangent(float3 v0, float3 v1, float3 v2, float2 t0, float2 t1, float2 t2)
{
    float2 t1t0 = t1 - t0;
    float2 t2t0 = t2 - t0;
    float3 v1v0 = v1 - v0;
    float3 v2v0 = v2 - v0;

    float3 nom = v1v0 * t2t0.y - v2v0 * t1t0.y;
    float denom = t1t0.x*t2t0.y - t1t0.y*t2t0.x;
    if (denom > 0.f)
        denom = max(1e-6, denom);
    else 
        denom = min(-1e-6, denom);
    float3 tangent = nom / denom;
    return tangent;
}

__global__ void CalculateTangentKernel(const TangentKernelParams p)
{
    int TriangleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (TriangleIdx >= p.NumTriangles)
        return;

    int vid0 = p.PosIndexes[TriangleIdx*3 + 0];
    int vid1 = p.PosIndexes[TriangleIdx*3 + 1];
    int vid2 = p.PosIndexes[TriangleIdx*3 + 2];
    float3 v0 = load3(p.Positions, vid0);
    float3 v1 = load3(p.Positions, vid1);
    float3 v2 = load3(p.Positions, vid2);
    int tid0 = p.UVIndexes[TriangleIdx*3 + 0];
    int tid1 = p.UVIndexes[TriangleIdx*3 + 1];
    int tid2 = p.UVIndexes[TriangleIdx*3 + 2];
    float2 t0 = load2(p.TexCoords, tid0);
    float2 t1 = load2(p.TexCoords, tid1);
    float2 t2 = load2(p.TexCoords, tid2);

    float3 tangent = face_tangent(v0, v1, v2, t0, t1, t2);
    atomicAdd_xyz((float3*)&p.Tangents[vid0*3], tangent);
    atomicAdd_xyz((float3*)&p.Tangents[vid1*3], tangent);
    atomicAdd_xyz((float3*)&p.Tangents[vid2*3], tangent);

}

__device__ DiffPair_float_0 diffPair(float primal)
{
    DiffPair_float_0 dp;
    dp.primal_0 = primal;
    dp.differential_0 = 0.f;
    return dp;
}

__device__ DiffPair_float2_0 diffPair(float2 primal)
{
    DiffPair_float2_0 dp;
    dp.primal_0 = primal;
    dp.differential_0 = make_float2(0.f, 0.f);
    return dp;
}

__device__ DiffPair_float3_0 diffPair(float3 primal)
{
    DiffPair_float3_0 dp;
    dp.primal_0 = primal;
    dp.differential_0 = make_float3(0.f, 0.f, 0.f);
    return dp;
}

__global__ void CalculateTangentGradKernel(const TangentKernelParams p)
{
    int TriangleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (TriangleIdx >= p.NumTriangles)
        return;

    int vid0 = p.PosIndexes[TriangleIdx*3 + 0];
    int vid1 = p.PosIndexes[TriangleIdx*3 + 1];
    int vid2 = p.PosIndexes[TriangleIdx*3 + 2];
    auto v0 = diffPair(load3(p.Positions, vid0));
    auto v1 = diffPair(load3(p.Positions, vid1));
    auto v2 = diffPair(load3(p.Positions, vid2));
    int tid0 = p.UVIndexes[TriangleIdx*3 + 0];
    int tid1 = p.UVIndexes[TriangleIdx*3 + 1];
    int tid2 = p.UVIndexes[TriangleIdx*3 + 2];
    auto t0 = diffPair(load2(p.TexCoords, tid0));
    auto t1 = diffPair(load2(p.TexCoords, tid1));
    auto t2 = diffPair(load2(p.TexCoords, tid2));

    float3 dv0 = make_float3(0.f, 0.f, 0.f);
    float3 dv1 = make_float3(0.f, 0.f, 0.f);
    float3 dv2 = make_float3(0.f, 0.f, 0.f);
    float2 dt0 = make_float2(0.f, 0.f);
    float2 dt1 = make_float2(0.f, 0.f);
    float2 dt2 = make_float2(0.f, 0.f);

    bwd_face_tangent(&v0, &v1, &v2, &t0, &t1, &t2, load3(p.TangentsGrad, vid0));
    dv0 += v0.differential_0;
    dv1 += v1.differential_0;
    dv2 += v2.differential_0;
    dt0 += t0.differential_0;
    dt1 += t1.differential_0;
    dt2 += t2.differential_0;

    bwd_face_tangent(&v0, &v1, &v2, &t0, &t1, &t2, load3(p.TangentsGrad, vid1));
    dv0 += v0.differential_0;
    dv1 += v1.differential_0;
    dv2 += v2.differential_0;
    dt0 += t0.differential_0;
    dt1 += t1.differential_0;
    dt2 += t2.differential_0;

    bwd_face_tangent(&v0, &v1, &v2, &t0, &t1, &t2, load3(p.TangentsGrad, vid2));
    dv0 += v0.differential_0;
    dv1 += v1.differential_0;
    dv2 += v2.differential_0;
    dt0 += t0.differential_0;
    dt1 += t1.differential_0;
    dt2 += t2.differential_0;

    atomicAdd_xyz((float3*)&p.PositionsGrad[vid0*3], dv0);
    atomicAdd_xyz((float3*)&p.PositionsGrad[vid1*3], dv1);
    atomicAdd_xyz((float3*)&p.PositionsGrad[vid2*3], dv2);
    atomicAdd_xy((float2*)&p.TexCoordsGrad[tid0*2], dt0);
    atomicAdd_xy((float2*)&p.TexCoordsGrad[tid1*2], dt1);
    atomicAdd_xy((float2*)&p.TexCoordsGrad[tid2*2], dt2);
}
