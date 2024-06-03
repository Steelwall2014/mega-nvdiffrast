// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include <cuda.h>
#ifdef ENABLE_HALF_TEXTURE
#include <cuda_fp16.h>
#endif
#include <vector_types.h>
#include <vector_functions.h>
#include <stdint.h>

#ifdef __CUDACC__
#define __CUDA_HOST__ __host__
#define __CUDA_DEVICE__ __device__
#define __CUDA_FORCEINLINE__ __forceinline__
#else
#define __CUDA_HOST__
#define __CUDA_DEVICE__
#define __CUDA_FORCEINLINE__ inline
#endif

//------------------------------------------------------------------------
// C++ helper function prototypes.

dim3 getLaunchBlockSize(int maxWidth, int maxHeight, int width, int height);
dim3 getLaunchGridSize(dim3 blockSize, int width, int height, int depth);

//------------------------------------------------------------------------
// The rest is CUDA device code specific stuff.

//------------------------------------------------------------------------
// Helpers for CUDA vector types.

template<typename T> struct value_type_traits { };
template<> struct value_type_traits<float> { using type = float; };
template<> struct value_type_traits<float2> { using type = float; };
template<> struct value_type_traits<float4> { using type = float; };

template<typename T, typename U> __CUDA_DEVICE__ __CUDA_FORCEINLINE__ T cast(U x) { return (T)x; }
template<> __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float cast(float x) { return x; }
template<> __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2 cast(float x) { return make_float2(x, x); }
template<> __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4 cast(float x) { return make_float4(x, x, x, x); }

static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2&   operator*=  (float2& a, const float2& b)       { a.x *= b.x; a.y *= b.y; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2&   operator+=  (float2& a, const float2& b)       { a.x += b.x; a.y += b.y; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2&   operator-=  (float2& a, const float2& b)       { a.x -= b.x; a.y -= b.y; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2&   operator*=  (float2& a, float b)               { a.x *= b; a.y *= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2&   operator+=  (float2& a, float b)               { a.x += b; a.y += b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2&   operator-=  (float2& a, float b)               { a.x -= b; a.y -= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2    operator*   (const float2& a, const float2& b) { return make_float2(a.x * b.x, a.y * b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2    operator+   (const float2& a, const float2& b) { return make_float2(a.x + b.x, a.y + b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2    operator-   (const float2& a, const float2& b) { return make_float2(a.x - b.x, a.y - b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2    operator*   (const float2& a, float b)         { return make_float2(a.x * b, a.y * b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2    operator+   (const float2& a, float b)         { return make_float2(a.x + b, a.y + b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2    operator-   (const float2& a, float b)         { return make_float2(a.x - b, a.y - b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2    operator*   (float a, const float2& b)         { return make_float2(a * b.x, a * b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2    operator+   (float a, const float2& b)         { return make_float2(a + b.x, a + b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2    operator-   (float a, const float2& b)         { return make_float2(a - b.x, a - b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2    operator-   (const float2& a)                  { return make_float2(-a.x, -a.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3&   operator*=  (float3& a, const float3& b)       { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3&   operator+=  (float3& a, const float3& b)       { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3&   operator-=  (float3& a, const float3& b)       { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3&   operator*=  (float3& a, float b)               { a.x *= b; a.y *= b; a.z *= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3&   operator+=  (float3& a, float b)               { a.x += b; a.y += b; a.z += b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3&   operator-=  (float3& a, float b)               { a.x -= b; a.y -= b; a.z -= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3    operator*   (const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3    operator+   (const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3    operator-   (const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3    operator/   (const float3& a, float b)         { return make_float3(a.x / b, a.y / b, a.z / b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3    operator/   (const float3& a, const float3& b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3    operator*   (const float3& a, float b)         { return make_float3(a.x * b, a.y * b, a.z * b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3    operator+   (const float3& a, float b)         { return make_float3(a.x + b, a.y + b, a.z + b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3    operator-   (const float3& a, float b)         { return make_float3(a.x - b, a.y - b, a.z - b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3    operator*   (float a, const float3& b)         { return make_float3(a * b.x, a * b.y, a * b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3    operator+   (float a, const float3& b)         { return make_float3(a + b.x, a + b.y, a + b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3    operator-   (float a, const float3& b)         { return make_float3(a - b.x, a - b.y, a - b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3    operator-   (const float3& a)                  { return make_float3(-a.x, -a.y, -a.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4&   operator*=  (float4& a, const float4& b)       { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4&   operator+=  (float4& a, const float4& b)       { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4&   operator-=  (float4& a, const float4& b)       { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4&   operator*=  (float4& a, float b)               { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4&   operator+=  (float4& a, float b)               { a.x += b; a.y += b; a.z += b; a.w += b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4&   operator-=  (float4& a, float b)               { a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4    operator*   (const float4& a, const float4& b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4    operator+   (const float4& a, const float4& b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4    operator-   (const float4& a, const float4& b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4    operator*   (const float4& a, float b)         { return make_float4(a.x * b, a.y * b, a.z * b, a.w * b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4    operator+   (const float4& a, float b)         { return make_float4(a.x + b, a.y + b, a.z + b, a.w + b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4    operator-   (const float4& a, float b)         { return make_float4(a.x - b, a.y - b, a.z - b, a.w - b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4    operator*   (float a, const float4& b)         { return make_float4(a * b.x, a * b.y, a * b.z, a * b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4    operator+   (float a, const float4& b)         { return make_float4(a + b.x, a + b.y, a + b.z, a + b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4    operator-   (float a, const float4& b)         { return make_float4(a - b.x, a - b.y, a - b.z, a - b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4    operator-   (const float4& a)                  { return make_float4(-a.x, -a.y, -a.z, -a.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int2&     operator*=  (int2& a, const int2& b)           { a.x *= b.x; a.y *= b.y; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int2&     operator+=  (int2& a, const int2& b)           { a.x += b.x; a.y += b.y; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int2&     operator-=  (int2& a, const int2& b)           { a.x -= b.x; a.y -= b.y; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int2&     operator*=  (int2& a, int b)                   { a.x *= b; a.y *= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int2&     operator+=  (int2& a, int b)                   { a.x += b; a.y += b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int2&     operator-=  (int2& a, int b)                   { a.x -= b; a.y -= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int2      operator*   (const int2& a, const int2& b)     { return make_int2(a.x * b.x, a.y * b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int2      operator+   (const int2& a, const int2& b)     { return make_int2(a.x + b.x, a.y + b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int2      operator-   (const int2& a, const int2& b)     { return make_int2(a.x - b.x, a.y - b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int2      operator*   (const int2& a, int b)             { return make_int2(a.x * b, a.y * b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int2      operator+   (const int2& a, int b)             { return make_int2(a.x + b, a.y + b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int2      operator-   (const int2& a, int b)             { return make_int2(a.x - b, a.y - b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int2      operator*   (int a, const int2& b)             { return make_int2(a * b.x, a * b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int2      operator+   (int a, const int2& b)             { return make_int2(a + b.x, a + b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int2      operator-   (int a, const int2& b)             { return make_int2(a - b.x, a - b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int2      operator-   (const int2& a)                    { return make_int2(-a.x, -a.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3&     operator*=  (int3& a, const int3& b)           { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3&     operator+=  (int3& a, const int3& b)           { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3&     operator-=  (int3& a, const int3& b)           { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3&     operator*=  (int3& a, int b)                   { a.x *= b; a.y *= b; a.z *= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3&     operator+=  (int3& a, int b)                   { a.x += b; a.y += b; a.z += b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3&     operator-=  (int3& a, int b)                   { a.x -= b; a.y -= b; a.z -= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3      operator*   (const int3& a, const int3& b)     { return make_int3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3      operator+   (const int3& a, const int3& b)     { return make_int3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3      operator-   (const int3& a, const int3& b)     { return make_int3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3      operator*   (const int3& a, int b)             { return make_int3(a.x * b, a.y * b, a.z * b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3      operator+   (const int3& a, int b)             { return make_int3(a.x + b, a.y + b, a.z + b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3      operator-   (const int3& a, int b)             { return make_int3(a.x - b, a.y - b, a.z - b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3      operator*   (int a, const int3& b)             { return make_int3(a * b.x, a * b.y, a * b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3      operator+   (int a, const int3& b)             { return make_int3(a + b.x, a + b.y, a + b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3      operator-   (int a, const int3& b)             { return make_int3(a - b.x, a - b.y, a - b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3      operator-   (const int3& a)                    { return make_int3(-a.x, -a.y, -a.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4&     operator*=  (int4& a, const int4& b)           { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4&     operator+=  (int4& a, const int4& b)           { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4&     operator-=  (int4& a, const int4& b)           { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4&     operator*=  (int4& a, int b)                   { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4&     operator+=  (int4& a, int b)                   { a.x += b; a.y += b; a.z += b; a.w += b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4&     operator-=  (int4& a, int b)                   { a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4      operator*   (const int4& a, const int4& b)     { return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4      operator+   (const int4& a, const int4& b)     { return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4      operator-   (const int4& a, const int4& b)     { return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4      operator*   (const int4& a, int b)             { return make_int4(a.x * b, a.y * b, a.z * b, a.w * b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4      operator+   (const int4& a, int b)             { return make_int4(a.x + b, a.y + b, a.z + b, a.w + b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4      operator-   (const int4& a, int b)             { return make_int4(a.x - b, a.y - b, a.z - b, a.w - b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4      operator*   (int a, const int4& b)             { return make_int4(a * b.x, a * b.y, a * b.z, a * b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4      operator+   (int a, const int4& b)             { return make_int4(a + b.x, a + b.y, a + b.z, a + b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4      operator-   (int a, const int4& b)             { return make_int4(a - b.x, a - b.y, a - b.z, a - b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4      operator-   (const int4& a)                    { return make_int4(-a.x, -a.y, -a.z, -a.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint2&    operator*=  (uint2& a, const uint2& b)         { a.x *= b.x; a.y *= b.y; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint2&    operator+=  (uint2& a, const uint2& b)         { a.x += b.x; a.y += b.y; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint2&    operator-=  (uint2& a, const uint2& b)         { a.x -= b.x; a.y -= b.y; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint2&    operator*=  (uint2& a, unsigned int b)         { a.x *= b; a.y *= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint2&    operator+=  (uint2& a, unsigned int b)         { a.x += b; a.y += b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint2&    operator-=  (uint2& a, unsigned int b)         { a.x -= b; a.y -= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint2     operator*   (const uint2& a, const uint2& b)   { return make_uint2(a.x * b.x, a.y * b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint2     operator+   (const uint2& a, const uint2& b)   { return make_uint2(a.x + b.x, a.y + b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint2     operator-   (const uint2& a, const uint2& b)   { return make_uint2(a.x - b.x, a.y - b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint2     operator*   (const uint2& a, unsigned int b)   { return make_uint2(a.x * b, a.y * b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint2     operator+   (const uint2& a, unsigned int b)   { return make_uint2(a.x + b, a.y + b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint2     operator-   (const uint2& a, unsigned int b)   { return make_uint2(a.x - b, a.y - b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint2     operator*   (unsigned int a, const uint2& b)   { return make_uint2(a * b.x, a * b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint2     operator+   (unsigned int a, const uint2& b)   { return make_uint2(a + b.x, a + b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint2     operator-   (unsigned int a, const uint2& b)   { return make_uint2(a - b.x, a - b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint3&    operator*=  (uint3& a, const uint3& b)         { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint3&    operator+=  (uint3& a, const uint3& b)         { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint3&    operator-=  (uint3& a, const uint3& b)         { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint3&    operator*=  (uint3& a, unsigned int b)         { a.x *= b; a.y *= b; a.z *= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint3&    operator+=  (uint3& a, unsigned int b)         { a.x += b; a.y += b; a.z += b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint3&    operator-=  (uint3& a, unsigned int b)         { a.x -= b; a.y -= b; a.z -= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint3     operator*   (const uint3& a, const uint3& b)   { return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint3     operator+   (const uint3& a, const uint3& b)   { return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint3     operator-   (const uint3& a, const uint3& b)   { return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint3     operator*   (const uint3& a, unsigned int b)   { return make_uint3(a.x * b, a.y * b, a.z * b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint3     operator+   (const uint3& a, unsigned int b)   { return make_uint3(a.x + b, a.y + b, a.z + b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint3     operator-   (const uint3& a, unsigned int b)   { return make_uint3(a.x - b, a.y - b, a.z - b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint3     operator*   (unsigned int a, const uint3& b)   { return make_uint3(a * b.x, a * b.y, a * b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint3     operator+   (unsigned int a, const uint3& b)   { return make_uint3(a + b.x, a + b.y, a + b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint3     operator-   (unsigned int a, const uint3& b)   { return make_uint3(a - b.x, a - b.y, a - b.z); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4&    operator*=  (uint4& a, const uint4& b)         { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4&    operator+=  (uint4& a, const uint4& b)         { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4&    operator-=  (uint4& a, const uint4& b)         { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4&    operator*=  (uint4& a, unsigned int b)         { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4&    operator+=  (uint4& a, unsigned int b)         { a.x += b; a.y += b; a.z += b; a.w += b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4&    operator-=  (uint4& a, unsigned int b)         { a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4     operator*   (const uint4& a, const uint4& b)   { return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4     operator+   (const uint4& a, const uint4& b)   { return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4     operator-   (const uint4& a, const uint4& b)   { return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4     operator*   (const uint4& a, unsigned int b)   { return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4     operator+   (const uint4& a, unsigned int b)   { return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4     operator-   (const uint4& a, unsigned int b)   { return make_uint4(a.x - b, a.y - b, a.z - b, a.w - b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4     operator*   (unsigned int a, const uint4& b)   { return make_uint4(a * b.x, a * b.y, a * b.z, a * b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4     operator+   (unsigned int a, const uint4& b)   { return make_uint4(a + b.x, a + b.y, a + b.z, a + b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4     operator-   (unsigned int a, const uint4& b)   { return make_uint4(a - b.x, a - b.y, a - b.z, a - b.w); }

template<class T> static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ T zero_value(void);
template<> __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float  zero_value<float> (void)                      { return 0.f; }
template<> __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2 zero_value<float2>(void)                      { return make_float2(0.f, 0.f); }
template<> __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4 zero_value<float4>(void)                      { return make_float4(0.f, 0.f, 0.f, 0.f); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3 make_float3(float a)                              { return make_float3(a, a, a); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float3 make_float3(const float2& a, float b)             { return make_float3(a.x, a.y, b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4 make_float4(const float3& a, float b)             { return make_float4(a.x, a.y, a.z, b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4 make_float4(const float2& a, const float2& b)     { return make_float4(a.x, a.y, b.x, b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int3 make_int3(const int2& a, int b)                     { return make_int3(a.x, a.y, b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4 make_int4(const int3& a, int b)                     { return make_int4(a.x, a.y, a.z, b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ int4 make_int4(const int2& a, const int2& b)             { return make_int4(a.x, a.y, b.x, b.y); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint3 make_uint3(const uint2& a, unsigned int b)         { return make_uint3(a.x, a.y, b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4 make_uint4(const uint3& a, unsigned int b)         { return make_uint4(a.x, a.y, a.z, b); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ uint4 make_uint4(const uint2& a, const uint2& b)         { return make_uint4(a.x, a.y, b.x, b.y); }


#ifdef __CUDACC__
template<class T> static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ void swap(T& a, T& b)                  { T temp = a; a = b; b = temp; }
//------------------------------------------------------------------------
// Coalesced atomics. These are all done via macros.

#if __CUDA_ARCH__ >= 700 // Warp match instruction __match_any_sync() is only available on compute capability 7.x and higher

#define CA_TEMP       _ca_temp
#define CA_TEMP_PARAM float* CA_TEMP
#define CA_DECLARE_TEMP(threads_per_block) \
    __shared__ float CA_TEMP[(threads_per_block)]

#define CA_SET_GROUP_MASK(group, thread_mask)                   \
    bool   _ca_leader;                                          \
    float* _ca_ptr;                                             \
    do {                                                        \
        int tidx   = threadIdx.x + blockDim.x * threadIdx.y;    \
        int lane   = tidx & 31;                                 \
        int warp   = tidx >> 5;                                 \
        int tmask  = __match_any_sync((thread_mask), (group));  \
        int leader = __ffs(tmask) - 1;                          \
        _ca_leader = (leader == lane);                          \
        _ca_ptr    = &_ca_temp[((warp << 5) + leader)];         \
    } while(0)

#define CA_SET_GROUP(group) \
    CA_SET_GROUP_MASK((group), 0xffffffffu)

#define caAtomicAdd(ptr, value)         \
    do {                                \
        if (_ca_leader)                 \
            *_ca_ptr = 0.f;             \
        atomicAdd(_ca_ptr, (value));    \
        if (_ca_leader)                 \
            atomicAdd((ptr), *_ca_ptr); \
    } while(0)

#define caAtomicAdd3_xyw(ptr, x, y, w)  \
    do {                                \
        caAtomicAdd((ptr), (x));        \
        caAtomicAdd((ptr)+1, (y));      \
        caAtomicAdd((ptr)+3, (w));      \
    } while(0)

#define caAtomicAddTexture(ptr, level, idx, value)  \
    do {                                            \
        CA_SET_GROUP((idx) ^ ((level) << 27));      \
        caAtomicAdd((ptr)+(idx), (value));          \
    } while(0)

//------------------------------------------------------------------------
// Disable atomic coalescing for compute capability lower than 7.x

#else // __CUDA_ARCH__ >= 700
#define CA_TEMP _ca_temp
#define CA_TEMP_PARAM float CA_TEMP
#define CA_DECLARE_TEMP(threads_per_block) CA_TEMP_PARAM
#define CA_SET_GROUP_MASK(group, thread_mask)
#define CA_SET_GROUP(group)
#define caAtomicAdd(ptr, value) atomicAdd((ptr), (value))
#define caAtomicAdd3_xyw(ptr, x, y, w)  \
    do {                                \
        atomicAdd((ptr), (x));          \
        atomicAdd((ptr)+1, (y));        \
        atomicAdd((ptr)+3, (w));        \
    } while(0)
#define caAtomicAddTexture(ptr, level, idx, value) atomicAdd((ptr)+(idx), (value))
#endif // __CUDA_ARCH__ >= 700

//------------------------------------------------------------------------
#endif // __CUDACC__

#ifdef ENABLE_HALF_TEXTURE

#ifndef __CUDACC__
inline half __hadd(half a, half b)
{
    return __float2half(__half2float(a) + __half2float(b));
}
inline half __hsub(half a, half b)
{
    return __float2half(__half2float(a) - __half2float(b));
}
inline half __hmul(half a, half b)
{
    return __float2half(__half2float(a) * __half2float(b));
}
inline half2 __hadd2(half2 a, half2 b)
{
    return half2(__float2half(__half2float(a.x) + __half2float(b.x)), __float2half(__half2float(a.y) + __half2float(b.y)));
}
inline half2 __hsub2(half2 a, half2 b)
{
    return half2(__float2half(__half2float(a.x) - __half2float(b.x)), __float2half(__half2float(a.y) - __half2float(b.y)));
}
inline half2 __hmul2(half2 a, half2 b)
{
    return half2(__float2half(__half2float(a.x) * __half2float(b.x)), __float2half(__half2float(a.y) * __half2float(b.y)));
}
#endif // __CUDACC__

struct alignas(4) half4 {
    half x, y, z, w;
    __CUDA_HOST__ __CUDA_DEVICE__ half4() = default;
    __CUDA_HOST__ __CUDA_DEVICE__ half4(half x, half y, half z, half w) { this->x=x; this->y=y; this->z=z; this->w=w; }
};

template<> struct value_type_traits<half> { using type = half; };
template<> struct value_type_traits<half2> { using type = half; };
template<> struct value_type_traits<half4> { using type = half; };

template<> __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half cast(float x) { return __float2half(x); }
template<> __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float cast(half x) { return __half2float(x); }
template<> __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half2 cast(float x) { return half2(cast<half>(x), cast<half>(x)); }
template<> __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half4 cast(float x) { return half4(cast<half>(x), cast<half>(x), cast<half>(x), cast<half>(x)); }

template<> __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float2 cast(half2 x) { return make_float2(cast<float>(x.x), cast<float>(x.y)); }
template<> __CUDA_DEVICE__ __CUDA_FORCEINLINE__ float4 cast(half4 x) { return make_float4(cast<float>(x.x), cast<float>(x.y), cast<float>(x.z), cast<float>(x.w)); }

static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half      operator+   (const half &lh, const half &rh)   { return __hadd(lh, rh); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half      operator-   (const half &lh, const half &rh)   { return __hsub(lh, rh); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half      operator*   (const half &lh, const half &rh)   { return __hmul(lh, rh); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half      operator+=  (const half &lh, const half &rh)   { return __hsub(lh, rh); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half&     operator+=  (half &lh, const half &rh)         { lh = __hadd(lh, rh); return lh; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half2     operator*   (const half2 &lh, const half2 &rh) { return __hmul2(lh, rh); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half2&    operator+=  (half2 &lh, const half2 &rh)       { lh = __hadd2(lh, rh); return lh; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half4     operator+   (const half4& a, const half4& b)   { return half4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half4     operator*   (const half4& a, float b)          { return half4(a.x*cast<half>(b), a.y*cast<half>(b), a.z*cast<half>(b), a.w*cast<half>(b)); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half4     operator*   (const half4& a, const half4& b)   { return half4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w); }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half4&    operator+=  (half4& a, const half4& b)         { a = a + b; return a; }
static __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half4&    operator*=  (half4& a, float b)                { a = a * b; return a; }

template<> __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half   zero_value<half> (void)                       { return cast<half>(0.f); }
template<> __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half2  zero_value<half2>(void)                       { return cast<half2>(0.f); }
template<> __CUDA_DEVICE__ __CUDA_FORCEINLINE__ half4  zero_value<half4>(void)                       { return cast<half4>(0.f); }

#endif // ENABLE_HALF_TEXTURE