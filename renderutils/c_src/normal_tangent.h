#include "common.h"

struct NormalKernelParams
{
    const float*    Positions;
    float*          PositionsGrad;
    int32_t*        Indexes;
    int             NumTriangles;
    float*          Normals;
    const float*    NormalsGrad;
};

struct TangentKernelParams
{
    const float*    Positions;
    float*          PositionsGrad;
    const float*    TexCoords;
    float*          TexCoordsGrad;
    int32_t*        PosIndexes;
    int32_t*        UVIndexes;
    int             NumTriangles;
    float*          Tangents;
    const float*    TangentsGrad;
    int             ClusterIdx;         // for debug
};