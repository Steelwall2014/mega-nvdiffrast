#pragma once
#include <cuda.h>

#define VSM_MAX_MIP_LEVEL 16
#define VSM_FILTER_MODE_NEAREST                        0
#define VSM_FILTER_MODE_LINEAR                         1

struct VirtualShadowMapFeedbackKernalParams
{
    const float*    gb_pos;                         // Incoming position gbuffer.
    const float*    camera_pos;                     // the camera positions
    const bool*     mask;                           // Incoming mask of gb_pos.
    float*          shadow_map_uv;                  // Incoming and outgoing uv used to sample the virtual shadow map.
    float*          vsm_mip_levels;                 // Incoming virtual shadow map mip levels.
    bool*           feedback[VSM_MAX_MIP_LEVEL];    // Outgoing virtual shadow map feedback.
    int             num_pages[VSM_MAX_MIP_LEVEL];   // the number of pages in each mipmap level
    int             max_mipmap_level;               // the maximum mipmap level of virtual shadow map
    int             n;                              // Minibatch size.
    int             imgWidth;                       // Image width.
    int             imgHeight;                      // Image height.
    int             vsmWidth;                       // VSM width.
    int             vsmHeight;                      // VSM height.
    int             page_size_x;                    // Number of pixels of a virtual texture page in x axis
    int             page_size_y;                    // Number of pixels of a virtual texture page in y axis
    float           first_level_extent;             // the radius around the camera that will have the highest resolution of VSM
};