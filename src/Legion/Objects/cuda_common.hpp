
// Copyright (C) 2011 R. Keith Morley 
// 
// (MIT/X11 License)
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef LEGION_OBJECTS_CUDA_COMMON_HPP_
#define LEGION_OBJECTS_CUDA_COMMON_HPP_

#include <optix_world.h>
#include <Legion/Common/Math/CUDA/Sobol.hpp>

namespace legion
{

struct RayGeometry
{
    float3 origin;
    float3 direction;
};


//------------------------------------------------------------------------------
//
// this struct is to be used as the per-ray-data for radiance rays
//
//------------------------------------------------------------------------------
struct RadiancePRD 
{
  float3   result;
  float    importance;
  int      depth;
  unsigned count_emitted_light;
};

//------------------------------------------------------------------------------
//
// this struct is to be used as the per-ray-data for shadow rays
//
//------------------------------------------------------------------------------
struct ShadowPRD 
{
  float3 attenuation;
};


//------------------------------------------------------------------------------
//
// Given a pixel and sample number, calculate screen, lens, and time samples
// using the Sobol sequence.  Returns the sobol index so that further dimensions
// can be queried (eg, for BSDF sampling).  Note that this function uses the
// first 5 dimensions of the sobol sequence, so any further sampling of this
// sobol_index should start at the 6th dim
//
//------------------------------------------------------------------------------
__device__ unsigned generateSobolSamples( const uint2& launch_dim,
                                          const uint2& pixel_coord,
                                          unsigned     sample_index,
                                          float2&      screen_sample,
                                          float2&      lens_sample,
                                          float&       time_sample )
{
    const float2 inv_dim = make_float2( 1.0f ) / 
                           make_float2( launch_dim.x, launch_dim.y );

    screen_sample = make_float2(0.5f );
    unsigned sobol_index;
    legion::Sobol::getRasterPos( 12, // 2^m should be > film max_dim
                                 sample_index,
                                 pixel_coord,
                                 screen_sample,
                                 sobol_index );

    screen_sample = screen_sample * inv_dim;
    lens_sample   = make_float2( legion::Sobol::gen( sobol_index, 2 ),
                                 legion::Sobol::gen( sobol_index, 3 ) );
    time_sample   = 0.0f;

    return sobol_index;
}

__device__ optix::Ray makePrimaryRay( float3 origin, float3 direction )
{
    return optix::make_Ray( origin, direction, 0u, 0.0f, RT_DEFAULT_MAX );
}


} // end namespace legion

//------------------------------------------------------------------------------
//
// Common rtVariables.  These will all be set internally by legion 
//
//------------------------------------------------------------------------------
rtDeclareVariable( legion::RadiancePRD, radiance_prd, rtPayload, );
rtDeclareVariable( legion::ShadowPRD,   shadow_prd,   rtPayload, );

// TODO: document these
rtDeclareVariable( unsigned, legion_radiance_ray_type, , );
rtDeclareVariable( unsigned, legion_shadow_ray_type  , , );
rtDeclareVariable( rtObject, legion_top_group, , );

//------------------------------------------------------------------------------
//
// Camera helpers
//
//------------------------------------------------------------------------------
rtCallableProgram( legion::RayGeometry,
                   legionCameraCreateRay, 
                   (float2, float2, float ) );


//------------------------------------------------------------------------------
//
// Surface helpers
//
//------------------------------------------------------------------------------

namespace legion
{
struct BSDFSample
{
    float3 w_in;
    float3 f_over_pdf;
};

struct LocalGeometry 
{
    float3   position_object;
    float3   geometric_normal;
    float3   shading_normal;
    float2   texcoord;
};

}

rtCallableProgram( legion::BSDFSample,
                   legionSampleBSDF,
                   ( float2, float3 , legion::LocalGeometry ) );

rtCallableProgram( float3, 
                   legionEvaluateBSDF,
                   ( float3 , legion::LocalGeometry, float3 ) );

rtCallableProgram( float, 
                   legionPDF,
                   ( float3, legion::LocalGeometry, float3 ) );

rtCallableProgram( float3, 
                   legionEmission,
                   ( float3, legion::LocalGeometry ) );

#endif // LEGION_OBJECTS_CUDA_COMMON_HPP_
