
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

//------------------------------------------------------------------------------
//
// Common rtVariables.  These will all be set internally by legion 
//
//------------------------------------------------------------------------------

// TODO: document these
rtDeclareVariable( unsigned, legion_radiance_ray_type, , );
rtDeclareVariable( unsigned, legion_shadow_ray_type  , , );
rtDeclareVariable( rtObject, legion_top_group, , );

rtDeclareVariable( uint2,    launch_index, rtLaunchIndex, );
rtDeclareVariable( uint2,    launch_dim,   rtLaunchDim, );



namespace legion
{

struct RayGeometry
{
    float3 origin;
    float3 direction;
};

} // end namespace legion


//------------------------------------------------------------------------------
//
// this struct is to be used as the per-ray-data for radiance rays
//
//------------------------------------------------------------------------------
namespace legion
{

// TODO: pack this
struct RadiancePRD 
{
    float3   origin;
    float3   direction;
    float3   radiance;
    float3   attenuation;
    unsigned long long sobol_index;
    unsigned sobol_dim;
    unsigned count_emitted_light;
    unsigned done;
};

//------------------------------------------------------------------------------
//
// this struct is to be used as the per-ray-data for shadow rays
//
//------------------------------------------------------------------------------
struct ShadowPRD 
{
    float3   hit_p;
    unsigned occluded;
};

} // end namespace legion

rtDeclareVariable( legion::RadiancePRD, radiance_prd, rtPayload, );
rtDeclareVariable( legion::ShadowPRD,   shadow_prd,   rtPayload, );


//------------------------------------------------------------------------------
//
// Given a pixel and sample number, calculate screen, lens, and time samples
// using the Sobol sequence.  Returns the sobol index so that further dimensions
// can be queried (eg, for BSDF sampling).  Note that this function uses the
// first 5 dimensions of the sobol sequence, so any further sampling of this
// sobol_index should start at the 6th dim
//
//------------------------------------------------------------------------------
namespace legion
{
__device__ uint64 generateSobolSamples( const uint2&   launch_dim,
                                        const uint2&   pixel_coord,
                                        legion::uint64 sample_index,
                                        float2&        screen_sample,
                                        float2&        lens_sample,
                                        float&         time_sample )
{
    const float2 inv_dim = make_float2( 1.0f ) / 
                           make_float2( launch_dim.x, launch_dim.y );

    screen_sample = make_float2(0.5f );
    legion::uint64 sobol_index;
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
    return optix::make_Ray( origin, direction, 0u, 0.001f, RT_DEFAULT_MAX );
}

__device__ unsigned pointOccluded( float3 p, float3 w_in, float dist )
{
    // TODO: epsilon BS
    ShadowPRD prd;
    prd.occluded   = 0u;
    optix::Ray ray = optix::make_Ray( p, w_in, 1u, 0.0001f, dist - 0.001f );
    rtTrace( legion_top_group, ray, prd );

    return prd.occluded;
}

} // end namespace legion


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
    float3   position;
    //float3   position_object;
    float3   geometric_normal;
    float3   shading_normal;
    float2   texcoord;
};

}

rtCallableProgram( legion::BSDFSample,
                   legionSurfaceSampleBSDF,
                   ( float2, float3 , legion::LocalGeometry ) );

rtCallableProgram( float3, 
                   legionSurfaceEvaluateBSDF,
                   ( float3 , legion::LocalGeometry, float3 ) );

rtCallableProgram( float, 
                   legionSurfacePDF,
                   ( float3, legion::LocalGeometry, float3 ) );

// (w_out, shading_point)
rtCallableProgram( float3, 
                   legionSurfaceEmission,
                   ( float3, legion::LocalGeometry ) ); 

//------------------------------------------------------------------------------
//
// Lighting
//
//------------------------------------------------------------------------------


namespace legion
{

struct LightSample 
{
    legion::LocalGeometry point_on_light;
    float  pdf; 
};

}

// TODO: these will become rtCallableBuffers soon (once I implement in optix)

// This comes from the IGeometry if it is an area light, ILight if it is a
// non-physical light
rtCallableProgram( legion::LightSample,
                   legionLightSample,
                   ( float2, float3 ) ); // ( sample_seed, shading_point )


rtCallableProgram( float3,
                   legionLightEmission,
                   ( float3, legion::LocalGeometry ) ); // (w_in, shading_point)

// TODO: unify env lighting and light source lighting???
rtCallableProgram( float3,
                   legionEnvironmentEvaluate,
                   ( float3 ) ); // ( dir )

#endif // LEGION_OBJECTS_CUDA_COMMON_HPP_
