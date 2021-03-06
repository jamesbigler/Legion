
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

#if 0
#    define WHERESTR  "[file %s, line %d]: "
#    define WHEREARG  __FILE__, __LINE__
#    define DEBUGPRINT( x )      printf( WHERESTR x, WHEREARG )
#    define CHECK_FINITE( x )  \
         if(!legion::finite(x)) DEBUGPRINT("Nonfinite value found: '" #x "'\n");
#else
#    define CHECK_FINITE( x )
#endif

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


struct LocalGeometry 
{
    float3   position;
    float3   geometric_normal;
    float3   shading_normal;
    float2   texcoord;
};

//------------------------------------------------------------------------------
//
// this struct is to be used as the per-ray-data for radiance rays
//
//------------------------------------------------------------------------------

// TODO: pack this
struct RadiancePRD 
{
    float3   origin;
    float3   direction;
    float3   radiance;
    float3   attenuation;
    float    pdf;
    unsigned long long sobol_index;
    unsigned sobol_dim;
    unsigned rand_seed;
    unsigned light_index;             ///////////////////////////////
    short    spec_depth;
    short    diff_depth;
    short    use_mis_weight;
    short    done;
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
    if( screen_sample.x < 0.0f || screen_sample.x > 1.0f ||
        screen_sample.y < 0.0f || screen_sample.y > 1.0f )
        screen_sample = make_float2( 0.0f );
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


namespace legion
{

}


#endif // LEGION_OBJECTS_CUDA_COMMON_HPP_
