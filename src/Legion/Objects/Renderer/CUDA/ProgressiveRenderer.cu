
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

#include <Legion/Objects/cuda_common.hpp>
#include <Legion/Common/Math/Sobol.hpp>
#include <optixu/optixu_math_namespace.h>


rtDeclareVariable( uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable( uint2, launch_dim,   rtLaunchDim, );

rtDeclareVariable( unsigned, sample_index, , );

rtBuffer<float4, 2> output_buffer;


RT_PROGRAM void progressiveRendererRayGen()
{
    const float2 inv_dim = make_float2( 1.0f ) / 
                           make_float2( launch_dim.x, launch_dim.y );

    const float sx = static_cast<float>( launch_index.x ) * inv_dim.x;
    const float sy = static_cast<float>( launch_index.y ) * inv_dim.y;

    // TODO: use Alex's code for grabbing the Ith sample within a given pixel
    //       so we can query a screen seed for this pixel directly
    const float  time_seed   = 0.0f;
    const float2 lens_seed   = legion::Sobol::genLensSample( sample_index );
    const float2 pixel_seed  = legion::Sobol::genPixelSample( sample_index );
    const float2 screen_seed = make_float2( sx + pixel_seed.x * inv_dim.x ,
                                            sy + pixel_seed.y * inv_dim.y );
    
    legion::RayGeometry rg = legionCameraCreateRay( lens_seed,
                                                    screen_seed,
                                                    time_seed );

    legion::RadiancePRD prd;
    prd.result = make_float3( 0.0f );
    prd.importance = 1.0f;
    prd.depth = 0u;

    optix::Ray ray = optix::make_Ray( 
            rg.origin,
            rg.direction,
            0u,
            0.0f,
            RT_DEFAULT_MAX );
    rtTrace( legion_top_group, ray, prd );

    const float4 prev   = output_buffer[ launch_index ];
    const float4 cur    = make_float4( prd.result, 1.0f );
    const float4 result = optix::lerp( prev, cur, 1.0f / static_cast<float>( sample_index+1 ) );
    output_buffer[ launch_index ] = result;
}






