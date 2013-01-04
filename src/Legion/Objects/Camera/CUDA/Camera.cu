
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

#include <Legion/Objects/Camera/CUDA/Camera.hpp>
#include <Legion/Objects/Surface/CUDA/Surface.hpp>
#include <optixu/optixu_math_namespace.h>
#include <optix.h>

rtDeclareVariable( uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable( uint2, launch_dim,   rtLaunchDim, );

rtDeclareVariable( rtObject, legion_top_group, , );

rtBuffer<float4, 2> legion_output_buffer;

rtCallableProgram( legion::RayGeometry,
                   legionCameraCreateRay, 
                   (float2, float2, float ) );

RT_PROGRAM void Camera()
{
    const float sx = static_cast<float>( launch_index.x ) /
                     static_cast<float>( launch_dim.x );
    const float sy = static_cast<float>( launch_index.y ) /
                     static_cast<float>( launch_dim.y );

    const float  time_seed   = 0.0f;
    const float2 lens_seed   = make_float2( 0.0f, 0.0f );
    const float2 screen_seed = make_float2( sx, sy );

    legion::RayGeometry rg = legionCameraCreateRay( lens_seed,
                                                    screen_seed,
                                                    time_seed );

    RadiancePRD prd;
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

    legion_output_buffer[ launch_index ] = make_float4( prd.result, 1.0f );
}






