
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
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optix.h>

rtDeclareVariable( optix::Matrix4x4, camera_to_world, , );
rtDeclareVariable( float           , focal_distance , , );
rtDeclareVariable( float           , aperture_radius, , );
rtDeclareVariable( float4          , view_plane     , , );


RT_CALLABLE_PROGRAM
legion::RayGeometry thinLensCreateRay(
        float2  aperture_sample,
        float2  screen_sample,
        float   time )
{
    const float2 disk_sample = optix::square_to_disk( aperture_sample );
    legion::RayGeometry r;
    r.origin      = make_float3( aperture_radius * disk_sample, 0.0f );
    r.direction.x = optix::lerp( view_plane.x, view_plane.y, screen_sample.x );
    r.direction.y = optix::lerp( view_plane.z, view_plane.w, screen_sample.y );
    r.direction.z = -focal_distance; 
    r.direction = optix::normalize( r.direction );

    // TODO: transform
    return r;
}

/*
// TODO: this is the function signature i want to use -- waiting for optix
//       bug fix 
RT_CALLABLE_PROGRAM
void thinLensCreateRay(
        float2  aperture_sample,
        float2  screen_sample,
        float   time,
        float3& origin,
        float3& direction
        )
{
    const float2 disk_sample = optix::square_to_disk( aperture_sample );
    origin      = make_float3( aperture_radius * disk_sample, 0.0f );
    direction.x = optix::lerp( view_plane.x, view_plane.y, screen_sample.x );
    direction.y = optix::lerp( view_plane.z, view_plane.w, screen_sample.y );
    direction.z = -focal_distance; 
    direction   = optix::normalize( direction );

    // TODO: transform
}
*/

