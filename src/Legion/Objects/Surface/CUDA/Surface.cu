
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

/// \file IGeometry.hpp
/// Pure virtual interface for Geometry classes


#include <Legion/Objects/cuda_common.hpp>


rtDeclareVariable( legion::LocalGeometry, local_geom, attribute local_geom, ); 
rtDeclareVariable( optix::Ray,            ray,        rtCurrentRay, );
rtDeclareVariable( float,                 t_hit,      rtIntersectionDistance, );



RT_CALLABLE_PROGRAM 
float3 legionDefaultEmission( float3, legion::LocalGeometry )
{
    return make_float3( 0.0f );
}


RT_PROGRAM
void legionAnyHit()
{
    shadow_prd.attenuation = make_float3( 0.0f );
    rtTerminateRay();
}


RT_PROGRAM
void legionClosestHit()
{
    float3 result = make_float3( 0.0f );

    //
    // emitted contribution
    //
    if( radiance_prd.count_emitted_light )
    {
        const float3 w_out = -ray.direction;
        result += legionEmission( w_out, local_geom );
    }

    //
    // direct lighting
    //
    const int num_lights = 1;

    for( int i = 0; i < num_lights; ++i )
    {
        const float3 light_pos = make_float3( 1.0f, 3.0f, 0.0f );
        const float3 light_col = make_float3( 1.0f, 1.0f, 1.0f );

        bool occluded = false; // Backface and shadow trace
        if( !occluded )
        {
            const float3 hit_point = ray.origin + t_hit * ray.direction;
            const float3 w_out     = -ray.direction;
            const float3 w_in      = light_pos - hit_point;
            
            result += legionEvaluateBSDF( w_out, local_geom, w_in );
        }


        // occlusion query

    }


    //
    // indirect lighting
    //


    radiance_prd.result = result; 
}
