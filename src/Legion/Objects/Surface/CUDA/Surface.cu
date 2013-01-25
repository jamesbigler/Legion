
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
#include <Legion/Common/Math/CUDA/Sobol.hpp>
#include <Legion/Common/Math/CUDA/Math.hpp>


rtDeclareVariable( legion::LocalGeometry, lgeom,    attribute local_geom, ); 
rtDeclareVariable( optix::Ray,            ray,      rtCurrentRay, );
rtDeclareVariable( float,                 t_hit,    rtIntersectionDistance, );



RT_PROGRAM
void legionAnyHit()
{
    shadow_prd.hit_p = ray.origin + t_hit * ray.direction;
    shadow_prd.occluded = 1u; 
    rtTerminateRay();
}


RT_PROGRAM
void legionClosestHit()
{
    float3 result = make_float3( 0.0f );
    const float3 P = ray.origin + t_hit * ray.direction;
    
    legion::LocalGeometry local_geom = lgeom;
    local_geom.position = P;

    //
    // emitted contribution
    //
    if( radiance_prd.count_emitted_light )
    {
        const float3 w_out = -ray.direction;
        result = legionSurfaceEmission( w_out, local_geom );
    }

    //
    // direct lighting
    //
    const int num_lights = 1;

    for( int i = 0; i < num_lights; ++i )
    {
        const unsigned sobol_index = radiance_prd.sobol_index;

        float2 seed = make_float2( 
                legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++),
                legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++) );

        const legion::LightSample light_sample = legionLightSample( seed, P ); 
        if( light_sample.pdf <= 0.0f )
            continue;

        float3       w_in       = light_sample.point_on_light.position - P;
        const float  light_dist = optix::length( w_in );
        w_in /= light_dist;

        // occlusion query
        const float n_dot_wi = optix::dot( w_in, local_geom.shading_normal );
        bool occluded = n_dot_wi <= 0.0f;
        if( !occluded )
            occluded = legion::pointOccluded( P, w_in, light_dist );  

        if( !occluded )
        {
            const float3 light_col = 
                legionLightEmission( -w_in, light_sample.point_on_light );
            const float3 w_out = -ray.direction;
            const float3 bsdf  = 
                legionSurfaceEvaluateBSDF( w_out, local_geom, w_in );
            result += n_dot_wi /
                      light_sample.pdf * 
                      light_col *
                      bsdf;
            if( !legion::finite( result ) )
            {
                printf( "ndw: %f lpdf %f lcol %f %f %f bsdf %f %f %f\n",
                        n_dot_wi,
                        light_sample.pdf,
                        light_col.x,
                        light_col.y,
                        light_col.z,
                        bsdf.x,
                        bsdf.y,
                        bsdf.z );
                printf( "\tw_in: %f %f %f local_geom.sn: %f %f %f\n",
                        w_in.x,
                        w_in.y,
                        w_in.z,
                        local_geom.shading_normal.x,
                        local_geom.shading_normal.y,
                        local_geom.shading_normal.z );
                printf( "\tP: %f %f %f light_p %f %f %f\n",
                        P.x,
                        P.y,
                        P.z,
                        light_sample.point_on_light.position.x,
                        light_sample.point_on_light.position.y,
                        light_sample.point_on_light.position.z );
            }


        //result = make_float3( light_sample.pdf ); 
        //result = make_float3( n_dot_wi ); 
        //result = ( light_col ); 
        }
    }


    //
    // indirect lighting
    //


    radiance_prd.result = result;
}
