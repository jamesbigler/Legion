
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


#include <Legion/Common/Math/CUDA/Math.hpp>
#include <Legion/Common/Math/CUDA/Sobol.hpp>
#include <Legion/Objects/Light/CUDA/Light.hpp>
#include <Legion/Objects/Surface/CUDA/Surface.hpp>
#include <Legion/Objects/cuda_common.hpp>


rtDeclareVariable( legion::LocalGeometry, local_geom, attribute local_geom, ); 
rtDeclareVariable( optix::Ray,            ray,        rtCurrentRay, );
rtDeclareVariable( float,                 t_hit,      rtIntersectionDistance, );



RT_PROGRAM
void legionAnyHit()
{
    shadow_prd.hit_p = ray.origin + t_hit * ray.direction;
    shadow_prd.occluded = 1u; 
    rtTerminateRay();
}


RT_PROGRAM
void legionClosestHit() // MIS
{
    float3 radiance = make_float3( 0.0f );

    //
    // Emitted contribution
    //
    float w                     = 1.0f;
    const float3 w_out          = -ray.direction;
    const float  choose_light_p = 1.0f / static_cast<float>( legionLightCount );
    const float3 P = ray.origin + t_hit * ray.direction;
    if( !radiance_prd.count_emitted_light )
    {
        const float light_pdf = legionLightPDF( ray.direction, P ); 
        w = legion::powerHeuristic( radiance_prd.pdf, light_pdf*choose_light_p );
    }
    radiance  = w * legionSurfaceEmission( w_out, local_geom );

    const unsigned sobol_index = radiance_prd.sobol_index;

    // 
    // Indirect lighting
    //
    const float2 bsdf_seed = 
        make_float2( 
                legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ ),
                legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ ) );

    legion::BSDFSample bsdf_sample = 
        legionSurfaceSampleBSDF( bsdf_seed, w_out, local_geom );


    const float  cosine   = optix::dot( bsdf_sample.w_in, local_geom.shading_normal ); // TODO: redundant
    radiance_prd.origin              = P;
    radiance_prd.direction           = bsdf_sample.w_in;
    radiance_prd.attenuation         = bsdf_sample.f_over_pdf;// * cosine;
    radiance_prd.done                = false; 
    radiance_prd.pdf                 = bsdf_sample.pdf; 
    radiance_prd.count_emitted_light = false; 

    
    //
    // direct lighting
    //

    // TODO: If not bsdf.isSingular
    const float    choose_light_seed = legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ );
    const unsigned light_index       = choose_light_seed * legionLightCount;

    const float2 light_seed = 
        make_float2( 
                legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ ),
                legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ ) );

    // TODO fold lightEvaluate into this
    const legion::LightSample light_sample = 
        legion::lightSample( light_index, light_seed, P, local_geom.shading_normal  ); 

    if( light_sample.pdf > 0.0f )
    {
        const float cos_theta = optix::dot( light_sample.w_in, local_geom.shading_normal );
        if( cos_theta > 0.0f ) 
        {
            const float3 w_out    = -ray.direction;
            const float  bsdf_pdf = legionSurfacePDF( w_out, local_geom, light_sample.w_in ); // TODO: fold into evaluate


            if( bsdf_pdf > 0.0f ) // TODO: redundant with above check on dot product
            {

                if( !legion::pointOccluded( P, light_sample.w_in, light_sample.distance ) )
                {
                    const float3 light_radiance = 
                        legion::lightEvaluate( 
                                light_index, 
                                light_sample.w_in, 
                                light_sample.distance,
                                light_sample.normal );
            

                    const float3 w_out  = -ray.direction;
                    const float  weight = legion::powerHeuristic( light_sample.pdf*choose_light_p, bsdf_pdf );
                    const float3 bsdf   = legionSurfaceEvaluateBSDF( w_out, local_geom, light_sample.w_in );


                    radiance +=  light_radiance * bsdf * ( weight  / ( light_sample.pdf*choose_light_p ) );
                }
            }
        }
    }

    //
    // Report result
    // 
    radiance_prd.radiance = radiance;
}


/*
RT_PROGRAM
void legionClosestHit2() // No mis`
{
    float3 radiance = make_float3( 0.0f );

    //
    // Emitted contribution
    //
    if( radiance_prd.count_emitted_light )
    {
        const float3 w_out = -ray.direction;
        radiance  = legionSurfaceEmission( w_out, local_geom );
    }

    // 
    // Indirect lighting
    //
    const unsigned sobol_index = radiance_prd.sobol_index;
    const float2 seed = 
        make_float2( 
            legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ ),
            legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ ) );

    const float3 w_out = -ray.direction;
    legion::BSDFSample bsdf_sample = 
        legionSurfaceSampleBSDF( seed, w_out, local_geom );

    const float3 P = ray.origin + t_hit * ray.direction;

    radiance_prd.origin              = P;
    radiance_prd.direction           = bsdf_sample.w_in;
    radiance_prd.attenuation         = bsdf_sample.f_over_pdf;
    radiance_prd.done                = false; 
    radiance_prd.count_emitted_light = false; 

    //
    // direct lighting
    //
    const unsigned num_lights  = 1;
    for( unsigned i = 0; i < num_lights; ++i )
    {
        const float2 seed = 
            make_float2( 
                legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ ),
                legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ ) );

        const legion::LightSample light_sample = legionLightSample( seed, P ); 
        if( light_sample.pdf > 0.0f )
        {
            float3       w_in       = light_sample.point_on_light.position - P;
            const float  light_dist = optix::length( w_in );
            w_in /= light_dist;

            if( optix::dot( w_in, local_geom.shading_normal ) > 0.0f ) 
            {
                if( !legion::pointOccluded( P, w_in, light_dist ) )
                {
                    const float3 light_col = 
                        legionLightEmission( 
                                -w_in, 
                                light_sample.point_on_light );

                    const float3 w_out = -ray.direction;
                    const float3 bsdf = 
                        legionSurfaceEvaluateBSDF( 
                                w_out, 
                                local_geom, 
                                w_in );

                    radiance +=  light_col * bsdf / light_sample.pdf;
                }
            }
        }
    }

    //
    // Report result
    // 
    radiance_prd.radiance = radiance;
}
*/
