
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
    const float  choose_light_p = 1.0f / static_cast<float>( legionLightCount );

    //
    // Emitted contribution
    //
    {
        const float3 w_in = ray.direction;

        radiance = legionSurfaceEmission( w_in, local_geom );

        if( !radiance_prd.count_emitted_light && !legion::isBlack( radiance  ))
        {
            const float3 P          = ray.origin;
            const float  light_pdf  = legionLightPDF( w_in, P )*choose_light_p;
            const float  bsdf_pdf   = radiance_prd.pdf;
            const float  mis_weight = legion::powerHeuristic( bsdf_pdf,
                                                              light_pdf );
            radiance *= mis_weight;
        }
    }


    // 
    // Indirect lighting
    //
    {
        const unsigned sobol_index = radiance_prd.sobol_index;
        const float2 bsdf_seed = 
            make_float2( 
                    legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ ),
                    legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ )
                    );

        const float3 w_out = -ray.direction;
        legion::BSDFSample bsdf_sample = 
            legionSurfaceSampleBSDF( bsdf_seed, w_out, local_geom );

        const float3 P = ray.origin + t_hit * ray.direction;

        radiance_prd.origin              = P;
        radiance_prd.direction           = bsdf_sample.w_in;
        radiance_prd.attenuation         = bsdf_sample.f_over_pdf;
        radiance_prd.pdf                 = bsdf_sample.pdf; 
        radiance_prd.done                = false; 
        radiance_prd.count_emitted_light = false; 
    }

    
    //
    // direct lighting
    //
    {
        const unsigned sobol_index = radiance_prd.sobol_index;
        const float choose_light_seed =
            legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ );

        const unsigned light_index = choose_light_seed * legionLightCount;

        const float2 light_seed = 
            make_float2( 
                    legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ ),
                    legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ )
                    );

        const float3 P = ray.origin + t_hit * ray.direction;
        const float3 N = local_geom.shading_normal;
        const legion::LightSample light_sample = 
            legion::lightSample( light_index, light_seed, P, N ); 

        const float3 w_in      = light_sample.w_in;
        const float  light_pdf = light_sample.pdf*choose_light_p;
        const float  cos_theta = optix::dot( w_in, N );

        if( light_pdf > 0.0f && 
            cos_theta > 0.0f && 
            !legion::pointOccluded( P, w_in, light_sample.distance ) )
        {
            const float3 w_out = -ray.direction;
            const float4 bsdf  = legionSurfaceEvaluateBSDF( 
                    w_out, local_geom, w_in );
            const float  bsdf_pdf = bsdf.w;
            const float3 bsdf_val = make_float3( bsdf );

            if( bsdf_pdf > 0.0f )
            {
                const float  weight = legion::powerHeuristic( light_pdf, bsdf_pdf );
                const float3 atten  = bsdf_val*( weight / ( light_pdf ) );
                const float3 light_radiance = 
                    legion::lightEvaluate( 
                            light_index, 
                            light_sample.w_in, 
                            light_sample.distance,
                            light_sample.normal );

                radiance += light_radiance*atten;
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
void legionClosestHit() // No mis`
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
    {
        const unsigned sobol_index = radiance_prd.sobol_index;
        const float choose_light_seed =
            legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ );

        const unsigned light_index = choose_light_seed * legionLightCount;

        const float2 seed = 
            make_float2( 
                    legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ ),
                    legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ ) );

        const float3 N = local_geom.shading_normal;
        const legion::LightSample light_sample = legion::lightSample( light_index, seed, P, N ); 
        if( light_sample.pdf > 0.0f )
        {
            const float3 w_in       = light_sample.w_in;
            const float  light_dist = light_sample.distance; 

            if( optix::dot( w_in, local_geom.shading_normal ) > 0.0f ) 
            {
                if( !legion::pointOccluded( P, w_in, light_dist ) )
                {
                    const float3 light_radiance = 
                        legion::lightEvaluate( 
                                light_index, 
                                light_sample.w_in, 
                                light_sample.distance,
                                light_sample.normal );


                    const float3 w_out = -ray.direction;
                    const float4 bsdf = 
                        legionSurfaceEvaluateBSDF( 
                                w_out, 
                                local_geom, 
                                w_in );

                    radiance +=  light_radiance * make_float3( bsdf ) / light_sample.pdf;
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
