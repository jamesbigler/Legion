
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
#include <Legion/Objects/Surface/CUDA/Surface.hpp>
#include <Legion/Common/Math/CUDA/ONB.hpp>
#include <Legion/Common/Math/CUDA/Math.hpp>

#define rtiComment( _a ) asm volatile("call _rti_comment_" #_a " , ();");

rtDeclareVariable( float3, radiance, , );

__device__
float3 diffuseEmitterEmission( legion::LightSample )
{
    return radiance;
}



#include <Legion/Common/Math/CUDA/Math.hpp>
#include <Legion/Common/Math/CUDA/Sobol.hpp>
#include <Legion/Common/Math/CUDA/Rand.hpp>
#include <Legion/Objects/Light/CUDA/Light.hpp>
#include <Legion/Objects/Surface/CUDA/Surface.hpp>
#include <Legion/Objects/cuda_common.hpp>


rtDeclareVariable( legion::LocalGeometry, local_geom, attribute local_geom, ); 
rtDeclareVariable( optix::Ray,            ray,        rtCurrentRay, );
rtDeclareVariable( float,                 t_hit,      rtIntersectionDistance, );

rtDeclareVariable( unsigned, max_diff_depth,  , );
rtDeclareVariable( unsigned, max_spec_depth,  , );

__device__
legion::BSDFSample diffuseEmitterSampleBSDF( float3 seed, float3 w_out, legion::LocalGeometry p )
{
    legion::BSDFSample sample;
    sample.w_in        = make_float3( 0.0f );
    sample.f_over_pdf  = make_float3( 0.0f );
    sample.pdf         = 0.0f;
    sample.is_singular = 1;
    return sample;
}


__device__
float4 diffuseEmitterEvaluateBSDF( float3 w_out, legion::LocalGeometry p, float3 w_in )
{
    return make_float4( 0.0f ); 
}

rtDeclareVariable( float3, center, , );
rtDeclareVariable( float , radius, , );

__device__
float spherePDF( float3 w_in, float3 p )
{
    float3 temp = center - p;

    float d = optix::length( temp );
    float r = radius;
    temp /= d;

    if ( d <= r )
    {
        return 0.0f;
    }

    // internal angle of cone surrounding light seen from viewpoint
    float sin_alpha_max = r / d;
    float cos_alpha_max = sqrtf( 1.0f - sin_alpha_max*sin_alpha_max);

    // check to see if direction misses light
    if( optix::dot( w_in, temp ) < cos_alpha_max )
        return 0.0f;

    // solid angle
    float q = 2.0f*legion::PI*( 1.0f - cos_alpha_max );

    // pdf is one over solid angle
    return 1.0f/q;
}


RT_PROGRAM
void diffuseEmitterClosestHit()
{
    const float last_pdf     = radiance_prd.pdf;
    const bool  last_use_mis = radiance_prd.use_mis_weight;

    /*
    radiance_prd.radiance = local_geom.geometric_normal;
    radiance_prd.radiance = local_geom.shading_normal;
    radiance_prd.done = true;
    return;
    */

    // 
    // Indirect lighting (BSDF sampling)
    //
    {
        legion::LCGRand rand( radiance_prd.rand_seed );
        const unsigned sobol_index = radiance_prd.sobol_index;
        const float3 bsdf_seed = radiance_prd.diff_depth > 0 ?
            make_float3( rand(), rand(), rand() )       :
            make_float3( 
                    legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ ),
                    legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ ),
                    rand()
                    );
        radiance_prd.rand_seed = rand.getSeed();

        const float3 w_out = -ray.direction;
#if 0
        legion::BSDFSample bsdf_sample = 
            legionSurfaceSampleBSDF( bsdf_seed, w_out, local_geom );
#else
        legion::BSDFSample bsdf_sample = 
            diffuseEmitterSampleBSDF( bsdf_seed, w_out, local_geom );
        
#endif
        CHECK_FINITE( bsdf_sample.w_in       );
        CHECK_FINITE( bsdf_sample.f_over_pdf );
        CHECK_FINITE( bsdf_sample.pdf        );


        const float3 P = ray.origin + t_hit * ray.direction;

        if( bsdf_sample.event_type == legion::BSDF_EVENT_SPECULAR ) radiance_prd.spec_depth += 1;
        if( bsdf_sample.event_type == legion::BSDF_EVENT_DIFFUSE  ) radiance_prd.diff_depth += 1;
        radiance_prd.origin         = P;
        radiance_prd.direction      = bsdf_sample.w_in;
        radiance_prd.attenuation    = bsdf_sample.f_over_pdf;
        radiance_prd.pdf            = bsdf_sample.pdf; 
        radiance_prd.use_mis_weight = !bsdf_sample.is_singular; 
        radiance_prd.done           = bsdf_sample.pdf <= 0.0 ||
                                      radiance_prd.spec_depth>max_spec_depth ||
                                      radiance_prd.diff_depth>max_diff_depth;
        
        /*
        radiance_prd.radiance = bsdf_sample.f_over_pdf;
        radiance_prd.done = true; 
        return;
        */
    }

    float3 radiance = make_float3( 0.0f );
    const float choose_light_p = 1.0f / static_cast<float>( legion::lightCount() );

    rtiComment(diffuseEmitter_emitted_contribution);
    //
    // Emitted contribution
    //
    {
        legion::LightSample light_sample;
        light_sample.w_in     = -ray.direction; 
        light_sample.distance = t_hit; 
        light_sample.normal   = local_geom.shading_normal; 
        light_sample.pdf      = 1.0f;

#if 0
        radiance = legionSurfaceEmission( light_sample );
#else
        radiance = diffuseEmitterEmission( light_sample );
#endif
        CHECK_FINITE( radiance );

        if( last_use_mis && !legion::isBlack( radiance  ))
        {
            const float3 P           = ray.origin;
#if 0
            const float  light_pdf  = legionLightPDF( light_sample.w_in, P )*choose_light_p;
#else
            const float  light_pdf  = spherePDF( light_sample.w_in, P )*choose_light_p;
#endif
            const float  bsdf_pdf   = last_pdf; 
            const float  mis_weight = legion::powerHeuristic(
                                          bsdf_pdf, light_pdf );
            CHECK_FINITE( light_pdf  );
            CHECK_FINITE( bsdf_pdf   );
            CHECK_FINITE( mis_weight );

            radiance *= mis_weight;
        }
    }

    rtiComment(diffuseEmitter_direct_lighting);

    //
    // Direct lighting (next event estimation)
    //
    if( 0 && radiance_prd.use_mis_weight )
    {
        const unsigned sobol_index = radiance_prd.sobol_index;
        const unsigned light_index = radiance_prd.light_index;

        const float2 light_seed = 
            make_float2( 
                    legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ ),
                    legion::Sobol::gen( sobol_index, radiance_prd.sobol_dim++ )
                    );

        const float3 w_out = -ray.direction;
        const float3 P = ray.origin + t_hit * ray.direction;
        const float3 N = 
            optix::faceforward( 
                local_geom.shading_normal, w_out, local_geom.geometric_normal
                );

        const legion::LightSample light_sample = 
            legion::lightSample( light_index, light_seed, P, N ); 
        CHECK_FINITE( light_sample.w_in     );
        CHECK_FINITE( light_sample.pdf      );
        CHECK_FINITE( light_sample.distance );
        CHECK_FINITE( light_sample.normal   );

        const float3 w_in      = light_sample.w_in;
        const float  light_pdf = light_sample.pdf*choose_light_p;
        const float  cos_theta = optix::dot( w_in, N );

        if( light_pdf > 0.0f && 
            cos_theta > 0.0f && 
            !legion::pointOccluded( P, w_in, light_sample.distance ) )
        {
            rtiComment(diffuseEmitterEvaluateBSDF_start);
#if 0
            const float4 bsdf = legionSurfaceEvaluateBSDF( 
                w_out, local_geom, w_in );
#else
            const float4 bsdf = diffuseEmitterEvaluateBSDF( 
                w_out, local_geom, w_in );
#endif
            rtiComment(diffuseEmitterEvaluateBSDF_end);

            const float  bsdf_pdf = bsdf.w;
            const float3 bsdf_val = make_float3( bsdf );
            CHECK_FINITE( bsdf_val );
            CHECK_FINITE( bsdf_pdf );

            if( bsdf_pdf > 0.0f )
            {
                const float  mis_weight = legion::powerHeuristic(
                                              light_pdf, bsdf_pdf );
                const float3 atten      = bsdf_val*( mis_weight / light_pdf );
                CHECK_FINITE( mis_weight );
                CHECK_FINITE( atten      );


                const float3 light_radiance = 
                    legion::lightEvaluate( light_index, light_sample );
                CHECK_FINITE( light_radiance );

                radiance += light_radiance*atten;
            }
        }
    }

    //
    // Report result
    // 
    radiance_prd.radiance = radiance;
}
