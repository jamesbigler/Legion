
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

rtDeclareVariable( float3, diff_reflectance, , );
rtDeclareVariable( float3, spec_reflectance, , );
rtDeclareVariable( float,  alpha_u, , );
rtDeclareVariable( float,  alpha_v, , );
rtDeclareVariable( float,  diffuse_weight, , );


static __device__ __inline__
float wardExpTerm( float inv_ax, float inv_ay, float3 H, const legion::ONB uvw )
{
    float h_dot_n = optix::dot(H, uvw.w());
    float h_dot_u = optix::dot(H, uvw.u());
    float h_dot_v = optix::dot(H, uvw.v());

    float u_coeff = h_dot_u * inv_ax;
    float v_coeff = h_dot_v * inv_ay;

    return expf( -( u_coeff * u_coeff + v_coeff * v_coeff) / (h_dot_n * h_dot_n));
}


static __device__ __inline__
float diffusePDF( float3 w_out, float3 normal, float3 w_in )
{
    return legion::ONE_DIV_PI*optix::dot( normal, w_in );
}


// static __device__ __inline__
// float3 diffuseEval( float3 w_out, float3 normal, float3 w_in )
// {
//     return legion::ONE_DIV_PI*optix::dot( normal, w_in) * diff_reflectance;
// }


static __device__ __inline__
float4 diffuseEvalPDF( float3 w_out, float3 normal, float3 w_in )
{
    const float g = legion::ONE_DIV_PI*optix::dot( normal, w_in);
    return make_float4( diff_reflectance*g, g );
}


static __device__ __inline__
float specularPDF( float3 w_out, float3 normal, float3 w_in )
{
    float cos_in  = optix::dot( w_in,  normal );
    float cos_out = optix::dot( w_out, normal );
    if( cos_in <= 0.0f || cos_out <= 0.0f )
        return 0.0f;

    legion::ONB onb( normal );
    const float3 H       = optix::normalize( w_out + w_in );
    const float  n_dot_h = optix::dot( normal, H );
    const float  inv_alpha_u = 1.0f / alpha_u;
    const float  inv_alpha_v = 1.0f / alpha_v;

    const float  ward_exp_term = wardExpTerm( inv_alpha_u, inv_alpha_v, H, onb);
    const float  spec_pdf = 
        0.25f * legion::ONE_DIV_PI * inv_alpha_u * inv_alpha_v /
        ( optix::dot(H, w_out) * n_dot_h*n_dot_h*n_dot_h ) *
        ward_exp_term;

    return spec_pdf;
}


// static __device__ __inline__
// float3 specularEval( float3 w_out, float3 normal, float3 w_in )
// {
//     legion::ONB  onb( normal );
//     const float  cos_in     = optix::dot( w_in,  normal );
//     const float  cos_out    = optix::dot( w_out, normal );
// 
//     if( cos_in <= 0.0f || cos_out <= 0.0f )
//         return make_float3( 0.0f );
//     const float3 H           = optix::normalize( w_out + w_in );
//     const float  inv_alpha_u = 1.0f / alpha_u;
//     const float  inv_alpha_v = 1.0f / alpha_v;
//     const float  spec_coeff = 
//         cos_in * legion::ONE_DIV_PI *
//         0.25f * inv_alpha_u * inv_alpha_v / sqrtf( cos_in*cos_out ) *
//         wardExpTerm( inv_alpha_u, inv_alpha_v, H, onb );
// 
//     return spec_coeff * spec_reflectance;
// }

static __device__ __inline__
float4 specularEvalPDF( float3 w_out, float3 normal, float3 w_in )
{
    legion::ONB  onb( normal );
    const float  cos_in  = optix::dot( w_in,  normal );
    const float  cos_out = optix::dot( w_out, normal );
    if( cos_in <= 0.0f || cos_out <= 0.0f )
        return make_float4( 0.0f );

    const float3 H           = optix::normalize( w_out + w_in );
    const float  inv_alpha_u = 1.0f / alpha_u;
    const float  inv_alpha_v = 1.0f / alpha_v;

    const float  ward_exp_term = wardExpTerm( inv_alpha_u, inv_alpha_v, H, onb);
    const float  spec_coeff    
        = 0.25f * cos_in * legion::ONE_DIV_PI *
          inv_alpha_u * inv_alpha_v / sqrtf( cos_in*cos_out ) *
          ward_exp_term;

    const float  n_dot_h  = optix::dot( normal, H );
    const float  spec_pdf = 0.25f * legion::ONE_DIV_PI * 
                            inv_alpha_u * inv_alpha_v /
                            ( optix::dot(H, w_out) * n_dot_h*n_dot_h*n_dot_h ) *
                            ward_exp_term;

    return make_float4( spec_coeff * spec_reflectance, spec_pdf );
}


static __device__ __inline__
float4 wardEvalPDF( float3 w_out, float3 N, float3 w_in, float diff_weight )
{
    const float4 spec     = specularEvalPDF( w_out, N, w_in );
    const float  spec_pdf = spec.w; 
    const float3 spec_val = make_float3( spec ); 

    const float4 diff     = diffuseEvalPDF( w_out, N, w_in );
    const float  diff_pdf = diff.w; 
    const float3 diff_val = make_float3( diff ); 

    const float  pdf      = optix::lerp( spec_pdf, diff_pdf, diff_weight );

    return make_float4( spec_val + diff_val, pdf );
}



__device__
legion::BSDFSample wardSampleBSDF( 
        float3 seed,
        float3 w_out,
        legion::LocalGeometry p )
{
    const float3 N = optix::faceforward( 
            p.shading_normal, w_out, p.geometric_normal
            );

    legion::ONB onb( N );
    legion::BSDFSample sample;
    if( seed.z < diffuse_weight )
    {

        // sample hemisphere with cosine density by uniformly sampling
        // unit disk and projecting up to hemisphere
        float2 on_disk( legion::squareToDisk( make_float2( seed ) ) );
        const float x = on_disk.x;
        const float y = on_disk.y;
        float z = 1.0f - x*x -y*y;

        z = z > 0.0f ? sqrtf( z ) : 0.0f;

        // Transform into world space
        sample.w_in = onb.inverseTransform( make_float3( x, y, z ) );

        // calculate pdf
        const float4 ward = wardEvalPDF( w_out, N, sample.w_in, diffuse_weight);
        sample.pdf        = ward.w;
        sample.f_over_pdf = sample.pdf > 0.00001f            ? 
                            make_float3( ward )/ sample.pdf  :
                            make_float3( 0.0f );
        sample.is_singular = 0u; 
        sample.event_type = legion::BSDF_EVENT_DIFFUSE;
    }
    else
    {
        // Sample specular lobe (see Walter's TR for info)
        float xi1 = seed.x; 
        float xi2 = seed.y; 

        float phi_h;
        if (xi1 < 0.25f )
        {
            xi1 *= 4.0f;
            phi_h = atanf( alpha_v * tanf( 0.5f * legion::PI * xi1) / alpha_u);
        }
        else if (xi1 < 0.5f)
        {
            xi1 = 4.0f * (xi1 - 0.25f);
            phi_h = legion::PI - atanf(alpha_v * tanf(0.5f * legion::PI * xi1) /
                    alpha_u);
        }
        else if (xi1 < 0.75f)
        {
            xi1 = 4.0f * (xi1 - 0.5f);
            phi_h = legion::PI + atanf(alpha_v * tanf(0.5f * legion::PI * xi1) /
                    alpha_u);
        }
        else
        {
            xi1 = 4.0f * (xi1 - 0.75f);
            phi_h = legion::PI*2.0f - 
                    atanf(alpha_v * tanf( 0.5f * legion::PI * xi1) / alpha_u);
        }

        float cos_phi_h = cosf( phi_h );
        float sin_phi_h = sinf( phi_h );
        float x_coeff   = cos_phi_h / alpha_u;
        float y_coeff   = sin_phi_h / alpha_v;

        float theta_h = 
            atanf( sqrtf( -logf( xi2 ) / 
                        ( x_coeff*x_coeff + y_coeff*y_coeff ) ) );

        float sin_theta_h = sinf( theta_h );
        float cos_theta_h = 1.0f - sin_theta_h*sin_theta_h;
        if (cos_theta_h > 0.0f)
            cos_theta_h = sqrtf( cos_theta_h );
        else
            cos_theta_h = 0.0f;

        float x = cos_phi_h * sin_theta_h;
        float y = sin_phi_h * sin_theta_h;
        float z = cos_theta_h; 


        const float3 H    = onb.inverseTransform( make_float3( x, y, z ) );
        sample.w_in       = optix::reflect( -w_out, H );
        
        const float4 ward = wardEvalPDF( w_out, N, sample.w_in, diffuse_weight);
        sample.pdf        = ward.w;
        sample.f_over_pdf = sample.pdf > 0.00001f            ? 
                            make_float3( ward )/ sample.pdf  :
                            make_float3( 0.0f );
        sample.is_singular = 0u; 
        sample.event_type  = legion::BSDF_EVENT_SPECULAR;
    }
        
    CHECK_FINITE( sample.w_in       );
    CHECK_FINITE( sample.f_over_pdf );
    CHECK_FINITE( sample.pdf        );
    return sample;
}


__device__
  float4 wardEvaluateBSDF(
        float3 w_out,
        legion::LocalGeometry p,
        float3 w_in )
{
    const float3 N = optix::faceforward( 
            p.shading_normal, w_out, p.geometric_normal
            );
    const float  cos_in   = optix::dot( w_in,  N );
    const float  cos_out  = optix::dot( w_out, N );
    if( cos_in <= 0.0f || cos_out <= 0.0f )
        return make_float4( 0.0f );

    const float4 val = wardEvalPDF( w_out, N, w_in, diffuse_weight);

    CHECK_FINITE( val );
    return val;
}


__device__
float wardPDF( float3 w_out, legion::LocalGeometry p, float3 w_in )
{
    const float3 N = optix::faceforward( 
            p.shading_normal, w_out, p.geometric_normal
            );
    const float  diffuse_pdf  = diffusePDF( w_out, N, w_in );
    const float  specular_pdf = specularPDF( w_out, N, w_in );
    const float  pdf          = optix::lerp( specular_pdf,
                                             diffuse_pdf,
                                             diffuse_weight );

    CHECK_FINITE( pdf );
    return pdf;
}

__device__
float3 wardSurfaceEmission( legion::LightSample )
{
    return make_float3( 0.0f );
}

rtDeclareVariable( float3, center, , );
rtDeclareVariable( float , radius, , );

struct SampleReporter
{
    __device__ __inline__ SampleReporter( legion::LightSample& sample_ ) : sample( sample_ ) {}

    __device__ __inline__ bool check_t( float t )
    { return t > 0.0001f; }

    __device__ __inline__ bool report ( float t, float3 normal ) 
    {
        sample.distance = t;
        sample.normal   = normal;
        return true;
    }

    legion::LightSample& sample;
};


template <typename Reporter>
static __device__ __inline__
bool sphereIntersectImpl( 
        float3 origin,
        float3 direction, 
        float3 center, 
        float  radius,
        Reporter& reporter )
{
    float3 O = origin - center;
    float3 D = direction;

    float b = optix::dot(O, D);
    float c = optix::dot(O, O)-radius*radius;
    float disc = b*b-c;
    
    bool intersection_found = false;

    if(disc > 0.0f)
    {
        float sdisc = sqrtf(disc);
        float root1 = (-b - sdisc);


        float root11 = 0.0f;

        // refine root1
        if( fabsf(root1) > 10.f * radius )
        {
            float3 O1 = O + root1 * direction;
            b = optix::dot(O1, D);
            c = optix::dot(O1, O1) - radius*radius;
            disc = b*b - c;

            if(disc > 0.0f)
            {
                sdisc = sqrtf(disc);
                root11 = (-b - sdisc);
            }
        }

        const float t = root1 + root11;
        if( reporter.check_t( t ) ) 
        {
            const float3 normal = (O + t*D)/radius;
            intersection_found = reporter.report( t, normal );
        } 

        if( !intersection_found )
        {
            const float t = (-b + sdisc) +  root1;
            if( reporter.check_t( t ) ) 
            {
                const float3 normal = (O + t*D)/radius;
                intersection_found = reporter.report( t, normal );
            }
        }
    }

    return intersection_found;
}

__device__
legion::LightSample sphereSample( float2 sample_seed, float3 shading_point, float3 shading_normal )
{
    legion::LightSample sample;
    sample.pdf = 0.0f;

    float3 temp = center - shading_point;
    float d = optix::length( temp );
    temp /= d;
    
    if ( d <= radius )
        return sample;

    // internal angle of cone surrounding light seen from viewpoint
    float sin_alpha_max = (radius / d);
    float cos_alpha_max = sqrtf( 1.0f - sin_alpha_max*sin_alpha_max );

    float q    = 2.0f*legion::PI*( 1.0f - cos_alpha_max ); // solid angle
    sample.pdf =  1.0f/q;                          // pdf is one / solid angle

    const float phi       = 2.0f*legion::PI*sample_seed.x;
    const float cos_theta = 1.0f - sample_seed.y * ( 1.0f - cos_alpha_max );
    const float sin_theta = sqrtf( 1.0f - cos_theta*cos_theta );
    const float cos_phi = cosf( phi );
    const float sin_phi = sinf( phi );

    legion::ONB uvw( temp );
    sample.w_in = optix::normalize( make_float3( cos_phi*sin_theta, sin_phi*sin_theta, cos_theta) );
    sample.w_in = uvw.inverseTransform( sample.w_in );

    SampleReporter reporter( sample );
    if( !sphereIntersectImpl<SampleReporter>(
                shading_point,
                sample.w_in,
                center,
                radius,
                reporter ) )
        sample.pdf = 0.0f;

    return sample;
}

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


RT_PROGRAM
void wardClosestHit()
{
#if 0
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
            wardSampleBSDF( bsdf_seed, w_out, local_geom );
        
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

    float3 radiance3 = make_float3( 0.0f );
    const float choose_light_p = 1.0f / static_cast<float>( legion::lightCount() );

    rtiComment(ward_emitted_contribution);
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
        radiance3 = legionSurfaceEmission( light_sample );
#else
        radiance3 = wardSurfaceEmission( light_sample );
#endif
        CHECK_FINITE( radiance3 );

        if( last_use_mis && !legion::isBlack( radiance3  ))
        {
            const float3 P           = ray.origin;
            const float  light_pdf  = legionLightPDF( light_sample.w_in, P )*choose_light_p;
            const float  bsdf_pdf   = last_pdf; 
            const float  mis_weight = legion::powerHeuristic(
                                          bsdf_pdf, light_pdf );
            CHECK_FINITE( light_pdf  );
            CHECK_FINITE( bsdf_pdf   );
            CHECK_FINITE( mis_weight );

            radiance3 *= mis_weight;
        }
    }

    rtiComment(ward_direct_lighting);

    //
    // Direct lighting (next event estimation)
    //
    if( radiance_prd.use_mis_weight )
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

#if 0
        const legion::LightSample light_sample = 
            legion::lightSample( light_index, light_seed, P, N );
#else
        const legion::LightSample light_sample =
            sphereSample( light_seed, P, N );
#endif
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
            rtiComment(wardEvaluateBSDF_start);
#if 0
            const float4 bsdf = legionSurfaceEvaluateBSDF( 
                w_out, local_geom, w_in );
#else
            const float4 bsdf = wardEvaluateBSDF( 
                w_out, local_geom, w_in );
#endif
            rtiComment(wardEvaluateBSDF_end);

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

#if 0
                const float3 light_radiance = 
                    legion::lightEvaluate( light_index, light_sample );
#else
                const float3 light_radiance = 
                    diffuseEmitterEmission( light_sample );
#endif
                CHECK_FINITE( light_radiance );

                radiance3 += light_radiance*atten;
            }
        }
    }

    //
    // Report result
    // 
    radiance_prd.radiance = radiance3;
#else
    radiance_prd.radiance = (optix::normalize(local_geom.shading_normal) + make_float3(1.f))*0.5f;
    radiance_prd.done   = 1;
#endif
}
