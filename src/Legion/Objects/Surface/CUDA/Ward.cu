
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

    const float  ward_exp_term = wardExpTerm( inv_alpha_u, inv_alpha_v, H, onb );
    const float  spec_pdf      = 0.25f * legion::ONE_DIV_PI * inv_alpha_u * inv_alpha_v /
                                 ( optix::dot(H, w_out) * n_dot_h*n_dot_h*n_dot_h ) *
                                 ward_exp_term;

    return spec_pdf;
}


static __device__ __inline__
float3 diffuseEval( float3 w_out, float3 normal, float3 w_in )
{
    return optix::dot( w_in, normal )*legion::ONE_DIV_PI * diff_reflectance;
}


static __device__ __inline__
float3 specularEval( float3 w_out, float3 normal, float3 w_in )
{
    legion::ONB  onb( normal );
    const float  cos_in     = optix::dot( w_in,  normal );
    const float  cos_out    = optix::dot( w_out, normal );

    if( cos_in <= 0.0f || cos_out <= 0.0f )
        return make_float3( 0.0f );
    const float3 H           = optix::normalize( w_out + w_in );
    const float  inv_alpha_u = 1.0f / alpha_u;
    const float  inv_alpha_v = 1.0f / alpha_v;
    const float  spec_coeff = cos_in * legion::ONE_DIV_PI *
                              0.25f * inv_alpha_u * inv_alpha_v / sqrtf( cos_in*cos_out ) *
                              wardExpTerm( inv_alpha_u, inv_alpha_v, H, onb );

    return spec_coeff * spec_reflectance;
}

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

    const float  ward_exp_term = wardExpTerm( inv_alpha_u, inv_alpha_v, H, onb );
    const float  spec_coeff    = 0.25f * cos_in * legion::ONE_DIV_PI *
                                 inv_alpha_u * inv_alpha_v / sqrtf( cos_in*cos_out ) *
                                 ward_exp_term;

    const float  n_dot_h       = optix::dot( normal, H );
    const float  spec_pdf      = 0.25f * legion::ONE_DIV_PI * inv_alpha_u * inv_alpha_v /
                                 ( optix::dot(H, w_out) * n_dot_h*n_dot_h*n_dot_h ) *
                                 ward_exp_term;

    return make_float4( spec_coeff * spec_reflectance, spec_pdf );
}


RT_CALLABLE_PROGRAM
legion::BSDFSample wardSampleBSDF( 
        float3 seed,
        float3 w_out,
        legion::LocalGeometry p )
{
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
        legion::ONB onb( p.shading_normal );
        sample.w_in = onb.inverseTransform( make_float3( x, y, z ) );

        // calculate pdf
        const float3 N = p.shading_normal;

        const float  diffuse_pdf  = z * legion::ONE_DIV_PI;
        const float  specular_pdf = specularPDF( w_out, N, sample.w_in );
        sample.pdf = optix::lerp( specular_pdf, diffuse_pdf, diffuse_weight );

        const float3 diff_val = diffuseEval( w_out, N, sample.w_in );
        const float3 spec_val = specularEval( w_out, N, sample.w_in );
        sample.f_over_pdf = sample.pdf > 0.00001f            ? 
                            (diff_val+spec_val) / sample.pdf :
                            make_float3( 0.0f );
        sample.is_specular = 0u; 
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
            phi_h = legion::PI - atanf(alpha_v * tanf(0.5f * legion::PI * xi1) / alpha_u);
        }
        else if (xi1 < 0.75f)
        {
            xi1 = 4.0f * (xi1 - 0.5f);
            phi_h = legion::PI + atanf(alpha_v * tanf(0.5 * legion::PI * xi1) / alpha_u);
        }
        else
        {
            xi1 = 4.0f * (xi1 - 0.75f);
            phi_h = legion::PI*2.0f - atanf(alpha_v * tanf( 0.5f * legion::PI * xi1) / alpha_u);
        }

        float cos_phi_h = cosf( phi_h );
        float sin_phi_h = sinf( phi_h );
        float x_coeff   = cos_phi_h / alpha_u;
        float y_coeff   = sin_phi_h / alpha_v;

        float theta_h   = atanf( sqrtf( -logf( xi2 ) / ( x_coeff*x_coeff + y_coeff*y_coeff ) ) );
        float sin_theta_h = sinf( theta_h );
        float cos_theta_h = 1.0f - sin_theta_h*sin_theta_h;
        if (cos_theta_h > 0.0f)
            cos_theta_h = sqrtf( cos_theta_h );
        else
            cos_theta_h = 0.0f;

        float x = cos_phi_h * sin_theta_h;
        float y = sin_phi_h * sin_theta_h;
        float z = cos_theta_h; 


        legion::ONB onb( p.shading_normal );
        float3 H = onb.inverseTransform( make_float3( x, y, z ) );
        
        sample.w_in = optix::reflect( -w_out, H );
        
        const float3 N = p.shading_normal;
        const float  diffuse_pdf  = diffusePDF( w_out, N, sample.w_in );
        const float  specular_pdf = specularPDF( w_out, N, sample.w_in );
        sample.pdf = optix::lerp( specular_pdf, diffuse_pdf, diffuse_weight );

        const float3 diff_val = diffuseEval( w_out, N, sample.w_in );
        const float3 spec_val = specularEval( w_out, N, sample.w_in );
        sample.f_over_pdf = sample.pdf > 0.00001f            ? 
                            (diff_val+spec_val) / sample.pdf :
                            make_float3( 0.0f );
        sample.is_specular  = 1u; 

    }
        
    return sample;
}


RT_CALLABLE_PROGRAM
float4 wardEvaluateBSDF(
        float3 w_out,
        legion::LocalGeometry p,
        float3 w_in )
{
    const float  cos_in   = optix::dot( w_in,  p.shading_normal );
    const float  cos_out  = optix::dot( w_out, p.shading_normal );
    if( cos_in <= 0.0f || cos_out <= 0.0f )
        return make_float4( 0.0f );

    const float3 N            = p.shading_normal;
    const float  diffuse_pdf  = diffusePDF( w_out, N, w_in );
    const float  specular_pdf = specularPDF( w_out, N, w_in );
    const float  pdf          = optix::lerp( specular_pdf, diffuse_pdf, diffuse_weight );

    const float3 diff_val     = diffuseEval( w_out, N, w_in );
    const float3 spec_val     = specularEval( w_out, N, w_in );
    return make_float4( spec_val + diff_val, pdf );
    
    /*
    legion::ONB  onb( p.shading_normal );
    const float3 H          = optix::normalize( w_out + w_in );
    const float  spec_pdf   = specularPDF( w_out, p.shading_normal, w_in ); 
    const float  spec_coeff = 0.25f / 
                              ( alpha_u * alpha_v * sqrtf( cos_in*cos_out ) ) *
                              wardExpTerm( 1.0f/alpha_u, 1.0f/alpha_v, H, onb );
    const float3 spec_val   = spec_coeff * spec_reflectance;

    const float pdf =  optix::lerp( spec_pdf, diff_pdf, diffuse_weight );
    return make_float4( pdf*( spec_val + diff_val ), pdf );
    */
}


RT_CALLABLE_PROGRAM
float wardPDF( float3 w_out, legion::LocalGeometry p, float3 w_in )
{
        const float3 N = p.shading_normal;
        const float  diffuse_pdf  = diffusePDF( w_out, N, w_in );
        const float  specular_pdf = specularPDF( w_out, N, w_in );
        return optix::lerp( specular_pdf, diffuse_pdf, diffuse_weight );
}
