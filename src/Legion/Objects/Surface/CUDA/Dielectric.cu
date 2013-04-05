
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
#include <Legion/Common/Math/CUDA/Math.hpp>


rtDeclareVariable( float,  ior_in    , , );
rtDeclareVariable( float,  ior_out   , , );
rtDeclareVariable( float3, absorption, , );
rtDeclareVariable( float3, transmittance, , );
rtDeclareVariable( float3, reflectance, , );

rtDeclareVariable( float,  t_hit, rtIntersectionDistance, );

LDEVICE inline float fresnel( float cos_theta_i, float cos_theta_t, float eta )
{
    const float rs = ( cos_theta_i - cos_theta_t*eta ) / 
                     ( cos_theta_i + eta*cos_theta_t );
    const float rp = ( cos_theta_i*eta - cos_theta_t ) /
                     ( cos_theta_i*eta + cos_theta_t );

    return 0.5f * ( rs*rs + rp*rp );
}


RT_CALLABLE_PROGRAM
legion::BSDFSample dielectricSampleBSDF( 
        float3 seed,
        float3 w_out,
        legion::LocalGeometry p )
{
    legion::BSDFSample sample;

    float3 normal      = p.shading_normal;
    float cos_theta_i = optix::dot( w_out, p.shading_normal );
    float eta;
    float3 attenuation = make_float3( 1.0f );
    if( cos_theta_i > 0.0f )
    {  
        // Ray is entering 
        eta         = ior_in / ior_out;
    }
    else
    {
        // Ray is exiting 
        attenuation.x = powf( absorption.x, t_hit );
        attenuation.y = powf( absorption.y, t_hit );
        attenuation.z = powf( absorption.z, t_hit );
        eta         = ior_out / ior_in;
        cos_theta_i = -cos_theta_i;
        normal      = -normal;
    }

    float3 w_t;
    const bool tir           = !optix::refract( w_t, -w_out, normal, eta );
    const float cos_theta_t  = -optix::dot( normal, w_t );
    const float R            = tir  ?
                               1.0f :
                               fresnel( cos_theta_i, cos_theta_t, eta );

    if( tir || seed.z < R )
    {
        // Reflect
        sample.w_in = optix::reflect( -w_out, normal ); 
        sample.pdf         = R; 
        sample.f_over_pdf  = reflectance*attenuation;

    }
    else
    {
        // Refract
        sample.w_in = w_t; 
        sample.pdf         = 1.0 - R; 
        sample.f_over_pdf  = transmittance*attenuation;
    }
    sample.is_singular = true;
    sample.event_type = legion::BSDF_EVENT_SPECULAR;

    CHECK_FINITE( sample.w_in       );
    CHECK_FINITE( sample.f_over_pdf );
    CHECK_FINITE( sample.pdf        );
    return sample;
}


RT_CALLABLE_PROGRAM
float4 dielectricEvaluateBSDF(
        float3 w_out,
        legion::LocalGeometry p,
        float3 w_in )
{
    return make_float4( 0.0f ); 
}


RT_CALLABLE_PROGRAM
float dielectricPDF( float3 w_out, legion::LocalGeometry p, float3 w_in )
{
    return 0.0f;
}
