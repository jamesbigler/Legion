
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

#ifndef LEGION_OBJECTS_SURFACE_CUDA_MICROFACET_HPP_
#define LEGION_OBJECTS_SURFACE_CUDA_MICROFACET_HPP_

#include <Legion/Common/Util/Preprocessor.hpp>
#include <Legion/Common/Math/CUDA/ONB.hpp>
#include <Legion/Objects/Surface/CUDA/Surface.hpp>

namespace legion
{

//-----------------------------------------------------------------------------
//
// Beckmann microfacet distribution model
//
//-----------------------------------------------------------------------------

class BeckmannDistribution
{
public:
    LDEVICE explicit BeckmannDistribution( float alpha ); 

    /// Returns m (the microfacet normal) in local space
    LDEVICE float3  sample( float2 seed         ) const;
    LDEVICE float   pdf   ( float3 N, float3 H  ) const;
    LDEVICE float   D     ( float3 N, float3 H  ) const;

    /// returns [ D(), pdf() ].  Avoids recomputation
    LDEVICE float2  DPDF  ( float3 N, float3 H  ) const;
    LDEVICE float   G     ( float3 N,
                            float3 H, 
                            float3 w_in,
                            float3 w_out ) const;
private:
    LDEVICE float smithG1( float3 N, float3 H, float3 w ) const;

    float m_alpha;
};
    

LDEVICE BeckmannDistribution::BeckmannDistribution( float alpha )
    : m_alpha( alpha )
{
}


LDEVICE float3 BeckmannDistribution::sample( float2 seed ) const
{
    if( m_alpha < 0.000001f )
        return make_float3( 0.0f, 0.0f, 1.0f );

    const float alpha_sqr     = m_alpha*m_alpha;
    const float tan_theta_sqr = -alpha_sqr*logf( 1.0f - seed.x );
    const float cos_theta     = 1.0f / sqrtf( 1.0f + tan_theta_sqr );
    const float sin_theta     = cos_theta*sqrtf( tan_theta_sqr );

    const float phi = 2.0f * legion::PI * seed.y;
    return make_float3( cosf( phi ) * sin_theta,
                        sinf( phi ) * sin_theta,
                        cos_theta );
}


LDEVICE float BeckmannDistribution::pdf( float3 N, float3 H ) const
{
    const float cos_theta = optix::dot( N, H );
    if( cos_theta <= 0.0f || m_alpha < 0.000001f )
      return 0.0f;

    const float alpha_sqr     = m_alpha*m_alpha;
    const float cos_theta_sqr = cos_theta*cos_theta;
    const float tan_theta_sqr = (1.0f - cos_theta_sqr ) / cos_theta_sqr;
    const float exponent      = -tan_theta_sqr / alpha_sqr;
    const float p = expf( exponent ) /
                    ( legion::PI*alpha_sqr*cos_theta_sqr*cos_theta );

    CHECK_FINITE( p ); 
    return p;
}


LDEVICE float BeckmannDistribution::D( float3 N, float3 H ) const
{
    const float cos_theta = optix::dot( N, H );
    if( cos_theta <= 0.0f || m_alpha < 0.000001f )
      return 0.0f;

    const float alpha_sqr     = m_alpha*m_alpha;
    const float cos_theta_sqr = cos_theta*cos_theta;
    const float tan_theta_sqr = (1.0f - cos_theta_sqr ) / cos_theta_sqr;
    const float exponent      = -tan_theta_sqr / alpha_sqr;
    const float D = expf( exponent ) /
                    ( legion::PI*alpha_sqr*cos_theta_sqr*cos_theta_sqr );
    CHECK_FINITE( D ); 
    return D;
}

LDEVICE float2 BeckmannDistribution::DPDF( float3 N, float3 H ) const
{
    const float cos_theta = optix::dot( N, H );
    const float dee       = D( N, H );
    CHECK_FINITE( dee       ); 
    CHECK_FINITE( cos_theta ); 
    return make_float2( dee, dee/cos_theta);
}


LDEVICE float BeckmannDistribution::G( 
        float3 N,
        float3 H,
        float3 w_in,
        float3 w_out
        ) const
{
    return smithG1( N, H, w_in ) * smithG1( N, H, w_out );
}


/// Approximation from Walter, et al
LDEVICE inline float BeckmannDistribution::smithG1(
        float3 N,
        float3 m,
        float3 v ) const
{

    const float v_dot_m = optix::dot( v, m );
    const float v_dot_n = optix::dot( v, N );
    if( v_dot_m * v_dot_n <= 0.0f )
        return 0.0f;

    const float cos_theta     = v_dot_n; 
    const float cos_theta_sqr = cos_theta*cos_theta;
    const float t             = 1.0f - cos_theta_sqr; 
    const float tan_theta     = t <= 0.0f ? 0.0f : sqrtf( t ) / cos_theta;

    if( tan_theta == 0.0f )
        return 1.0f;

    const float a = 1.0f / ( tan_theta*m_alpha );

    if( a >= 1.6f )
        return 1.0f;

    const float a_sqr = a*a;
    return ( 3.535f*a + 2.181f*a_sqr ) /
           ( 1.0f + 2.276f*a + 2.577f*a_sqr );
}


//-----------------------------------------------------------------------------
//
//  
//
//-----------------------------------------------------------------------------

class ConductorFresnel
{
public:
    LDEVICE ConductorFresnel( float3 eta, float3 k )
        : m_eta( eta ),
          m_k( k )
    {}

    LDEVICE float3 F( float cos_theta )
    { return fresnelConductor( cos_theta, m_eta, m_k ); }

private:
    float3 m_eta;
    float3 m_k;
};


class SchlickFresnel
{
public:
    LDEVICE SchlickFresnel( float eta )
        : m_eta( eta )
    {}

    LDEVICE float3 F( float cos_theta )
    { return make_float3( fresnelSchlick( cos_theta, m_eta ) ); }

private:
    float m_eta;
};


class NopFresnel
{
public:
    LDEVICE float3 F( float cos_theta )
    { return make_float3( 1.0f ); }
};


//-----------------------------------------------------------------------------
//
// microfacet distribution model
//
//-----------------------------------------------------------------------------

template< typename Distribution, typename Fresnel>
class MicrofacetSurface
{
public:
    LDEVICE MicrofacetSurface( 
            float3       reflectance,
            Distribution distribution,
            Fresnel      fresnel
            )
        : m_reflectance( reflectance),
          m_distribution( distribution ),
          m_fresnel( fresnel )
    {
    }


    LDEVICE legion::BSDFSample sample(
            float2 seed,
            float3 w_out,
            legion::LocalGeometry p )
    {
        legion::BSDFSample sample;
        sample.is_singular = true;
        sample.pdf         = 0.0f;
        sample.f_over_pdf  = make_float3( 0.0f );
        sample.event_type  = legion::BSDF_EVENT_SPECULAR; 


        // Get the microfacet normal
        const float3 N = optix::faceforward( 
                p.shading_normal, w_out, p.geometric_normal
                );

        
        //if( optix::dot( w_out, N ) <= 0.0 )
        //    return sample;

        const legion::ONB onb( N );
        const float3 m = onb.inverseTransform( m_distribution.sample( seed ) );

        sample.w_in = optix::reflect( -w_out, m );

        const float n_dot_i        = optix::dot( N, sample.w_in );
        const float m_dot_i        = optix::dot( m, sample.w_in );
        const float microfacet_pdf = m_distribution.pdf( N, m );
        if( m_dot_i        <= 0.0f || 
            n_dot_i        <= 0.0f || 
            microfacet_pdf <= 0.0f )
            return sample;
        const float pdf = microfacet_pdf / ( 4.0f*m_dot_i );

        const float n_dot_m = fabs( optix::dot( N, m ) );

        const float  G    = m_distribution.G( N, m, sample.w_in, w_out );
        const float3 F    = m_fresnel.F( m_dot_i );
        const float3 w    = m_reflectance*F*( G*m_dot_i / ( n_dot_i*n_dot_m ) );

        sample.is_singular = false;
        sample.pdf         = pdf; 
        sample.f_over_pdf =  w; 
        return sample;
    }


    LDEVICE float4 evaluate(
            float3 w_out,
            legion::LocalGeometry p,
            float3 w_in 
            )
    {
        const float3 N = optix::faceforward( 
                p.shading_normal, w_out, p.geometric_normal
                );
        const float  cos_in   = optix::dot( w_in,  N );
        const float  cos_out  = optix::dot( w_out, N );
        if( cos_in <= 0.0f || cos_out <= 0.0f )
            return make_float4( 0.0f );

        const float3 H         = optix::normalize( w_in + w_out );
        const float  cos_theta = optix::dot( w_in, H );

        const float2 dpdf = m_distribution.DPDF( N, H );

        const float  D = dpdf.x; 
        const float  G = m_distribution.G( N, H, w_in, w_out );
        const float3 F = m_fresnel.F( cos_theta );
        const float3 f = m_reflectance*F*( D*G/( 4.0f*cos_in*cos_out ) );
        const float  pdf = dpdf.y/( 4.0*cos_theta );

        return make_float4( f, pdf ); 
    }


    LDEVICE float pdf( float3 w_out, legion::LocalGeometry p, float3 w_in )
    {
        const float3 N = optix::faceforward( 
                p.shading_normal, w_out, p.geometric_normal
                );
        const float  cos_in   = optix::dot( w_in,  N );
        const float  cos_out  = optix::dot( w_out, N );
        if( cos_in <= 0.0f || cos_out <= 0.0f )
            return 0.0f;
        
        const float3 H = optix::normalize( w_in + w_out );
        const float  cos_theta = optix::dot( w_in, H );
        const float  term = 1.0f/( 4.0f*fabs( cos_theta ) );
        return m_distribution.pdf( N, H )*term;
    }

private:
    Distribution m_distribution;
    Fresnel      m_fresnel;
    float3       m_reflectance;
};


} // namespace legion

#endif // LEGION_OBJECTS_SURFACE_CUDA_MICROFACET_HPP_
