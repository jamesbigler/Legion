
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
    LDEVICE float3  sample( float2 seed                         ) const;
    LDEVICE float   G     ( float3 w_in, float3 H, float3 w_out ) const;
    LDEVICE float   pdf   ( float3 N,    float3 H               ) const;
    LDEVICE float   D     ( float3 N,    float3 H               ) const;
private:
    float m_alpha;
};
    

BeckmannDistribution::BeckmannDistribution( float alpha )
    : m_alpha( alpha )
{
}


float3 BeckmannDistribution::sample( float2 seed ) const
{
    const float alpha_sqr     = m_alpha*m_alpha;
    const float tan_theta_sqr = -alpha_sqr*logf( 1.0f - seed.x );
    const float tan_theta     = sqrtf( tan_theta );
    const float cos_theta     = 1.0f / sqrtf( 1.0f _ tan_theta_sqr );
    const float sin_theta     = cos_theta * tan_theta;

    const float phi = 2.0f * legin::PI * seed.y;
    return make_float3( cosf( phi ) * sin_theta,
                        sinf( phi ) * sin_theta,
                        cos_theta );
}


float BeckmannDistribution::pdf( float3 N, float3 H ) const
{
    const float cos_theta = optix::dot( N, H );
    const float3 H = optix::normalize( w_in + w_out );
    return D( 
}


float BeckmannDistribution::G( float3 w_in, float3 H, float3 w_out ) const
{
}


float BeckmannDistribution::D( float3 m ) const
{
}


//-----------------------------------------------------------------------------
//
//  
//
//-----------------------------------------------------------------------------

class ConductorFresnel
{
    LDEVICE float F( float cos_theta );
};


class DielectricFresnel 
{
    LDEVICE float F( float cos_theta );
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
            );
        : m_reflectance( reflectance),
          m_distribution( dist ),
          m_fresnel( fresnel )
    {
    }


    legion::BSDFSample sampleBSDF(
            float3 seed,
            float3 w_out,
            legion::LocalGeometry p )
    {
        // Get the microfacet normal
        const float3 m = m_distribution.sample( seed );
        const float3 N = optix::faceforward( 
                p.shading_normal, w_out, p.geometric_normal
                );
        const legion::ONB onb( N );

        legion::BSDFSample sample;
        sample.w_in = optix::reflect( -w_out, onb.inverseTransform( m ) );

        if( optix::dot( w_in, p.geometric_normal <= 0.0f ) )
        {
            sample.pdf = 0.0f;
            sample.f_over_pdf = make_float3( 0.0f );
            return;
        }

        const float3 H         = optix::normalize( w_in + sample.w_out );
        const float  cos_theta = optix::dot( w_in, H );

        const float  D = m_distribution.D( m );
        const float  G = m_distribution.G( w_in, H, w_out );
        const float  F = m_fresnel.F( cos_theta );
        const float3 f = m_reflectance*( D*G*F/( 4.0f*cos_theta ) );

        sample.is_singular = false;
        sample.pdf         = m_distribution.pdf( w_in, H, w_out );
        sample.f_over_pdf  = f / sample.pdf; 
    }


    float4 evaluateBSDF(
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

        const float3 H         = optix::normalize( w_in + sample.w_out );
        const float  cos_theta = optix::dot( w_in, H );

        const float  D = m_distribution.D( m );
        const float  G = m_distribution.G( w_in, H, w_out );
        const float  F = m_fresnel.F( cos_theta );
        const float3 f = m_reflectance*( D*G*F/( 4.0f*cos_theta ) );
        const float  pdf = m_distribution.pdf( w_in, H, w_out );

        return make_float4( f, pdf ); 
    }


    float pdf( float3 w_out, legion::LocalGeometry p, float3 w_in )
    {
        const float3 N = optix::faceforward( 
                p.shading_normal, w_out, p.geometric_normal
                );
        const float  cos_in   = optix::dot( w_in,  N );
        const float  cos_out  = optix::dot( w_out, N );
        if( cos_in <= 0.0f || cos_out <= 0.0f )
            return 0.0f;
        
        const float3 H = optix::normalize( w_in + sample.w_out );
        const float  cos_theta = optix::dot( w_in, H );
        const float3 term = 1.0f/( 4.0f*cos_theta );
        return m_distribution.pdf( w_in, H, w_out )*term;
    }

private:
    Distribution m_distribution;
    Fresnel      m_fresnel;
    float3       m_reflectance;
};


} // namespace legion

#endif // LEGION_OBJECTS_SURFACE_CUDA_MICROFACET_HPP_
