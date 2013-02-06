
#ifndef LEGION_COMMON_MATH_MATH_HPP_
#define LEGION_COMMON_MATH_MATH_HPP_

#include <optixu/optixu_math_namespace.h>
#include <Legion/Common/Util/Preprocessor.hpp>


namespace legion
{
    typedef unsigned           uint32;
    typedef unsigned long long uint64;

    const float PI             = static_cast<float>( M_PI );
    const float TWO_PI         = PI * 2.0f;
    const float ONE_DIV_PI     = 1.0f / PI;
    const float ONE_DIV_TWO_PI = 1.0f / TWO_PI;


    LDEVICE inline bool nan( float2 x )
    { return isnan( x.x ) || isnan( x.y ); }


    LDEVICE inline bool nan( float3 x )
    { return isnan( x.x ) || isnan( x.y ) || isnan( x.z ); }
    

    LDEVICE inline bool finite( float2 x )
    { return isfinite( x.x ) && isfinite( x.y ); }


    LDEVICE inline bool finite( float3 x )
    { return isfinite( x.x ) && isfinite( x.y ) && isfinite( x.z ); }

    LDEVICE inline bool finite( float4 x )
    { return isfinite( x.x ) && isfinite( x.y ) && isfinite( x.z ) && 
             isfinite( x.w ); }

    LDEVICE inline float lerp( float a, float b, float t )
    {
        return a + t*(b-a);
    }


    template<typename T>
    LDEVICE T trilerp( 
            const T& x000, const T& x100,
            const T& x010, const T& x110,
            const T& x001, const T& x101,
            const T& x011, const T& x111,
            float s, float t, float u )
    {
        return optix::lerp(
                optix::bilerp( x000, x100, x010, x110, s, t ),
                optix::bilerp( x001, x101, x011, x111, s, t ),
                u);
    }

    
    LDEVICE inline float powerHeuristic(float pdf1, float pdf2)
    { 
        const float temp = pdf1*pdf1;
        return temp/(temp+pdf2*pdf2); 
    }


    LDEVICE inline float pdfAreaToSolidAngle(
        float area_pdf,
        float distance,
        float cosine )
    {
        return area_pdf * distance * distance / fabsf( cosine );
    }


    LDEVICE inline bool isBlack( float3 x )
    {
        return x.x == 0.0f && x.y == 0.0f && x.z == 0.0f;
    }


    LDEVICE inline float2 squareToDisk( float2 sample )
    {
        const float PI_4 = static_cast<float>( M_PI ) / 4.0f;
        const float PI_2 = static_cast<float>( M_PI ) / 2.0f;

        const float a = 2.0f * sample.x - 1.0f;
        const float b = 2.0f * sample.y - 1.0f;

        float phi,r;
        if( a*a > b*b ) 
        {
            r = a;
            phi = PI_4 * ( b/a );
        }
        else
        {
            r = b;
            phi =  b ? PI_4 *( a/b ) + PI_2 : 0.0f;
        }

        return make_float2( r*cosf( phi ), r*sinf( phi ) );
    }


    LDEVICE inline void uniformSampleSphere( 
            float2  seed,
            float3& dir,
            float&  pdf )
    {
        float phi       = 2.0f * legion::PI * seed.x;
        float cos_theta = 1.0f - 2.0f * seed.y;
        float sin_theta = sqrtf( 1.0f - cos_theta * cos_theta);
        dir = make_float3( cosf( phi ) * sin_theta,
                           sinf( phi ) * sin_theta,
                           cos_theta );
        pdf = ONE_DIV_PI * 0.25f;
    }
}

#endif //LEGION_COMMON_MATH_MATH_HPP_
