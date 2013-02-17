
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


    LDEVICE inline float3 Yxy2rgb( float3 Yxy )
    {
    
        using optix::dot;
        // First convert to xyz
        float3 xyz = make_float3( Yxy.y * ( Yxy.x / Yxy.z ),
                                  Yxy.x,
                                  ( 1.0f - Yxy.y - Yxy.z ) * ( Yxy.x/Yxy.z ) );

        const float R = dot( xyz, make_float3(  3.2410f, -1.5374f, -0.4986f ) );
        const float G = dot( xyz, make_float3( -0.9692f,  1.8760f,  0.0416f ) );
        const float B = dot( xyz, make_float3(  0.0556f, -0.2040,   1.0570f ) );
        return make_float3( R, G, B );
    }


    LDEVICE inline float3 rgb2Yxy( float3 rgb)
    {
        using optix::dot;
        // convert to xyz
        const float X = dot( rgb, make_float3( 0.4124, 0.3576, 0.1805 ) );
        const float Y = dot( rgb, make_float3( 0.2126, 0.7152, 0.0722 ) );
        const float Z = dot( rgb, make_float3( 0.0193, 0.1192, 0.9505 ) );

        // convert xyz to Yxy
        return make_float3( Y,
                            X / ( X + Y + Z ),
                            Y / ( X + Y + Z ) );
    }

    LDEVICE inline float3 reinhardToneOperator( float3 c )
    {
        const float3 Yxy         = rgb2Yxy( c );
        const float   Y          = Yxy.x;
        const float   mapped_Y   = Y / ( Y + 1.0f );
        const float3 mapped_Yxy  = make_float3( mapped_Y, Yxy.y, Yxy.z );

        return Yxy2rgb( mapped_Yxy );
    }
    
    LDEVICE inline float3 gammaCorrect( float3 c, float gamma )
    {
        const float gamma_inv = 1.0f / gamma;
        return make_float3( powf( c.x, gamma_inv ),
                            powf( c.y, gamma_inv ),
                            powf( c.z, gamma_inv ) );
    }
}

#endif //LEGION_COMMON_MATH_MATH_HPP_
