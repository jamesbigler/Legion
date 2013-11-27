
#ifndef LEGION_COMMON_MATH_MATH_HPP_
#define LEGION_COMMON_MATH_MATH_HPP_

#include <Legion/Core/Color.hpp>
#include <Legion/Common/Math/Vector.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>
#include <cmath>


namespace legion
{
    const float PI         = static_cast<float>( M_PI );
    const float TWO_PI     = static_cast<float>( M_PI ) * 2.0f;
    const float ONE_DIV_PI = 1.0f / static_cast<float>( M_PI );


    LHOSTDEVICE inline float lerp( float a, float b, float t )
    {
        return a + t*(b-a);
    }
    
    LHOSTDEVICE inline Color lerp( const Color& a, const Color& b, float t )
    {
        return a + t*(b-a);
    }


    LHOSTDEVICE inline legion::Vector2 squareToDisk( 
            const legion::Vector2& sample )
    {
        const float PI_4 = static_cast<float>( M_PI ) / 4.0f;
        const float PI_2 = static_cast<float>( M_PI ) / 2.0f;

        const float a = 2.0f * sample.x() - 1.0f;
        const float b = 2.0f * sample.y() - 1.0f;
        
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

        return Vector2( r*cosf( phi ), r*sinf( phi ) );
    }

    inline Vector3 Yxy2rgb( const Vector3& Yxy )
    {
      // First convert to xyz
      Vector3 xyz( Yxy.y() * ( Yxy.x() / Yxy.z() ),
                   Yxy.x(),
                   ( 1.0f - Yxy.y() - Yxy.z() ) * ( Yxy.x() / Yxy.z() ) );

      const float R = dot( xyz, Vector3(  3.2410f, -1.5374f, -0.4986f ) );
      const float G = dot( xyz, Vector3( -0.9692f,  1.8760f,  0.0416f ) );
      const float B = dot( xyz, Vector3(  0.0556f, -0.2040f,  1.0570f ) );
      return Vector3( R, G, B );
    }


    inline Vector3 rgb2Yxy( Vector3 rgb)
    {
      // convert to xyz
      const float X = dot( rgb, Vector3( 0.4124f, 0.3576f, 0.1805f ) );
      const float Y = dot( rgb, Vector3( 0.2126f, 0.7152f, 0.0722f ) );
      const float Z = dot( rgb, Vector3( 0.0193f, 0.1192f, 0.9505f ) );

      // convert xyz to Yxy
      return Vector3( Y,
                      X / ( X + Y + Z ),
                      Y / ( X + Y + Z ) );
    }

    inline Vector3 reinhardToneMappingOperator(
            const Vector3& c,
            float exposure )
    {
        const Vector3 Yxy        = rgb2Yxy( c );
        const float   Y          = Yxy.x();
        const float   scaled_Y   = Y * exposure;
        const float   mapped_Y   = scaled_Y / ( scaled_Y + 1.0f );
        const Vector3 mapped_Yxy( mapped_Y, Yxy.y(), Yxy.z() );

        return Yxy2rgb( mapped_Yxy );
    }
}

#endif //LEGION_COMMON_MATH_MATH_HPP_
