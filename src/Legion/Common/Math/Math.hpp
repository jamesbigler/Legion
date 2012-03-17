
#ifndef LEGION_COMMON_MATH_MATH_HPP_
#define LEGION_COMMON_MATH_MATH_HPP_

#include <Legion/Core/Color.hpp>
#include <Legion/Common/Math/Vector.hpp>
#include <cmath>


namespace legion
{
    const float PI         = static_cast<float>( M_PI );
    const float TWO_PI     = static_cast<float>( M_PI ) * 2.0f;
    const float ONE_DIV_PI = 1.0f / static_cast<float>( M_PI );


    inline float lerp( float a, float b, float t )
    {
        return a + t*(b-a);
    }
    
    inline Color lerp( const Color& a, const Color& b, float t )
    {
        return a + t*(b-a);
    }


    inline legion::Vector2 squareToDisk( const legion::Vector2& sample )
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
}

#endif //LEGION_COMMON_MATH_MATH_HPP_
