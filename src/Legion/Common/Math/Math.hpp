
#ifndef LEGION_COMMON_MATH_MATH_HPP_
#define LEGION_COMMON_MATH_MATH_HPP_

#include <Legion/Core/Vector.hpp>
#include <cmath>


namespace legion
{
    inline float lerp( float a, float b, float t )
    {
        return a + t*(b-a);
    }


    inline legion::Vector2 squareToDisk( const legion::Vector2& sample )
    {
        float phi, r;

        const float a = 2.0f * sample.x() - 1.0f;
        const float b = 2.0f * sample.y() - 1.0f;

        if (a > -b)
        {
            if (a > b)
            {
                r = a;
                phi = static_cast<float>( M_PI ) / 4.0f * (b/a);
            }
            else
            {
                r = b;
                phi = static_cast<float>( M_PI ) / 4.0f * ( 2.0f - (a/b) );
            }
        }
        else
        {
            if (a < b)
            {
                r = -a;
                phi = static_cast<float>( M_PI ) / 4.0f * ( 4.0f + (b/a) );
            }
            else
            {
                r = -b;
                phi = b ? static_cast<float>( M_PI ) / 4.0f * ( 6.0f - (a/b) ) : 0.0f;
            }
        }

        return legion::Vector2( r * cosf(phi), r * sinf(phi) );
    }

 
}

#endif //LEGION_COMMON_MATH_MATH_HPP_
