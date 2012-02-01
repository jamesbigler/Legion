
#ifndef LEGION_CORE_RAY_HPP_
#define LEGION_CORE_RAY_HPP_

#include <Core/Vector.hpp>

namespace legion
{

    class Ray
    {
    public:
        Ray() {}

        Ray( const Vector3& origin,
             const Vector3& direction,
             const Vector2& interval )
            : m_origin( origin ),
              m_direction( direction ),
              m_interval( interval )
        {

        }

        Vector3 origin()const    { return m_origin;       }
        Vector3 direction()const { return m_direction;    }
        float   tmin()const      { return m_interval.x(); }
        float   tmax()const      { return m_interval.y(); }
        Vector2 tInterval()const { return m_interval;     }

    private:
        Vector3 m_origin;
        Vector3 m_direction;
        Vector2 m_interval;
    };

}

#endif //  LEGION_CORE_RAY_HPP_
