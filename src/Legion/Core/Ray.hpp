
#ifndef LEGION_CORE_RAY_HPP_
#define LEGION_CORE_RAY_HPP_

#include <Legion/Core/Vector.hpp>

namespace legion
{

    class Ray
    {
    public:
        Ray() {}

        Ray( const Vector3& origin, const Vector3& dir, float tmax, float time )
            : m_origin( origin ),
              m_direction( dir ),
              m_tmax( tmax ),
              m_time( time )
        {
        }

        Vector3 getOrigin()const                  { return m_origin;       }
        Vector3 getDirection()const               { return m_direction;    }

        void    setOrigin   ( const Vector3& o )  { m_origin    = o;       }
        void    setDirection( const Vector3& d )  { m_direction = d;       }

        float   getTMax()const                    { return m_tmax;         }
        float   getTime()const                    { return m_time;         }

    private:
        Vector3 m_origin;
        Vector3 m_direction;
        float   m_tmax;
        float   m_time;
    };

}

#endif //  LEGION_CORE_RAY_HPP_
