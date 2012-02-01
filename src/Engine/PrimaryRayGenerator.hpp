
#ifndef LEGION_ENGINE_PRIMARY_RAY_GENERATOR_HPP_
#define LEGION_ENGINE_PRIMARY_RAY_GENERATOR_HPP_

#include <Core/Vector.hpp>

namespace legion
{
    class ICamera;

    class PrimaryRayGenerator
    {
    public:
        PrimaryRayGenerator( const Index2& screen_resolution, const ICamera* camera );
        void generate( unsigned pass);
    private:
        Index2          m_screen_resolution;
        const ICamera*  m_camera;
    };
};

#endif // LEGION_ENGINE_PRIMARY_RAY_GENERATOR_HPP_
