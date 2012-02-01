
#include <Engine/PrimaryRayGenerator.hpp>
#include <Interface/ICamera.hpp>
#include <Util/Stream.hpp>
#include <cstdlib>
#include <iostream>


using namespace legion;


PrimaryRayGenerator::PrimaryRayGenerator( const Index2& screen_resolution, const ICamera* camera )
    : m_screen_resolution( screen_resolution ),
      m_camera( camera )
{
}


void PrimaryRayGenerator::generate( unsigned pass )
{
    const Vector2 screen_resf = Vector2( m_screen_resolution );
    const Vector2 pixel_dimsf = Vector2( 1.0f, 1.0f ) / screen_resf;

    for( int i = 0; i < m_screen_resolution.x(); ++i )
    {
        for( int j = 0; j < m_screen_resolution.y(); ++j )
        {
            // TODO: This will use a sampler object
            Camera::Sample camera_sample;
            camera_sample.viewplane = Vector2( i, j ) / pixel_dimsf;
            camera_sample.lens      = Vector2( drand48(), drand48() );
            camera_sample.time      = drand48();
            Ray camera_ray;
            m_camera->generateRay( camera_sample, camera_ray );
            std::cerr << "Ray[" << i << ", " << j << "]: " << camera_ray << std::endl;
        }
    }
}
