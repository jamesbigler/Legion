

#include <Legion/Scene/Camera/ThinLensCamera.hpp>
#include <Legion/Common/Math/Math.hpp>
#include <Legion/Common/Util/Stream.hpp>
#include <Legion/Core/Ray.hpp>
#include <iostream>

using namespace legion;


ThinLensCamera::ThinLensCamera( Context* context, const std::string& name )
  : BasicCamera( context, name )
{
    // TODO: give default values to all params
}


ThinLensCamera::~ThinLensCamera()
{
}


void ThinLensCamera::setViewPlane( float left,
                                   float right,
                                   float bottom,
                                   float top )
{
    m_left   = left; 
    m_right  = right; 
    m_bottom = bottom;
    m_top    = top;
}


void ThinLensCamera::setFocalDistance( float distance )
{
    m_focal_distance = distance;
}


void ThinLensCamera::setLensRadius( float radius )
{
    m_lens_radius = radius;
}


void ThinLensCamera::generateCameraSpaceRay( const CameraSample& sample,
                                             Ray& ray )const
{
    Vector3 on_viewplane( legion::lerp( m_left, m_right, sample.viewplane.x() ),
                          legion::lerp( m_bottom, m_top, sample.viewplane.y() ),
                          0.0f );
    
    Vector2 lens_sample( legion::squareToDisk( sample.lens ) * m_lens_radius );
    Vector3 on_lens( lens_sample.x(), lens_sample.y(), 0.0f );
                        
    ray.setOrigin( on_lens );
    ray.setDirection( legion::normalize( on_viewplane - on_lens ) );

    std::cerr << " thinlens ray: " << ray.getOrigin() << " | " 
              << ray.getDirection() << std::endl;
}
