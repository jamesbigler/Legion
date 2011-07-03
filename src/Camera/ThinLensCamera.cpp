

#include <Camera/ThinLensCamera.hpp>
#include <Math/Math.hpp>

using namespace legion;


ThinLensCamera::ThinLensCamera( const std::string& name )
  : IBasicCamera( name )
{
    // TODO: give default values to all params
    
}


ThinLensCamera::~ThinLensCamera()
{
}


void ThinLensCamera::setViewPlane( float left, float right, float bottom, float top )
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


void ThinLensCamera::generateCameraSpaceRay( const Camera::Sample& filtered_sample, CameraSpaceRay& ray )const
{
    Vector3 on_viewplane( legion::lerp( m_left, m_right, filtered_sample.viewplane.x() ),
                          legion::lerp( m_bottom, m_top, filtered_sample.viewplane.y() ),
                          0.0f );
    
    Vector2 lens_sample( legion::squareToDisk( filtered_sample.lens ) * m_lens_radius );
    Vector3 on_lens(     lens_sample.x(), lens_sample.y(), 0.0f );
                        
    ray.origin    = on_lens;
    ray.direction = legion::normalize( on_viewplane - on_lens );
                        
}
