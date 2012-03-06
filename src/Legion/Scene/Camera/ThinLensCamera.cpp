

#include <Legion/Scene/Camera/ThinLensCamera.hpp>
#include <Legion/Common/Math/Math.hpp>
#include <Legion/Common/Util/Stream.hpp>
#include <Legion/Common/Util/Logger.hpp>
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
                                             Vector3& origin,
                                             Vector3& direction )const
{
    Vector3 on_viewplane( legion::lerp( m_left, m_right, sample.screen.x() ),
                          legion::lerp( m_bottom, m_top, sample.screen.y() ),
                          -1.0f );

    Vector2 on_lens( legion::squareToDisk( sample.lens ) * m_lens_radius );

    origin    = Vector3( on_lens.x(), on_lens.y(), 0.0f );
    direction = legion::normalize( on_viewplane - origin );
}
