

#include <Camera/ThinLensCamera.hpp>

using namespace legion;


ThinLensCamera::ThinLensCamera( const std::string& name )
  : IBasicCamera( name )
{
    
}


ThinLensCamera::~ThinLensCamera()
{
}


void ThinLensCamera::setViewPlane( float left, float right, float bottom, float top )
{
}


void ThinLensCamera::setFocalDistance( float distance )
{
}


void ThinLensCamera::setLensRadius( float radius )
{
}


void ThinLensCamera::generateCameraSpaceRay( const Camera::Sample& filtered_sample, CameraSpaceRay& ray )
{
}
