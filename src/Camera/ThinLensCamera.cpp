
#ifndef LEGION_CAMERA_H_
#define LEGION_CAMERA_H_


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


void ThinLensCamera::generateCameraSpaceRay( const Sample& filtered_sample, CameraSpaceRay& ray )
{
}
