

#include <Interface/IBasicCamera.hpp>

using namespace legion;


IBasicCamera::IBasicCamera( const std::string& name )
  : ICamera( name )
{
}


IBasicCamera::~IBasicCamera()
{
}


void IBasicCamera::setTransform( const Matrix4x4& matrix, float time )
{
}


void IBasicCamera::setShutterOpenClose( float open, float close )
{
}


void IBasicCamera::generateRay( const Camera::Sample& sample, Ray& transformed_ray )const
{
  // generate camera space ray
  CameraSpaceRay ray;
  generateCameraSpaceRay( sample, ray );

  // Transform ray into world space
  transformed_ray = Ray(); 
}
