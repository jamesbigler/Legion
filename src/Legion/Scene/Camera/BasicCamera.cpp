

#include <Legion/Scene/Camera/BasicCamera.hpp>
#include <Legion/Core/Ray.hpp>

using namespace legion;


BasicCamera::BasicCamera( Context* context, const std::string& name )
  : ICamera( context, name )
{
}


BasicCamera::~BasicCamera()
{
}


void BasicCamera::setTransform( const Matrix4x4& matrix, float time )
{
}


void BasicCamera::setShutterOpenClose( float open, float close )
{
}


void BasicCamera::generateRay( const CameraSample& sample,
                               Ray& transformed_ray )const
{
  // generate camera space ray
  Ray ray;
  generateCameraSpaceRay( sample, ray );

  // Transform ray into world space
  transformed_ray = ray;
}
