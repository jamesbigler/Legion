

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


void BasicCamera::setTransform( const Matrix& matrix, float time )
{
}


void BasicCamera::setShutterOpenClose( float open, float close )
{
}


void BasicCamera::generateRay( const CameraSample& sample,
                               Ray& transformed_ray )const
{
  // generate camera space ray
  Vector3 origin, direction;
  generateCameraSpaceRay( sample, origin, direction );

  // TODO: transform origin, direction

  // Transform ray into world space
  transformed_ray.setOrigin( origin );
  transformed_ray.setDirection( direction );
  transformed_ray.setTMax( 1.0e16f );
  transformed_ray.setTime( 0.0f );
}
