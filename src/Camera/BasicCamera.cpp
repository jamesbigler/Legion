

#include <Camera/BasicCamera.hpp>
#include <Core/Ray.hpp>

using namespace legion;


BasicCamera::BasicCamera( const std::string& name )
  : ICamera( name )
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
  transformed_ray = Ray( ray.getOrigin(),
                         ray.getDirection(),
                         Vector2( 0.0f, 1e8f ) ); 
}
