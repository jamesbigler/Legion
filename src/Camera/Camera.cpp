

using namespace legion;


ICamera::ICamera( const std::string& name )
{
    m_impl = new
    class Impl;
    SharedPtr<Impl> m_impl;
}


void ICamera::setFilter( FilterType filter )
{
}


void ICamera::setTransform( Matrix4x4, float time )
{
}


void ICamera::generateRay( const Sample& sample, Ray& transformed_ray )
{
  // Filter the pixel sample
  Sample filtered_sample = sample;

  // generate camera space ray
  Ray ray;
  generateCameraSpaceRay( filtered_sample, ray );

  // Transform ray into world space
  transformed_ray = ray;
}


};


}
#endif // LEGION_CAMERA_H_
