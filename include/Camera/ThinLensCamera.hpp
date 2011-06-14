
#ifndef LEGION_THIN_LENS_CAMERA_HPP_
#define LEGION_THIN_LENS_CAMERA_HPP_

#include <Interface/IBasicCamera.hpp>


namespace legion
{


class ThinLensCamera : public IBasicCamera
{
public:
    explicit ThinLensCamera( const std::string& name );
    ~ThinLensCamera();

    void setViewPlane( float left, float right, float bottom, float top );
    void setFocalDistance( float distance );
    void setLensRadius( float radius );

private:
    void generateCameraSpaceRay( const Camera::Sample& filtered_sample, CameraSpaceRay& ray );

    class Impl;
    std::tr1::shared_ptr<Impl> m_impl;
};


}

#endif // LEGION_THIN_LENS_CAMERA_HPP_