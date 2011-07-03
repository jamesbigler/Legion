
#ifndef LEGION_INTERFACE_IBASIC_CAMERA_H_
#define LEGION_INTERFACE_IBASIC_CAMERA_H_

#include <Interface/ICamera.hpp>


namespace legion
{



class IBasicCamera : public ICamera
{
public:
    explicit IBasicCamera( const std::string& name );
    virtual ~IBasicCamera();

    void setTransform( const Matrix4x4& matrix, float time );
    void setShutterOpenClose( float open, float close );
    void generateRay( const Camera::Sample& sample, legion::Ray& transformed_ray )const;

protected:
    struct CameraSpaceRay
    {
        Vector3 origin;
        Vector3 direction;
    };

private:
    virtual void generateCameraSpaceRay( const Camera::Sample& filtered_sample, CameraSpaceRay& ray )const=0;

    class Impl;
    std::tr1::shared_ptr<Impl> m_impl;
};


}

#endif // LEGION_INTERFACE_IBASIC_CAMERA_H_
