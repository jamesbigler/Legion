
#ifndef LEGION_CAMERA_H_
#define LEGION_CAMERA_H_

#include <private/APIBase.hpp>

namespace legion
{

class ICamera : public APIBase
{
public:
    struct Sample
    {
        Vector2 pixel;
        Vector2 lens;
        float   time;
    };

    enum PixelFilter
    {
        FILTER_NONE = 0,
        FILTER_BOX,
        FILTER_TENT,
        FILTER_CUBIC_SPLINE
    };

    explicit Camera( const std::string& name );
    virtual ~Camera();

    virtual void setFilter( FilterType filter )=0;
    virtual void setTransform( legion::Matrix4x4, float time )=0;
    virtual void setShutterOpenClose( float open, float close )=0;
    virtual void generateRay( const Sample& sample, legion::Ray& transformed_ray )=0;

};

class IBasicCamera : public ICamera
{
public:
    explicit Camera( const std::string& name );
    virtual ~Camera();

    void setFilter( FilterType filter );
    void setTransform( legion::Matrix4x4, float time );
    void setShutterOpenClose( float open, float close );
    void generateRay( const Sample& sample, legion::Ray& transformed_ray );

private:
    struct CameraSpaceRay
    {
        Vector3 direction;
    };

    virtual void generateCameraSpaceRay( const Sample& filtered_sample, CameraSpaceRay& ray )=0;

    class Impl;
    std::shared_ptr<Impl> m_impl;
};

}
#endif // LEGION_CAMERA_H_
