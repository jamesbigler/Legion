
#ifndef LEGION_INTERFACE_ICAMERA_H_
#define LEGION_INTERFACE_ICAMERA_H_

#include <private/APIBase.hpp>
#include <Core/Vector.hpp>
#include <Core/Matrix.hpp>
#include <Core/Ray.hpp>


namespace legion
{


namespace Camera
{
    struct Sample
    {
        Vector2 viewplane;
        Vector2 lens;
        float   time;
    };
}


class ICamera : public APIBase
{
public:

    explicit ICamera( const std::string& name );
    virtual ~ICamera();

    virtual void setTransform( const Matrix4x4& matrix, float time )=0;
    virtual void setShutterOpenClose( float open, float close )=0;
    virtual void generateRay( const Camera::Sample& sample, legion::Ray& transformed_ray )=0;

};


}

#endif // LEGION_INTERFACE_ICAMERA_H_
