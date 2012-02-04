
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


// TODO: need to figure out Lens/Film/Camera relations

class ICamera : public APIBase
{
public:

    /// Create named camera object
    /// \param name Then name of the camera
    explicit ICamera( const std::string& name );

    /// Destroy the camera object 
    virtual ~ICamera();

    /// Set camera-to-world transform at the given time.
    /// \param matrix  The camera-to-world transform
    /// \param time    Time associated with the tranform
    virtual void setTransform( const Matrix4x4& matrix, float time )=0;

    /// Set times of shutter open and close
    /// \param open  Time of shutter open
    /// \param close Time of shutter close
    virtual void setShutterOpenClose( float open, float close )=0;

    /// Generate a camera space ray given a 2D sample
    /// \param sample           A 2D sample in [0,1]^2
    /// \param time             The time associated with this ray
    /// \param transformed_ray  The generated ray in camera space 
    virtual void generateRay( const Camera::Sample& sample,
                              legion::Ray& transformed_ray )const=0;
};


}

#endif // LEGION_INTERFACE_ICAMERA_H_
