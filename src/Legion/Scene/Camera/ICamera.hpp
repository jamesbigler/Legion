

/// \file ICamera.hpp
/// Pure virtual interface for Camera classes


#ifndef LEGION_SCENE_CAMERA_ICAMERA_H_
#define LEGION_SCENE_CAMERA_ICAMERA_H_

#include <Legion/Core/APIBase.hpp>
#include <Legion/Common/Math/Vector.hpp>

namespace legion
{

class Context;
class Ray;
class Matrix;


/// Contains the necessary sample information to generate a Camera ray
struct CameraSample
{
    Vector2 screen;  ///< Viewplane coordinates 
    Vector2 lens;       ///< Lens coordinates
    float   time;       ///< Time value associated with ray
};



/// Pure virtual interface for Camera objects
class ICamera : public APIBase
{
public:

    /// Create named camera object
    /// \param name Then name of the camera
                       ICamera( Context* context, const std::string& name );

    /// Destroy the camera object 
    virtual            ~ICamera();

    /// Set camera-to-world transform at the given time.
    /// \param matrix  The camera-to-world transform
    /// \param time    Time associated with the tranform
    virtual void       setTransform( const Matrix& matrix, float time )=0;

    /// Set times of shutter open and close
    /// \param open  Time of shutter open
    /// \param close Time of shutter close
    virtual void       setShutterOpenClose( float open, float close )=0;

    /// Generate a camera space ray given a 2D sample
    /// \param sample           A 2D sample in [0,1]^2
    /// \param time             The time associated with this ray
    /// \param transformed_ray  The generated ray in camera space 
    virtual void       generateRay( const CameraSample& sample,
                                    legion::Ray& transformed_ray )const=0;
};


}

#endif // LEGION_SCENE_CAMERA_ICAMERA_H_
