

/// \file BasicCamera.hpp
/// Base class for cameras with a default implementation of Ray camera-to-world
/// transformation

#ifndef LEGION_INTERFACE_BASIC_CAMERA_H_
#define LEGION_INTERFACE_BASIC_CAMERA_H_

#include <Legion/Scene/Camera/ICamera.hpp>
#include <tr1/memory>

namespace legion
{


/// 
/// Base class for cameras with a default implementation of Ray camera-to-world
/// transformation.  This means that derived class need only supply a function
/// for generating rays in camera space.
///
class BasicCamera : public ICamera
{
public:
    /// Create named BasicCamera
    /// \param name  The name of the BasicCamera
    explicit BasicCamera( const std::string& name );

    /// Destroy BasicCamera
    virtual ~BasicCamera();

    /// Set the Camera-to-World transform associated with a given time
    /// \param matrix   The transform matrix 
    /// \param time     The time associated with this transform 
    void setTransform( const Matrix4x4& matrix, float time );

    /// Set the shutter open and close time.  If not called the default values
    /// zero, and zero are used.
    /// \param open   Shutter open time
    /// \param open   Shutter close time
    void setShutterOpenClose( float open, float close );

    /// Generates a camera space ray by calling the user supplied
    /// BasicCamera::generateCameraSpaceRay function and transforming according
    /// to the transforms specified via setTransform.  
    /// \param sample      A 2D sample in [0,1]^2
    /// \param ray         Output parameter for generated world space ray
    void generateRay( const CameraSample& sample, Ray& ray )const;

protected:
    /// Generate ray in camera space.
    /// \param sample      A 2D sample in [0,1]^2
    /// \param ray         Output parameter for generated world space ray
    virtual void generateCameraSpaceRay( const CameraSample& sample,
                                         Ray& ray )const=0;

private:
    class Impl;
    std::tr1::shared_ptr<Impl> m_impl;
};


}

#endif // LEGION_INTERFACE_BASIC_CAMERA_H_
