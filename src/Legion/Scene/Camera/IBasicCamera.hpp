

/// \file IBasicCamera.hpp
/// Base class for cameras with a default implementation of Ray camera-to-world
/// transformation

#ifndef LEGION_SCENE_CAMERA_IBASICCAMERA_H_
#define LEGION_SCENE_CAMERA_IBASICCAMERA_H_

#include <Legion/Scene/Camera/ICamera.hpp>

#include <vector>
#include <tr1/memory>

namespace legion
{

class Context;
class Matrix;

/// 
/// Base class for cameras with a default implementation of Ray camera-to-world
/// transformation.  This means that derived class need only supply a function
/// for generating rays in camera space.
///
class IBasicCamera : public ICamera
{
public:
    /// Create named IBasicCamera
    /// \param name  The name of the IBasicCamera
    IBasicCamera( Context* context, const std::string& name );

    /// Destroy IBasicCamera
    virtual ~IBasicCamera();

    /// Set the Camera-to-World transform associated with a given time
    /// \param matrix   The transform matrix 
    /// \param time     The time associated with this transform 
    void setTransform( const Matrix& matrix, float time );

    /// Set the shutter open and close time.  If not called the default values
    /// zero, and zero are used.
    /// \param open   Shutter open time
    /// \param open   Shutter close time
    void setShutterOpenClose( float open, float close );

    /// Generates a camera space ray by calling the user supplied
    /// IBasicCamera::generateCameraSpaceRay function and transforming according
    /// to the transforms specified via setTransform.  
    /// \param sample      A 2D sample in [0,1]^2
    /// \param ray         Output parameter for generated world space ray
    void generateRay( const CameraSample& sample, Ray& ray )const;

protected:
    /// Generate ray in camera space.
    /// \param sample      A 2D sample in [0,1]^2
    /// \param ray         Output parameter for generated world space ray
    virtual void generateCameraSpaceRay( const CameraSample& sample,
                                         Vector3& origin,
                                         Vector3& direction )const=0;

private:

    std::vector<Matrix>  m_transform;

    float m_shutter_open_time;
    float m_shutter_close_time;
};


}

#endif // LEGION_SCENE_CAMERA_IBASICCAMERA_H_
