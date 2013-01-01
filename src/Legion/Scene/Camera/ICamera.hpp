
// Copyright (C) 2011 R. Keith Morley 
// 
// (MIT/X11 License)
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.


/// \file ICamera.hpp
/// Pure virtual interface for Camera classes


#ifndef LEGION_SCENE_CAMERA_ICAMERA_H_
#define LEGION_SCENE_CAMERA_ICAMERA_H_

#include <Legion/Scene/ISceneObject.hpp>

namespace legion
{

class Matrix;
/// Contains the necessary sample information to generate a Camera ray
/*
struct CameraSample
{
    Vector2 screen;     ///< Viewplane coordinates 
    Vector2 lens;       ///< Lens coordinates
    float   time;       ///< Time value associated with ray
};
*/

/// Pure virtual interface for Camera objects
class ICamera : public ISceneObject
{
public:

    virtual ~ICamera() {}

    /// Return the name of this Camera type.  The associated PTX file should
    /// be named {name()}.ptx
    virtual const char* name()const=0;

    /// Return the name of this Camera's cuda ray generation function 
    virtual const char* createRayFunctionName()const=0;

    /// Set the Camera-to-World transform
    ///    \param camera_to_world   Camera-to-World transform
    virtual void setCameraToWorld( const Matrix& camera_to_world )=0;

    /*
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
                                    
    */
};


}

#endif // LEGION_SCENE_CAMERA_ICAMERA_H_
