
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


#ifndef LEGION_OBJECTS_CAMERA_ICAMERA_HPP_
#define LEGION_OBJECTS_CAMERA_ICAMERA_HPP_

#include <Legion/Objects/IObject.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>

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
class LCLASSAPI ICamera : public IObject
{
public:
    LAPI ICamera( Context* context ) : IObject( context ) {}

    LAPI virtual ~ICamera() {}

    /// Return the name of this Camera type.  The associated PTX file should
    /// be named {name()}.ptx
    LAPI virtual const char* name()const=0;

    /// Return the name of this Camera's cuda ray generation function 
    LAPI virtual const char* createRayFunctionName()const=0;

    /// Set the Camera-to-World transform
    ///    \param camera_to_world   Camera-to-World transform
    LAPI virtual void setCameraToWorld( const Matrix& camera_to_world )=0;
};


}

#endif // LEGION_OBJECTS_CAMERA_ICAMERA_HPP_
