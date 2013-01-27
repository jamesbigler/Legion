
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

#ifndef LEGION_OBJECTS_CAMERA_THIN_LENS_HPP_
#define LEGION_OBJECTS_CAMERA_THIN_LENS_HPP_

#include <Legion/Objects/Camera/ICamera.hpp>
#include <Legion/Common/Math/Matrix.hpp>
#include <Legion/Common/Math/Vector.hpp>

namespace legion
{
/// TODO: there should be wrapper class(es) which give reasonable interfaces
///       for controlling DOF, FOV, etc.
///       Example: ThinLensSLR which gives focal_length (eg, 35mm) f_stop, etc.

/// A basic thin lens camera implementation.  Supports depth-of-field with a
/// round lens model.
class ThinLens : public ICamera
{
public:
    /// Create a ThinLens object
    ThinLens( Context* context );

    /// Destroy a ThinLens object
    ~ThinLens();

    const char* name()const;

    const char* createRayFunctionName()const;
    
    /// Set the Camera-to-World transform
    ///    \param camera_to_world   Camera-to-World transform
    void setCameraToWorld( const Matrix& camera_to_world );

    /// Set the distance from the lens to the world point of perfect focus
    ///   \param distance  distance to world focal plane
    void setFocalDistance( float distance );

    /// Set the radius of the lens aperture to control the extent of the circle
    /// of confusion
    ///   \param radius  The lens radius
    void setApertureRadius( float radius );

    ///
    void setViewPlane( float l, float r, float b, float t );

    /// See IObject::setVariables
    void setVariables( const VariableContainer& container ) const;
    
private:
    Matrix       m_camera_to_world;  ///< Camera-to-world transform
    float        m_focal_distance;   ///< Dist from lens to world focal plane
    float        m_aperture_radius;  ///< Lens aperture radius
    Vector4      m_view_plane;       ///< 
};

}

#endif // LEGION_OBJECTS_CAMERA_THIN_LENS_HPP_
