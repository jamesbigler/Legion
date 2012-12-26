
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

#ifndef LEGION_SCENE_CAMERA_THINLENSCAMERA_HPP_
#define LEGION_SCENE_CAMERA_THINLENSCAMERA_HPP_

#include <Legion/Scene/Camera/ICamera.hpp>
#include <Legion/Common/Math/Matrix.hpp>

namespace legion
{


/// A basic thin lens camera implementation.  Supports depth-of-field with a
/// round lens model.
class ThinLensCamera : public ICamera
{
public:
    /// Create a named ThinLensCamera object
    ThinLensCamera();

    /// Destroy a ThinLensCamera object
    ~ThinLensCamera();

    const char* rayGenFunctionName();
    
    /// Set the Camera-to-World transform
    ///    \param camera_to_world   Camera-to-World transform
    void setCameraToWorld( const Matrix& camera_to_world );

    /// Set the image plane aspect ratio
    ///   \param ratio image plane aspect ratio
    void setAspectRatio( float ratio );

    /// Set the distance from the lens to the world point of perfect focus
    ///   \param distance  distance to world focal plane
    void setFocalDistance( float distance );

    /// Set the distance from the lens to the sensor-side point of perfect focus
    ///   \param length distance to internal focal plane 
    void setFocalLength( float length );

    /// Set the radius of the lens aperture to control the extent of the circle
    /// of confusion
    ///   \param radius  The lens radius
    void setApertureRadius( float radius );

    /// See ISceneObject::setVariables
    void setVariables( VariableContainer& container ) const;
    
private:
    Matrix       m_camera_to_world;  ///< Camera-to-world transform
    float        m_aspect_ratio;     ///< Width / height
    float        m_focal_distance;   ///< Dist from lens to world focal plane
    float        m_focal_length;     ///< Dist from lens to internal focal point
    float        m_aperture_radius;  ///< Lens aperture radius
};

}

#endif // LEGION_SCENE_CAMERA_THINLENSCAMERA_HPP_
