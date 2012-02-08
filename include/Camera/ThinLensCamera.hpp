
#ifndef LEGION_THIN_LENS_CAMERA_HPP_
#define LEGION_THIN_LENS_CAMERA_HPP_

#include <Camera/BasicCamera.hpp>


namespace legion
{

/// A basic thin lens camera implementation.  Supports depth-of-field with a
/// round lens model.
class ThinLensCamera : public BasicCamera
{
public:

    /// Create a named ThinLensCamera object
    explicit          ThinLensCamera( const std::string& name );

    /// Destroy a ThinLensCamera object
                      ~ThinLensCamera();

    /// Set the position of the viewplane in camera coordinates.  The viewplane
    /// is set at 1 unit from the camera and determintes the viewplane skew,
    /// horizontal field of view and vertical field of view.
    ///   \param left   The position of the left edge of the viewplane
    ///   \param right  The position of the right edge of the viewplane
    ///   \param bottom The position of the bottom edge of the viewplane
    ///   \param top    The position of the top edge of the viewplane
    void setViewPlane( float left, float right, float bottom, float top );

    /// The distance from the camera to the plane of perfect focus
    ///   \param distance  The distance to focal plane
    void setFocalDistance( float distance );

    /// Set the radius of the lens aperture to control the extent of the circle
    /// of confusion
    ///   \param radius  The lens radius
    void setLensRadius( float radius );

private:
    /// See IBasicCamera::generateCameraSpaceRay.
    void generateCameraSpaceRay( const CameraSample& sample, Ray& ray )const;

    
    float m_left;                 ///< Left edge of viewplane in camera coords
    float m_right;                ///< Right edge of viewplane in camera coords
    float m_bottom;               ///< Bottom edge of viewplane in camera coords
    float m_top;                  ///< Top edge of viewplane in camera coords

    float m_focal_distance;       ///< Distance from lens to focal plane
    float m_lens_radius;          ///< Lens aperture radius
};

}

#endif // LEGION_THIN_LENS_CAMERA_HPP_
