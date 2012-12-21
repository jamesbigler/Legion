
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


#include <Legion/Scene/Camera/ThinLensCamera.hpp>


using namespace legion;


ThinLensCamera::ThinLensCamera() 
    : m_camera_to_world(),
      m_aspect_ratio( 1.0f ),
      m_focal_distance( 1.0f ),
      m_focal_length( 1.0f ),
      m_aperture_radius( 0.0f )

{
}


ThinLensCamera::~ThinLensCamera()
{
}

    
void ThinLensCamera::setCameraToWorld( Matrix camera_to_world )
{
    m_camera_to_world = camera_to_world;
}


void ThinLensCamera::setFocalDistance( float distance )
{
    m_focal_distance = distance;
}
    

void ThinLensCamera::setFocalLength( float length )
{
    m_focal_length = length;
}


void ThinLensCamera::setApertureRadius( float radius )
{
    m_aperture_radius = radius;
}
    

void ThinLensCamera::setVariables( VariableContainer& container ) const
{
}


/*
void ThinLensCamera::generateCameraSpaceRay( const CameraSample& sample,
                                             Vector3& origin,
                                             Vector3& direction )const
{
    Vector3 on_viewplane( legion::lerp( m_left, m_right, sample.screen.x() ),
                          legion::lerp( m_top, m_bottom, sample.screen.y() ),
                          -m_focal_distance );

    Vector2 on_lens( legion::squareToDisk( sample.lens ) * m_lens_radius );

    origin    = Vector3( on_lens.x(), on_lens.y(), 0.0f );
    direction = legion::normalize( on_viewplane - origin );
}
*/
