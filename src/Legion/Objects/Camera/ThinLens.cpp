
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


#include <Legion/Objects/Camera/ThinLens.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Common/Util/Parameters.hpp>


using namespace legion;

ICamera* ThinLens::create( Context* context, const Parameters& params )
{
    const Vector4 default_vp( -0.5f, 0.5f, -0.375f, 0.375f );

    ThinLens* thinlens = new ThinLens( context );
    thinlens->setFocalDistance ( params.get( "focal_distance",  1.0f       ) );
    thinlens->setApertureRadius( params.get( "aperture_radius", 0.0f       ) );
    thinlens->setViewPlane     ( params.get( "view_plane",      default_vp ) );
    return thinlens;
}


ThinLens::ThinLens( Context* context ) 
    : ICamera( context ),
      m_camera_to_world( Matrix::identity() ),
      m_focal_distance( 1.0f ),
      m_aperture_radius( 0.0f )

{
    // 4:3 Aspect ratio default
    m_view_plane[ 0 ] = -0.5f;
    m_view_plane[ 1 ] =  0.5f;
    m_view_plane[ 2 ] = -0.375f;
    m_view_plane[ 3 ] =  0.375f;
}


ThinLens::~ThinLens()
{
}


const char* ThinLens::name()const
{
    return "ThinLens";
}

    
const char* ThinLens::createRayFunctionName()const
{
    return "thinLensCreateRay";
}

    
void ThinLens::setCameraToWorld( const Matrix& camera_to_world )
{
    m_camera_to_world = camera_to_world;
}


void ThinLens::setFocalDistance( float distance )
{
    m_focal_distance = distance;
}
    

void ThinLens::setApertureRadius( float radius )
{
    m_aperture_radius = radius;
}
    

void ThinLens::setViewPlane( float l, float r, float b, float t )
{
    setViewPlane( Vector4( l, r, b, t ) );
}

void ThinLens::setViewPlane( const Vector4& lrbt )
{
    m_view_plane = lrbt;
}


void ThinLens::setVariables( VariableContainer& container ) const
{
    container.setMatrix( "camera_to_world", m_camera_to_world );
    container.setFloat ( "focal_distance",  m_focal_distance );
    container.setFloat ( "aperture_radius", m_aperture_radius );
    container.setFloat ( "view_plane",      m_view_plane );
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
