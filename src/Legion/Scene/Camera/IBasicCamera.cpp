
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


#include <Legion/Common/Math/Math.hpp>
#include <Legion/Common/Math/Matrix.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Common/Util/Stream.hpp>
#include <Legion/Core/Ray.hpp>
#include <Legion/Scene/Camera/IBasicCamera.hpp>

using namespace legion;


IBasicCamera::IBasicCamera( Context* context, const std::string& name )
    : ICamera( context, name ),
      m_shutter_open_time( 0.0f ),
      m_shutter_close_time( 0.0f )
{
}


IBasicCamera::~IBasicCamera()
{
}


void IBasicCamera::setTransform( const Matrix& matrix, float time )
{
    m_transform.push_back( matrix );
}


void IBasicCamera::setShutterOpenClose( float open, float close )
{
    m_shutter_open_time  = open;
    m_shutter_close_time = close;
}


void IBasicCamera::generateRay( const CameraSample& sample,
                               Ray& transformed_ray )const
{
    // generate camera space ray 
    Vector3 origin, direction;
    generateCameraSpaceRay( sample, origin, direction );

    // Transform ray into world space
    if( !m_transform.empty() )
    {
        origin    = m_transform[0].transformPoint( origin );
        direction = normalize( m_transform[0].transformVector( direction ) );
    }


    // Create ray
    transformed_ray.setOrigin( origin );
    transformed_ray.setDirection( direction );
    transformed_ray.setTime( lerp( m_shutter_open_time, 
                                   m_shutter_close_time,
                                   sample.time ) );
    transformed_ray.setTMax( 1e15f );
}
