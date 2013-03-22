
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

/// \file Sphere.cpp

#include <Legion/Objects/Geometry/Sphere.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Common/Util/Parameters.hpp>
#include <Legion/Common/Math/Math.hpp>
#include <Legion/Objects/VariableContainer.hpp>


using namespace legion;
    

IGeometry* Sphere::create( Context* context, const Parameters& params )
{
    Sphere* sphere = new Sphere( context );
    sphere->setCenter( params.get( "center",  Vector3( 0.0f ) ) );
    sphere->setRadius( params.get( "radius",  0.5f            ) );
    return sphere;
}

Sphere::Sphere( Context* context )
    : IGeometry( context ),
      m_transform( Matrix::identity() ),
      m_radius( 0.5f ),
      m_center( 0.0f, 0.0f, 0.0f ),
      m_surface( 0 )
{
}


const char* Sphere::name()const
{
    return "Sphere";
}


const char* Sphere::intersectionFunctionName()const
{
    return "sphereIntersect";
}


const char* Sphere::boundingBoxFunctionName()const
{
    return "sphereBoundingBox";
}


const char* Sphere::sampleFunctionName()const
{
    return "sphereSample";
}

const char* Sphere::pdfFunctionName()const
{
    return "spherePDF";
}

unsigned Sphere::numPrimitives()const
{
    return 1u;
}


float Sphere::area()const
{
    return 4.0*legion::PI*m_radius*m_radius;
}


void Sphere::setTransform( const Matrix& transform )
{
    m_transform = transform;
}


Matrix Sphere::getTransform() const
{
    return m_transform;
}


void Sphere::setSurface( ISurface* surface )
{
    m_surface = surface;
}


ISurface* Sphere::getSurface()const
{
    return m_surface;
}


void Sphere::setCenter( Vector3 center )
{
    m_center = center;
}


void Sphere::setRadius( float radius )
{
    m_radius = radius;
}


void Sphere::setVariables( VariableContainer& container ) const
{
    container.setFloat( "center", m_center );
    container.setFloat( "radius", m_radius );
}
