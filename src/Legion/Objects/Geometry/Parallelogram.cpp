
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

/// \file Parallelogram.cpp

#include <Legion/Objects/Geometry/Parallelogram.hpp>
#include <Legion/Objects/VariableContainer.hpp>
#include <Legion/Common/Util/Parameters.hpp>


using namespace legion;
    
    
IGeometry* Parallelogram::create( Context* context, const Parameters& params )
{
    Parallelogram* pgram = new Parallelogram( context );
    const Vector3 anchor = params.get( "anchor", Vector3(-0.5f, 0.0f, -0.5f ) );
    const Vector3 U      = params.get( "U"     , Vector3( 1.0f, 0.0f,  0.0f ) );
    const Vector3 V      = params.get( "V"     , Vector3( 0.0f, 0.0f, -1.0f ) );
    pgram->setAnchorUV( anchor, U, V );
    return pgram;
}


Parallelogram::Parallelogram( Context* context )
    : IGeometry( context ),
      m_transform( Matrix::identity() ),
      m_anchor( -0.5f, -0.5f, -0.5f ),
      m_U( 1.0f, 0.0f, 0.0f ),
      m_V( 0.0f, 0.0f, -1.0f ),
      m_surface( 0 )
{
}


const char* Parallelogram::name()const
{
    return "Parallelogram";
}


const char* Parallelogram::intersectionFunctionName()const
{
    return "parallelogramIntersect";
}


const char* Parallelogram::boundingBoxFunctionName()const
{
    return "parallelogramBoundingBox";
}


const char* Parallelogram::sampleFunctionName()const
{
    return "parallelogramSample";
}


const char* Parallelogram::pdfFunctionName()const
{
    return "parallelogramPDF";
}


unsigned Parallelogram::numPrimitives()const
{
    return 1u;
}


float Parallelogram::area()const
{
    return cross( m_U, m_V ).length();
}


void Parallelogram::setTransform( const Matrix& transform )
{
    m_transform = transform;
}


Matrix Parallelogram::getTransform() const
{
    return m_transform;
}


void Parallelogram::setSurface( ISurface* surface )
{
    m_surface = surface;
}


ISurface* Parallelogram::getSurface()const
{
    return m_surface;
}


void Parallelogram::setAnchorUV(
    const Vector3& anchor,
    const Vector3& U,
    const Vector3& V )
{
    m_anchor = anchor;
    m_U      = U;
    m_V      = V;
}



void Parallelogram::setVariables( VariableContainer& container ) const
{
    const Vector3 normal = normalize( cross( m_U, m_V ) );
    const float   d      = dot( normal, m_anchor );
    const Vector4 plane  = Vector4( normal.x(), normal.y(), normal.z(), d );

    const Vector3 v1 = m_U / dot( m_U, m_U );
    const Vector3 v2 = m_V / dot( m_V, m_V );

    container.setFloat( "inv_area", 1.0f / area() );
    container.setFloat( "anchor",   m_anchor      );
    container.setFloat( "plane",    plane         );
    container.setFloat( "v1",       v1            );
    container.setFloat( "v2",       v2            );
    container.setFloat( "U",        m_U           );
    container.setFloat( "V",        m_V           );
}
