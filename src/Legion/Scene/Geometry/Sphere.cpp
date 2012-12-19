
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

#include <Legion/Scene/Geometry/Sphere.hpp>
#include <Legion/Common/Util/Logger.hpp>


using namespace legion;
    

IGeometry* Sphere::create( const Parameters& params )
{
    return new Sphere( params );
}

Sphere::Sphere()
    : m_radius( 1.0f ),
      m_center( 0.0f, 0.0f, 0.0f )
{
}


Sphere::Sphere( const Parameters& params)
    : m_radius( 1.0f ),
      m_center( 0.0f, 0.0f, 0.0f )
{
    LLOG_INFO << "\t\tSphere::Sphere( params );";
}


void Sphere::setVariables( VariableContainer& container ) const
{
}
