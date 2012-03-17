 
// Copyright (C) 2011 R. Keith Morley
//
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
// (MIT/X11 License)

/// \file ONB.hpp
/// ONB


#ifndef LEGION_COMMON_MATH_ONB_HPP_
#define LEGION_COMMON_MATH_ONB_HPP_

#include <Legion/Common/Math/Vector.hpp>

namespace legion
{

class ONB
{
public:
  /// Assumes normal is normalized
  ONB( const Vector3& normal );
  
  /// Transforms the Vector from world space to the local space 
  Vector3 transform( const Vector3& v)const;

  /// Transforms the Vector from the local space to world
  Vector3 inverseTransform( const Vector3& v)const;
  
  Vector3 tangent()const;
  Vector3 binormal()const;
  Vector3 normal()const;

private:
  Vector3 m_tangent;
  Vector3 m_binormal;
  Vector3 m_normal;
};

inline ONB::ONB( const Vector3& normal )
{
    m_normal = normal;

    if( fabs( m_normal.x() ) > fabs( m_normal.z() ) )
    {
        m_binormal.setX( -m_normal.y() );
        m_binormal.setY(  m_normal.x() );
        m_binormal.setZ(  0.0f );
    }
    else
    {
        m_binormal.setX(  0.0f );
        m_binormal.setY( -m_normal.z() );
        m_binormal.setZ(  m_normal.y() );
    }

    m_binormal = normalize(m_binormal);
    m_tangent  = cross( m_binormal, m_normal );
}
  

inline Vector3 ONB::tangent()const
{
    return m_tangent;
}


inline Vector3 ONB::binormal()const
{
    return m_binormal;
}


inline Vector3 ONB::normal()const
{
    return m_normal;
}


inline Vector3 ONB::transform( const Vector3& v)const
{

    return Vector3( dot( v, m_tangent  ),
                    dot( v, m_binormal ),
                    dot( v, m_normal   ) );
}


inline Vector3 ONB::inverseTransform( const Vector3& v )const
{
    return  v.x()*m_tangent + v.y()*m_binormal + v.z()*m_normal;
}

}


#endif // LEGION_COMMON_MATH_ONB_HPP_
