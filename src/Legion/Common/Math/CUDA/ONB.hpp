 
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

#include <optixu/optixu_math_namespace.h>
#include <Legion/Common/Util/Preprocessor.hpp>

namespace legion
{

class ONB
{
public:
  /// Assumes normal is normalized
  LDEVICE ONB( const float3& normal );
  
  /// Transforms the Vector from world space to the local space 
  LDEVICE float3 transform( const float3& v)const;

  /// Transforms the Vector from the local space to world
  LDEVICE float3 inverseTransform( const float3& v)const;
  
  LDEVICE float3 tangent()const;
  LDEVICE float3 binormal()const;
  LDEVICE float3 normal()const;

private:
  float3 m_tangent;
  float3 m_binormal;
  float3 m_normal;
};

LDEVICE inline ONB::ONB( const float3& normal )
{
    m_normal = normal;

    if( fabs( m_normal.x ) > fabs( m_normal.z ) )
    {
        m_binormal.x = -m_normal.y;
        m_binormal.y =  m_normal.x;
        m_binormal.z =  0.0f;
    }
    else
    {
        m_binormal.x =  0.0f;
        m_binormal.y = -m_normal.z;
        m_binormal.z =  m_normal.y;
    }

    m_binormal = optix::normalize(m_binormal);
    m_tangent  = optix::cross( m_binormal, m_normal );
}
  

LDEVICE inline float3 ONB::tangent()const
{
    return m_tangent;
}


LDEVICE inline float3 ONB::binormal()const
{
    return m_binormal;
}


LDEVICE inline float3 ONB::normal()const
{
    return m_normal;
}


LDEVICE inline float3 ONB::transform( const float3& v)const
{

    return make_float3( optix::dot( v, m_tangent  ),
                        optix::dot( v, m_binormal ),
                        optix::dot( v, m_normal   ) );
}


LDEVICE inline float3 ONB::inverseTransform( const float3& v )const
{
    return  v.x*m_tangent + v.y*m_binormal + v.z*m_normal;
}

}


#endif // LEGION_COMMON_MATH_ONB_HPP_
