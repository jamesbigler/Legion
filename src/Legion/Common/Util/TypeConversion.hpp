 

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

/// \file TypeConversion.hpp
/// TypeConversion

#ifndef LEGION_TYPECONVERSION_HPP_
#define LEGION_TYPECONVERSION_HPP_

#include <Legion/Core/Color.hpp>
#include <Legion/Common/Math/Vector.hpp>
#include <optixu/optixu_vector_types.h>

namespace legion
{
    Vector3  toVector3( const optix::float3& v );
    Vector4  toVector4( const optix::float4& v );

    Color toColor( const Vector3& v );
    Color toColor( const optix::float3& v );
}


#endif // LEGION_TYPECONVERSION_HPP_
