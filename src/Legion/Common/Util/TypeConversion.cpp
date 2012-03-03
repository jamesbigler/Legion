 

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

#include <Legion/Common/Util/TypeConversion.hpp>

using namespace legion;



Color legion::toColor( const Vector3& v )
{
    Vector3 t( normalize( v ) );
    t = ( t + Vector3( 1.0f ) ) * 0.5f;
    return Color( t.x(), t.y(), t.z() );
}


Color legion::toColor( const optix::float3& v )
{
    return toColor( toVector3( v ) );
}


Vector3 legion::toVector3( const optix::float3& v )
{
    return Vector3( v.x, v.y, v.z );
}


Vector4 legion::toVector4( const optix::float4& v )
{
    return Vector4( v.x, v.y, v.z, v.w );
}



