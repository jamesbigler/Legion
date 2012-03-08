
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

using namespace legion;

namespace
{
    const float PI_4 = static_cast<float>( M_PI ) / 4.0f;
    const float PI_2 = static_cast<float>( M_PI ) / 2.0f;
}

/*
legion::Vector2 legion::squareToDisk( const legion::Vector2& sample )
{
    const float PI_4 = static_cast<float>( M_PI ) / 4.0f;
    const float PI_2 = static_cast<float>( M_PI ) / 2.0f;

    const float a = 2.0f * sample.x() - 1.0f;
    const float b = 2.0f * sample.y() - 1.0f;

    float phi = PI_4 * ( b/a );
    float r   = a;
    //if( a*a > b*b ) 
    if( abs( a ) <= abs( b ) ) 
    {
        r = b;
        phi = PI_4 * ( a/b ) + PI_2;
    }
    return Vector2( r*cosf( phi ), r*sinf( phi ) );
}
*/
/*
legion::Vector2 legion::squareToDisk( const legion::Vector2& sample )
{
    const float a = 2.0f * sample.x() - 1.0f;
    const float b = 2.0f * sample.y() - 1.0f;

    float phi,r;
    //if( a*a > b*b ) 
    if( abs( a ) > abs( b ) ) 
    {
        r = a;
        phi = PI_4 * ( b/a );
    }
    else 
    {
        r = b;
        phi = PI_4 * ( a/b ) + PI_2;
    }
    return Vector2( r*cosf( phi ), r*sinf( phi ) );
}
*/
