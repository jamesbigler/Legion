 
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

/// \file Util.hpp
/// Util

#ifndef LEGION_COMMON_UTIL_UTIL_HPP_
#define LEGION_COMMON_UTIL_UTIL_HPP_

#include <Legion/Common/Math/Vector.hpp>
#include <cassert>

namespace legion
{

inline unsigned index2DTo1D( const Index2& index, const Index2& dimensions )
{
    assert( index.x() < dimensions.x() && 
            index.y() < dimensions.y() );

    return index.y()*dimensions.x() + index.x();
}


inline Index2 index1DTo2D( unsigned index, const Index2& dimensions )
{
    assert( index < dimensions.x() * dimensions.y() );
    return Index2( index % dimensions.x(), index / dimensions.x() );
}

}


#endif // LEGION_COMMON_UTIL_UTIL_HPP_
