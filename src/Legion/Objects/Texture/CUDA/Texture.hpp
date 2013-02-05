
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

#ifndef LEGION_OBJECTS_TEXTURE_CUDA_TEXTURE_HPP_
#define LEGION_OBJECTS_TEXTURE_CUDA_TEXTURE_HPP_

#include <Legion/Objects/cuda_common.hpp>

#define legionDeclareTexture( ret_type, name )                                 \
    typedef ret_type             name ## _ret_type__;                          \
    rtDeclareVariable( unsigned, name ## _type__  , , );                       \
    rtDeclareVariable( ret_type, name ## _const__ , , );                       \
    rtDeclareVariable( int     , name ## _texid__ , , );                       \
    rtCallableProgram( ret_type, name ## _proc__  , ( float2, float3 ) );

      
#define legionTex( name, uv, p )                                               \
    ( name ## _type__ == 0 ? name ## _const__                               :  \
      name ## _type__ == 1 ? optix::rtTex2D<name ## _ret_type__>(              \
                                 name ## _texid__, (uv).x, (uv).y )         :  \
                             name ## _proc__( (uv), (p) ) ) 


#endif // LEGION_OBJECTS_TEXTURE_CUDA_TEXTURE_HPP_
