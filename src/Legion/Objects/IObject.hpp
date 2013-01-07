
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


#ifndef LEGION_OBJECTS_IOBJECT_OBJECT_HPP_
#define LEGION_OBJECTS_IOBJECT_OBJECT_HPP_

#include <optixu/optixpp_namespace.h>

namespace legion
{

class Context;
class VariableContainer;

class IObject
{
public:
    IObject( Context* context ) : m_context( context ) {}

    Context* getContext() { return m_context; }
    
    virtual ~IObject() {}

    virtual void setVariables( VariableContainer& container ) const = 0;
    
    optix::Buffer createBuffer( unsigned type,
                                RTformat format, 
                                RTsize   width=0u,
                                RTsize   heidth=0u,
                                RTsize   depth=0u );

private:
    Context* m_context;
};

}


#endif // LEGION_OBJECTS_IOBJECTS_OBJECT_HPP_
