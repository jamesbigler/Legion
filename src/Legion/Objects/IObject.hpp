
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

#include <Legion/Core/Context.hpp>
#include <Legion/Core/PluginContext.hpp>
#include <optixu/optixpp_namespace.h>

namespace legion
{

class VariableContainer;

class IObject
{
public:
    IObject( Context* context );

    virtual ~IObject();

    Context*       getContext();
    PluginContext* getPluginContext();

    void launchOptiX( const Index2& dimensions );

    optix::Buffer createOptiXBuffer(
            unsigned type,
            RTformat format,
            unsigned width  = 0u,
            unsigned height = 0u,
            unsigned depth  = 0u );
    
    optix::TextureSampler createOptiXTextureSampler(
            optix::Buffer      buffer,
            RTwrapmode         wrap   = RT_WRAP_REPEAT,
            RTfiltermode       filter = RT_FILTER_LINEAR,
            RTtextureindexmode index  = RT_TEXTURE_INDEX_NORMALIZED_COORDINATES,
            RTtexturereadmode  read   = RT_TEXTURE_READ_NORMALIZED_FLOAT, 
            float              aniso  = 1.0f
            );

    virtual void setVariables( VariableContainer& ) const {}

private:
    Context*       m_context;
    PluginContext& m_plugin_context;
};

}


#endif // LEGION_OBJECTS_IOBJECTS_OBJECT_HPP_
