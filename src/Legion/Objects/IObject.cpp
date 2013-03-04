
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



#include <Legion/Objects/IObject.hpp>

using namespace legion;


IObject::IObject( Context* context ) 
    : m_context( context ),
      m_plugin_context( context->getPluginContext() )
{
}



IObject::~IObject()
{
}


Context* IObject::getContext()
{ 
    return m_context;
}


PluginContext* IObject::getPluginContext()
{ 
    return &m_plugin_context;
}


optix::Buffer IObject::createOptiXBuffer( unsigned type,
                                          RTformat format,
                                          unsigned width,
                                          unsigned height,
                                          unsigned depth )
{
    
    optix::Context context = m_plugin_context.getOptiXContext();
    optix::Buffer buffer   = context->createBuffer( type, format );
    if( depth != 0 )
    {
        LEGION_ASSERT( width != 0u && depth != 0u );
        buffer->setSize( width, height, depth );
    }
    else if( height != 0u )
    {
        LEGION_ASSERT( width != 0u );
        buffer->setSize( width, height );
    }
    else if( width != 0u )
    {
        buffer->setSize( width );
    }

    return buffer;
}


optix::TextureSampler IObject::createOptiXTextureSampler(
        optix::Buffer      buffer,
        RTwrapmode         wrap,
        RTfiltermode       filter,
        RTtextureindexmode index,
        RTtexturereadmode  read,
        float              aniso
        )
{
    optix::Context        context = m_plugin_context.getOptiXContext();
    optix::TextureSampler sampler = context->createTextureSampler();

    sampler->setWrapMode     ( 0, wrap );
    sampler->setWrapMode     ( 1, wrap );
    sampler->setWrapMode     ( 2, wrap );
    sampler->setIndexingMode ( index   );
    sampler->setReadMode     ( read    );
    sampler->setMaxAnisotropy( aniso   );
    sampler->setMipLevelCount( 1u      );
    sampler->setArraySize    ( 1u      );

    sampler->setBuffer( 0u, 0u, buffer );
    sampler->setFilteringModes( filter, filter, RT_FILTER_NONE );
    return sampler;
}


void IObject::launchOptiX( const Index2& dimensions )
{
    m_plugin_context.getOptiXContext()->launch( 
        0, dimensions.x(), dimensions.y() );
}
