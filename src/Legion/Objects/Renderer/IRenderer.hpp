
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

#ifndef LEGION_OBJECTS_RENDERER_IRENDERER_HPP_
#define LEGION_OBJECTS_RENDERER_IRENDERER_HPP_

#include <Legion/Common/Math/Vector.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>
#include <Legion/Objects/IObject.hpp>
#include <optixu/optixpp_namespace.h>

namespace legion
{

class VariableContainer;
class IDisplay;

/// Owns the optix raygen program, the output framebuffer,  and all ray
/// scheduling
class LCLASSAPI IRenderer : public IObject
{
public:
    LAPI IRenderer( Context* context ) 
        : IObject( context ),
          m_display( 0u),
          m_resolution( 1024, 768u ),
          m_spp( 64u ),
          m_max_diff_depth( 2u ),
          m_max_spec_depth( 8u )
    {}

    LAPI virtual ~IRenderer() {}

    LAPI void      setDisplay( IDisplay* display )      { m_display = display; }
    LAPI IDisplay* getDisplay()const                    { return m_display;    }

    LAPI void     setSamplesPerPixel( unsigned spp )    { m_spp = spp;  }
    LAPI unsigned getSamplesPerPixel()const             { return m_spp; }

    LAPI void     setResolution( const Index2& res )    { m_resolution = res;  }
    LAPI Index2   getResolution()const                  { return m_resolution; }

    LAPI void     setMaxDiffuseDepth( unsigned depth )  { m_max_diff_depth = depth; }
    LAPI void     setMaxSpecularDepth( unsigned depth ) { m_max_spec_depth = depth; }

    LAPI virtual const char*    name()const=0;
    LAPI virtual const char*    rayGenProgramName()const=0;
    LAPI virtual void           preRender( VariableContainer& container )=0;
    LAPI virtual void           render( VariableContainer& container )=0;

protected:
    IDisplay* m_display;
    Index2    m_resolution;
    unsigned  m_spp;
    unsigned  m_max_diff_depth;
    unsigned  m_max_spec_depth;
};

}

#endif // LEGION_OBJECTS_RENDERER_IRENDERER_HPP_
