
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

#ifndef LEGION_RENDERER_RENDERER_IRENDERER_HPP_
#define LEGION_RENDERER_RENDERER_IRENDERER_HPP_

#include <Legion/Common/Math/Vector.hpp>

namespace legion
{

class VariableContainer;
class IDisplay;

/// Owns the optix raygen program, the output framebuffer,  and all ray
/// scheduling
class IRenderer
{
public:
    IRenderer() 
        : m_display( 0u),
          m_resolution( 1280u, 960u ),
          m_spp( 64u )
    {}

    virtual ~IRenderer() {}

    void setDisplay( IDisplay* display );

    void setSamplesPerPixel( unsigned spp ) {  m_spp = spp; }
    
    void setResolution( const Index2& resolution ) { m_resolution = resolution; }

    virtual const char* name()const=0;
    virtual const char* rayGenProgramName()const=0;

    virtual void setVariables( VariableContainer& container )const = 0;

protected:
    IDisplay* m_display;
    Index2    m_resolution;
    unsigned  m_spp;
};

}

#endif // LEGION_RENDERER_RENDERER_IRENDERER_HPP_
