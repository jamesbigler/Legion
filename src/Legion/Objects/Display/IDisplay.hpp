
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

#ifndef LEGION_OBJECTS_DISPLAY_IDISPLAY_HPP_
#define LEGION_OBJECTS_DISPLAY_IDISPLAY_HPP_

#include <Legion/Common/Math/Vector.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>
#include <Legion/Objects/IObject.hpp>
#include <optixu/optixpp_namespace.h>

namespace legion
{

class VariableContainer;

class LAPI IDisplay: public IObject
{
public:
    typedef unsigned char Byte;
    enum FrameType
    {
        NONE  = 0,    //< Report neither
        BYTE  = 1,    //< Report four byte  pixels: 0xffRRGGBB 
        FLOAT = 2,    //< Report four float pixels: R,G,B,1.0f
        BOTH  = 3     //< Report both ( BYTE | FLOAT )
    };

    LAPI IDisplay( Context* context ) : IObject( context ) {}

    /// Report desired update frame format 
    LAPI virtual FrameType getUpdateFrameType()=0;

    /// Report desired complete frame format 
    LAPI virtual FrameType getCompleteFrameType()=0;

    LAPI virtual ~IDisplay() {}
    
    /// Set the number of expected frame updates until complete
    LAPI virtual void setUpdateCount( unsigned m_update_count             )=0;

    /// Scene processing has begun
    LAPI virtual void beginScene    ( const std::string& scene_name       )=0;
    
    /// Scene processing has begun
    LAPI virtual void beginFrame    ( const Index2& resolution            )=0;

    /// Update frame buffer with current partially finished results
    LAPI virtual bool updateFrame   ( const float* fpix, const Byte* cpix )=0;

    /// Complete frame buffer 
    LAPI virtual void completeFrame ( const float* fpix, const Byte* cpix )=0;
};

}

#endif // LEGION_OBJECTS_DISPLAY_IDISPLAY_HPP_
