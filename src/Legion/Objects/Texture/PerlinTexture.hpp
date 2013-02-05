
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


/// \file PerlinTexture.hpp
/// Pure virtual interface for Texture Shaders

#ifndef LEGION_OBJECTS_TEXTURE_PERLIN_TEXTURE_H_
#define LEGION_OBJECTS_TEXTURE_PERLIN_TEXTURE_H_

#include <Legion/Objects/Texture/ProceduralTexture.hpp>
#include <Legion/Common/Math/Vector.hpp>
#include <Legion/Core/Color.hpp>


namespace legion
{


/// Perlin value textures.
class PerlinTexture : public ProceduralTexture
{
public:
    PerlinTexture( Context* context );

    ~PerlinTexture();

    void        setColors( const Color& c0, const Color& c1 );

    Type        getType()const;
    unsigned    getValueDim()const;

    const char* name()const;
    const char* proceduralFunctionName()const;

    void setVariables( VariableContainer& ) const;
private:
    Color m_c0;
    Color m_c1;
};


}

#endif // LEGION_OBJECTS_TEXTURE_PERLIN_TEXTURE_H_
