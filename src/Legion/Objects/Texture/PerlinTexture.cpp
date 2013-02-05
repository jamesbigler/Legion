
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


#include <Legion/Objects/Texture/PerlinTexture.hpp>
#include <Legion/Core/VariableContainer.hpp>

using namespace legion;


PerlinTexture::PerlinTexture( Context* context )
    : ProceduralTexture( context ),
      m_c0( 0.9f, 0.9f, 0.9f ),
      m_c1( 0.1f, 0.1f, 0.1f )
{
}


PerlinTexture::~PerlinTexture()
{
}


void PerlinTexture::setColors( const Color& c0, const Color& c1 )
{
    m_c0 = c0;
    m_c1 = c1;
}


ITexture::Type PerlinTexture::getType()const
{
    return TYPE_PROCEDURAL;
}


unsigned PerlinTexture::getValueDim()const
{
    return 4u;
}


const char* PerlinTexture::name()const
{
    return "PerlinTexture";
}


const char* PerlinTexture::proceduralFunctionName()const
{
    return "perlinTextureProc";
}


void PerlinTexture::setVariables( VariableContainer& container )const 
{
    container.setFloat( "c0", m_c0 );
    container.setFloat( "c1", m_c1 );
}
