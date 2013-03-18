
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


#include <Legion/Objects/Texture/CheckerTexture.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Common/Util/Parameters.hpp>

using namespace legion;

ITexture* CheckerTexture::create( Context* context, const Parameters& params )
{
    CheckerTexture* checker = new CheckerTexture( context );

    checker->setColors( params.get( "c0", Color( 0.9f, 0.9f, 0.9f ) ), 
                        params.get( "c1", Color( 0.1f, 0.1f, 0.1f ) ) ); 
    checker->setScale(  params.get( "scale", 10.0f ) );

    return checker;
}


CheckerTexture::CheckerTexture( Context* context )
    : ProceduralTexture( context ),
      m_c0( 0.9f, 0.9f, 0.9f ),
      m_c1( 0.1f, 0.1f, 0.1f ),
      m_scale( 10.0f )
{
}


CheckerTexture::~CheckerTexture()
{
}


void CheckerTexture::setColors( const Color& c0, const Color& c1 )
{
    m_c0 = c0;
    m_c1 = c1;
}


void CheckerTexture::setScale( float scale )
{
    m_scale = scale;
}


ITexture::Type CheckerTexture::getType()const
{
    return TYPE_PROCEDURAL;
}


unsigned CheckerTexture::getValueDim()const
{
    return 4u;
}


const char* CheckerTexture::name()const
{
    return "CheckerTexture";
}


const char* CheckerTexture::proceduralFunctionName()const
{
    return "checkerTextureProc";
}


void CheckerTexture::setVariables( VariableContainer& container )const 
{
    container.setFloat( "c0", m_c0 );
    container.setFloat( "c1", m_c1 );
    container.setFloat( "scale", m_scale );
}
