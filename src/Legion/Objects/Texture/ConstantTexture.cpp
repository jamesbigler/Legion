
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


#include <Legion/Objects/Texture/ConstantTexture.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Core/Color.hpp>

using namespace legion;


ConstantTexture::ConstantTexture( Context* context )
    : ITexture( context ),
      m_val_dim( 1u )
{
    m_val[0] = 
    m_val[1] = 
    m_val[2] = 
    m_val[3] = 0.0f;
}


ConstantTexture::~ConstantTexture()
{
}


void ConstantTexture::set( const Color&   c )
{
    m_val_dim = 4u;
    m_val[0] = c.r();
    m_val[1] = c.g();
    m_val[2] = c.b();
    m_val[2] = 1.0f;
}


void ConstantTexture::set( const float&   f )
{
    m_val_dim = 1u;
    m_val[0] = f; 
}


void ConstantTexture::set( const Vector2& v )
{
    m_val_dim = 2u;
    m_val[0] = v.x();
    m_val[1] = v.y();
}


void ConstantTexture::set( const Vector4& v )
{
    m_val_dim = 4u;
    m_val[0] = v.x();
    m_val[1] = v.y();
    m_val[2] = v.z();
    m_val[3] = v.w();
}

    
ITexture::Type ConstantTexture::getType()const
{
    return TYPE_CONSTANT;
}


unsigned ConstantTexture::getValueDim()const
{
    return m_val_dim;
}


void ConstantTexture::getConstValue( float* val )const
{
    for( unsigned i = 0u; i < m_val_dim; ++i )
        val[i] = m_val[i];
    for( unsigned i = m_val_dim; i < MAX_VALUE_DIM; ++i )
        val[i] = 0.0f;
}

