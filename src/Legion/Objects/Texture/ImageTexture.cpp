
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


#include <Legion/Objects/Texture/ImageTexture.hpp>

#include <Legion/Common/Util/Image.hpp>
#include <Legion/Core/Color.hpp>
#include <Legion/Core/VariableContainer.hpp>

using namespace legion;


ImageTexture::ImageTexture( Context* context )
    : ITexture( context ),
      m_val_dim( 1u ),
      m_tex_id( -1 )
{
}


ImageTexture::~ImageTexture()
{
}

    
void ImageTexture::set( const std::string& filename )
{
    optix::Buffer buffer = createOptiXBuffer( 
            RT_BUFFER_INPUT,
            RT_FORMAT_FLOAT4
            );

    readOpenEXR( filename, buffer );

    optix::TextureSampler texture_sampler = createOptiXTextureSampler( buffer );
    m_tex_id = texture_sampler->getId();
}

    
ITexture::Type ImageTexture::getType()const
{
    return TYPE_IMAGE;
}


unsigned ImageTexture::getValueDim()const
{
    return m_val_dim;
}

int ImageTexture::getTexID()const
{
    return m_tex_id;
}
