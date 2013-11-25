
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


/// \file ImageTexture.hpp
/// Pure virtual interface for Texture Shaders

#ifndef LEGION_OBJECTS_TEXTURE_IMAGE_TEXTURE_HPP_
#define LEGION_OBJECTS_TEXTURE_IMAGE_TEXTURE_HPP_

#include <Legion/Objects/Texture/ITexture.hpp>
#include <Legion/Common/Math/Vector.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>


namespace legion
{

/// Image value textures.
class LAPI ImageTexture : public ITexture
{
public:
    LAPI static ITexture* create( Context* context, const Parameters& params );

    LAPI ImageTexture( Context* context );
    LAPI ~ImageTexture();

    LAPI void set( const std::string& filename ); 
    LAPI 
    LAPI Type     getType()const;
    LAPI unsigned getValueDim()const;
    LAPI int      getTexID()const;

private:
    unsigned    m_val_dim;
    int         m_tex_id;
    std::string m_filename;
};


}

#endif // LEGION_OBJECTS_TEXTURE_IMAGE_TEXTURE_HPP_
