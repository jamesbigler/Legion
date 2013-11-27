
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


/// \file ITexture.hpp
/// Pure virtual interface for Texture Shaders

#ifndef LEGION_OBJECTS_TEXTURE_ITEXTURE_HPP_
#define LEGION_OBJECTS_TEXTURE_ITEXTURE_HPP_

#include <Legion/Objects/IObject.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>


namespace legion
{

/// Pure virtual interface for Textures.
class LCLASSAPI ITexture : public IObject
{
public:
    static const unsigned MAX_VALUE_DIM=4u;
    static const int      NULL_TEX_ID;
    static const char*    NULL_TEXTURE_PROCEDURE;

    enum Type
    {
        TYPE_CONSTANT=0,
        TYPE_IMAGE,
        TYPE_PROCEDURAL
    };

    LAPI ITexture( Context* context );
    LAPI virtual ~ITexture();

    LAPI virtual Type     getType()const=0;

    LAPI virtual unsigned getValueDim()const=0;

    //
    // For constant valued textures
    //
    LAPI virtual void getConstValue( float* val )const;

    //
    // For image textures 
    //
    LAPI virtual int getTexID()const;


    //
    // For procedural functions
    //
    LAPI virtual const char* name()const;
    LAPI virtual const char* proceduralFunctionName()const;
};


}

#endif // LEGION_OBJECTS_TEXTURE_ITEXTURE_HPP_
