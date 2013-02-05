
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

/// \file ITexture.cpp

#include <Legion/Objects/Texture/ITexture.hpp>

using namespace legion;

const int      ITexture::NULL_TEX_ID            = -1;
const char*    ITexture::NULL_TEXTURE_PROCEDURE = "NULLTEXTUREPROC";

ITexture::ITexture( Context* context ) 
    : IObject( context )
{
}


ITexture::~ITexture() {} 


void ITexture::getConstValue( float* val )const
{
    val[0] = val[1] = val[2] = val[3] = 0.0f;
}


int ITexture::getTexID()const
{
    return NULL_TEX_ID;
}


const char* ITexture::name()const
{
    return NULL_TEXTURE_PROCEDURE;
}


const char* ITexture::proceduralFunctionName()const
{
    return NULL_TEXTURE_PROCEDURE;
}
