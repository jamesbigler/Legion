
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


#include <Legion/Objects/Texture/FresnelSchlickTexture.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Common/Util/Parameters.hpp>

using namespace legion;

ITexture* FresnelSchlickTexture::create( Context* context, const Parameters& params )
{
    FresnelSchlickTexture* fresnel = new FresnelSchlickTexture( context );
    fresnel->setEta( params.get( "eta", 1.5f ) ); 
    return fresnel;
}


FresnelSchlickTexture::FresnelSchlickTexture( Context* context )
    : ProceduralTexture( context ),
      m_eta(1.5f )
{
}


FresnelSchlickTexture::~FresnelSchlickTexture()
{
}


void FresnelSchlickTexture::setEta( float eta )
{
    m_eta = eta;
}


ITexture::Type FresnelSchlickTexture::getType()const
{
    return TYPE_PROCEDURAL;
}


unsigned FresnelSchlickTexture::getValueDim()const
{
    return 1u;
}


const char* FresnelSchlickTexture::name()const
{
   return "FresnelSchlickTexture";
}


const char* FresnelSchlickTexture::proceduralFunctionName()const
{
    return "fresnelSchlickTextureProc";
}


void FresnelSchlickTexture::setVariables( VariableContainer& container )const 
{
    container.setFloat( "eta", m_eta );
}