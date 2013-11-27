
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

#ifndef LEGION_OBJECTS_SURFACE_BECKMANN_HPP_
#define LEGION_OBJECTS_SURFACE_BECKMANN_HPP_

#include <Legion/Objects/Surface/ISurface.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>

namespace legion
{

class VariableContainer;
class ITexture;

class LCLASSAPI Beckmann : public ISurface
{
public:
    LAPI static ISurface* create( Context* context, const Parameters& params );

    LAPI Beckmann( Context* context );
    LAPI ~Beckmann();

    LAPI void setReflectance( const ITexture* reflectance );
    LAPI void setAlpha      ( const ITexture* alpha);

    LAPI const char* name()const;
    LAPI const char* sampleBSDFFunctionName()const;
    LAPI const char* evaluateBSDFFunctionName()const;
    LAPI const char* pdfFunctionName()const;
    LAPI const char* emissionFunctionName()const;

    LAPI void setVariables( VariableContainer& container ) const ;

private:
    const ITexture* m_reflectance;
    const ITexture* m_alpha;
};

}

#endif // LEGION_OBJECTS_SURFACE_BECKMANN_HPP_
