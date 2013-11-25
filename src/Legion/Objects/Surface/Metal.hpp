
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

#ifndef LEGION_OBJECTS_SURFACE_METAL_HPP_
#define LEGION_OBJECTS_SURFACE_METAL_HPP_

#include <Legion/Objects/Surface/ISurface.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>
#include <string>
#include <map>

namespace legion
{

class VariableContainer;
class ITexture;
class ConstantTexture;
class Color;

class LAPI Metal : public ISurface
{
public:
    LAPI static ISurface* create( Context* context, const Parameters& params );

    LAPI Metal( Context* context );
    LAPI ~Metal();
    LAPI 
    LAPI void setMetalType  ( const std::string& type     );
    LAPI void setReflectance( const ITexture* reflectance );
    LAPI void setAlpha      ( const ITexture* alpha       );
    LAPI void setEta        ( const ITexture* eta         );
    LAPI void setK          ( const ITexture* k           );
    LAPI 
    LAPI const char* name()const;
    LAPI const char* sampleBSDFFunctionName()const;
    LAPI const char* evaluateBSDFFunctionName()const;
    LAPI const char* pdfFunctionName()const;
    LAPI const char* emissionFunctionName()const;

    LAPI void setVariables( VariableContainer& container ) const ;


private:
    // TODO: Get rid of static var
    typedef std::map<std::string, std::pair<Color, Color> > MetalLookup;
    static MetalLookup s_metal_lookup;
    
    static void initializeMetalLookup();
    static void addMetalLookup(
            const std::string& name,
            const Color& eta,
            const Color& k );


    const ITexture* m_reflectance;
    const ITexture* m_alpha;
    const ITexture* m_eta;
    const ITexture* m_k;
    
    std::auto_ptr<ConstantTexture> m_metal_type_eta;
    std::auto_ptr<ConstantTexture> m_metal_type_k;
};

}

#endif // LEGION_OBJECTS_SURFACE_METAL_HPP_
