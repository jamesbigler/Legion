
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

#ifndef LEGION_CORE_CONTEXT_HPP_
#define LEGION_CORE_CONTEXT_HPP_

#include <Legion/Common/Util/Noncopyable.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>
#include <Legion/Common/Math/Vector.hpp>

#include <memory>

namespace legion
{

class ICamera;
class IDisplay;
class IGeometry;
class IEnvironment;
class ILight;
class IRenderer;
class ISurface;
class ITexture;

class Parameters;
class PluginContext;


class LCLASSAPI Context : public Noncopyable
{
public:
    class Impl;

    LAPI Context();
    LAPI ~Context();

    LAPI ICamera*      createCamera     ( const char* name, const Parameters& p );
    LAPI IDisplay*     createDisplay    ( const char* name, const Parameters& p );
    LAPI IEnvironment* createEnvironment( const char* name, const Parameters& p );
    LAPI IGeometry*    createGeometry   ( const char* name, const Parameters& p );
    LAPI ILight*       createLight      ( const char* name, const Parameters& p );
    LAPI IRenderer*    createRenderer   ( const char* name, const Parameters& p );
    LAPI ISurface*     createSurface    ( const char* name, const Parameters& p );
    LAPI ITexture*     createTexture    ( const char* name, const Parameters& p );


    LAPI void setRenderer   ( IRenderer*    renderer    );
    LAPI void setCamera     ( ICamera*      camera      );
    LAPI void setEnvironment( IEnvironment* environment );
    LAPI void addGeometry   ( IGeometry*    geometry    );
    LAPI void addLight      ( ILight*       light       );

    LAPI void setAccelCacheFileName( const char* name );
    
    LAPI IRenderer*  getRenderer();

    LAPI void addAssetPath( const std::string& path );

    LAPI void render();

    LAPI PluginContext& getPluginContext();

private:
    //std::unique_ptr<Impl> m_impl;
    std::auto_ptr<Impl> m_impl;
};


}
#endif // LEGION_CORE_CONTEXT_HPP_
