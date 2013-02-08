
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

#ifndef LEGION_COMMON_UTIL_FACTORY_H_
#define LEGION_COMMON_UTIL_FACTORY_H_

#include <string>
#include <map>

namespace legion
{

class Context;
class Parameters;

class ICamera;
class IDisplay;
class IEnvironment;
class IGeometry;
class ILight;
class IRenderer;
class ISurface;
class ITexture;


class Factory 
{
public:
    typedef ICamera*     (*CameraCreator     )( Context*, const Parameters& );
    typedef IDisplay*    (*DisplayCreator    )( Context*, const Parameters& );
    typedef IEnvironment*(*EnvironmentCreator)( Context*, const Parameters& );
    typedef IGeometry*   (*GeometryCreator   )( Context*, const Parameters& );
    typedef ILight*      (*LightCreator      )( Context*, const Parameters& );
    typedef IRenderer*   (*RendererCreator   )( Context*, const Parameters& );
    typedef ISurface*    (*SurfaceCreator    )( Context*, const Parameters& );
    typedef ITexture*    (*TextureCreator    )( Context*, const Parameters& );

    Factory( Context* ctx ) : m_context( ctx ) {}
    ~Factory() {}

    void registerObject( const std::string& name, CameraCreator      );
    void registerObject( const std::string& name, DisplayCreator     );
    void registerObject( const std::string& name, EnvironmentCreator );
    void registerObject( const std::string& name, GeometryCreator    );
    void registerObject( const std::string& name, LightCreator       );
    void registerObject( const std::string& name, RendererCreator    );
    void registerObject( const std::string& name, SurfaceCreator     );
    void registerObject( const std::string& name, TextureCreator     );

    ICamera*   createCamera  ( const std::string& name, const Parameters& p );
    IDisplay*  createDisplay ( const std::string& name, const Parameters& p );
    IGeometry* createGeometry( const std::string& name, const Parameters& p );
    ILight*    createLight   ( const std::string& name, const Parameters& p );
    IRenderer* createRenderer( const std::string& name, const Parameters& p );
    ISurface*  createSurface ( const std::string& name, const Parameters& p );
    ITexture*  createTexture ( const std::string& name, const Parameters& p );
    IEnvironment* createEnvironment  ( const std::string& name,
                                       const Parameters& p );

private:
    typedef std::map<std::string, CameraCreator>      CameraCreators;
    typedef std::map<std::string, DisplayCreator>     DisplayCreators;
    typedef std::map<std::string, EnvironmentCreator> EnvironmentCreators;
    typedef std::map<std::string, GeometryCreator>    GeometryCreators;
    typedef std::map<std::string, LightCreator>       LightCreators;
    typedef std::map<std::string, RendererCreator>    RendererCreators;
    typedef std::map<std::string, SurfaceCreator>     SurfaceCreators;
    typedef std::map<std::string, TextureCreator>     TextureCreators;

    Context*              m_context;

    CameraCreators        m_camera_creators;
    DisplayCreators       m_display_creators;
    EnvironmentCreators   m_environment_creators;
    GeometryCreators      m_geometry_creators;
    LightCreators         m_light_creators;
    RendererCreators      m_renderer_creators;
    SurfaceCreators       m_surface_creators;
    TextureCreators       m_texture_creators;
};

}

#endif // LEGION_COMMON_UTIL_FACTORY_H_
