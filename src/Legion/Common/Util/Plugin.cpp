
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

#include <Legion/Common/Util/Plugin.hpp>
#include <Legion/Common/Util/Assert.hpp>

#include <map>

using namespace legion;

class PluginManager::Impl
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

    Impl( Context* ctx ) : m_context( ctx ) {}
    ~Impl() {}

    void registerCamera     ( const std::string& name, CameraCreator      );
    void registerDisplay    ( const std::string& name, DisplayCreator     );
    void registerEnvironment( const std::string& name, EnvironmentCreator );
    void registerGeometry   ( const std::string& name, GeometryCreator    );
    void registerLight      ( const std::string& name, LightCreator       );
    void registerRenderer   ( const std::string& name, RendererCreator    );
    void registerSurface    ( const std::string& name, SurfaceCreator     );
    void registerTexture    ( const std::string& name, TextureCreator     );

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

#define DEFINE_PLUGIN_IMPL_FUNCTIONS( plugin, plugin_map )                     \
    void PluginManager::Impl::register ## plugin(                              \
            const std::string& name,                                           \
            plugin ## Creator creator )                                        \
    {                                                                          \
        LEGION_ASSERT( creator != 0 );                                         \
        LEGION_ASSERT( plugin_map.count( name ) == 0 );                        \
        plugin_map.insert( std::make_pair( name, creator ) );                  \
    }                                                                          \
                                                                               \
    I ## plugin* PluginManager::Impl::create ## plugin(                        \
            const std::string& name,                                           \
            const Parameters& p )                                              \
    {                                                                          \
        plugin ## Creators::iterator it = plugin_map.find( name );             \
        LEGION_ASSERT( it != plugin_map.end() );                               \
        return it->second( m_context, p );                                     \
    }


DEFINE_PLUGIN_IMPL_FUNCTIONS( Camera,      m_camera_creators      );
DEFINE_PLUGIN_IMPL_FUNCTIONS( Display,     m_display_creators     );
DEFINE_PLUGIN_IMPL_FUNCTIONS( Environment, m_environment_creators );
DEFINE_PLUGIN_IMPL_FUNCTIONS( Geometry,    m_geometry_creators    );
DEFINE_PLUGIN_IMPL_FUNCTIONS( Light,       m_light_creators       );
DEFINE_PLUGIN_IMPL_FUNCTIONS( Surface,     m_surface_creators     );
DEFINE_PLUGIN_IMPL_FUNCTIONS( Renderer,    m_renderer_creators    );
DEFINE_PLUGIN_IMPL_FUNCTIONS( Texture,     m_texture_creators     );



//-----------------------------------------------------------------------------
//
// PluginManager 
//
//-----------------------------------------------------------------------------
PluginManager::PluginManager( Context* ctx )
    : m_impl( new Impl( ctx ) )
{
}


PluginManager::~PluginManager()
{
}

#define DEFINE_PLUGIN_FUNCTIONS( plugin )                                      \
    template <>                                                                \
    void PluginManager::registerPlugin<I ## plugin>(                           \
            const std::string& name,                                           \
            I ## plugin* (*create)( Context* ctx, const Parameters& params ) ) \
    {                                                                          \
        m_impl->register ## plugin( name, create );                            \
    }                                                                          \
                                                                               \
    template <>                                                                \
    I ## plugin* PluginManager::create<I ## plugin>(                           \
            const std::string& plugin_name,                                    \
            const Parameters& params )                                         \
    {                                                                          \
        return m_impl->create ## plugin( plugin_name, params );                \
    }


DEFINE_PLUGIN_FUNCTIONS( Camera      );
DEFINE_PLUGIN_FUNCTIONS( Display     );
DEFINE_PLUGIN_FUNCTIONS( Environment );
DEFINE_PLUGIN_FUNCTIONS( Geometry    );
DEFINE_PLUGIN_FUNCTIONS( Light       );
DEFINE_PLUGIN_FUNCTIONS( Renderer    );
DEFINE_PLUGIN_FUNCTIONS( Surface     );
DEFINE_PLUGIN_FUNCTIONS( Texture     );
