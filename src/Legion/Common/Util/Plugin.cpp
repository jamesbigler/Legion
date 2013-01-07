
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
    typedef ICamera*  (*CameraCreator  )( Context*, const Parameters& );
    typedef IFilm*    (*FilmCreator    )( Context*, const Parameters& );
    typedef IGeometry*(*GeometryCreator)( Context*, const Parameters& );
    typedef ILight*   (*LightCreator   )( Context*, const Parameters& );
    typedef ISurface* (*SurfaceCreator )( Context*, const Parameters& );

    Impl( Context* ctx ) : m_context( ctx ) {}
    ~Impl() {}

    void registerCamera  ( const std::string& name, CameraCreator   );
    void registerFilm    ( const std::string& name, FilmCreator     );
    void registerGeometry( const std::string& name, GeometryCreator );
    void registerLight   ( const std::string& name, LightCreator    );
    void registerSurface ( const std::string& name, SurfaceCreator  );

    ICamera*  createCamera  ( const std::string& name, const Parameters& p );
    IFilm*    createFilm    ( const std::string& name, const Parameters& p );
    IGeometry*createGeometry( const std::string& name, const Parameters& p );
    ILight*   createLight   ( const std::string& name, const Parameters& p );
    ISurface* createSurface ( const std::string& name, const Parameters& p );

private:
    typedef std::map<std::string, CameraCreator>    CameraCreators;
    typedef std::map<std::string, FilmCreator>      FilmCreators;    
    typedef std::map<std::string, GeometryCreator>  GeometryCreators;
    typedef std::map<std::string, LightCreator>     LightCreators;
    typedef std::map<std::string, SurfaceCreator>   SurfaceCreators;

    Context*              m_context;
    CameraCreators        m_camera_creators;
    FilmCreators          m_film_creators;
    GeometryCreators      m_geometry_creators;
    LightCreators         m_light_creators;
    SurfaceCreators       m_surface_creators;
};


void PluginManager::Impl::registerCamera( 
        const std::string& name,
        CameraCreator creator )
{
    LEGION_ASSERT( creator != 0 ); 
    LEGION_ASSERT( m_camera_creators.find( name ) != m_camera_creators.end() );

    m_camera_creators.insert( std::make_pair( name, creator ) );
}


void PluginManager::Impl::registerFilm(
        const std::string& name,
        FilmCreator creator )
{
    LEGION_ASSERT( creator != 0 ); 
    LEGION_ASSERT( m_film_creators.find( name ) != m_film_creators.end() );

    m_film_creators.insert( std::make_pair( name, creator ) );
}


void PluginManager::Impl::registerGeometry(
        const std::string& name,
        GeometryCreator creator )
{
    LEGION_ASSERT( creator != 0 ); 
    LEGION_ASSERT( m_geometry_creators.count( name ) == 0 );

    m_geometry_creators.insert( std::make_pair( name, creator ) );
}


void PluginManager::Impl::registerLight(
        const std::string& name,
        LightCreator creator )
{
    LEGION_ASSERT( creator != 0 ); 
    LEGION_ASSERT( m_light_creators.find( name ) != m_light_creators.end() );

    m_light_creators.insert( std::make_pair( name, creator ) );
}


void PluginManager::Impl::registerSurface(
        const std::string& name,
        SurfaceCreator creator )
{
    LEGION_ASSERT( creator != 0 ); 
    LEGION_ASSERT( m_surface_creators.find( name ) != 
                   m_surface_creators.end() );

    m_surface_creators.insert( std::make_pair( name, creator ) );
}


ICamera* PluginManager::Impl::createCamera( 
        const std::string& name, 
        const Parameters& p )
{
    CameraCreators::iterator it = m_camera_creators.find( name );
    LEGION_ASSERT( it != m_camera_creators.end() ); 

    return it->second( m_context, p );
}


IFilm* PluginManager::Impl::createFilm( 
        const std::string& name, 
        const Parameters& p )
{
    FilmCreators::iterator it = m_film_creators.find( name );
    LEGION_ASSERT( it != m_film_creators.end() ); 

    return it->second( m_context, p );
}


IGeometry* PluginManager::Impl::createGeometry(
        const std::string& name,
        const Parameters& p )
{
    GeometryCreators::iterator it = m_geometry_creators.find( name );
    LEGION_ASSERT( it != m_geometry_creators.end() ); 

    return it->second( m_context, p );
}


ILight* PluginManager::Impl::createLight(
        const std::string& name,
        const Parameters& p )
{
    LightCreators::iterator it = m_light_creators.find( name );
    LEGION_ASSERT( it != m_light_creators.end() ); 

    return it->second( m_context, p );
}


ISurface* PluginManager::Impl::createSurface(
        const std::string& name,
        const Parameters& p )
{
    SurfaceCreators::iterator it = m_surface_creators.find( name );
    LEGION_ASSERT( it != m_surface_creators.end() ); 

    return it->second( m_context, p );
}


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


template <>
void PluginManager::registerPlugin<ICamera>( 
        const std::string& name,
        ICamera* (*create)( Context* ctx, const Parameters& params ) )
{
    m_impl->registerCamera( name, create );
}


template <>
void PluginManager::registerPlugin<IFilm>(
        const std::string& name,
        IFilm* (*create)( Context* ctx, const Parameters& params ) )
{
    m_impl->registerFilm( name, create );
}


template <>
void PluginManager::registerPlugin<IGeometry>(
        const std::string& name,
        IGeometry* (*create)( Context* ctx, const Parameters& params ) )
{
    m_impl->registerGeometry( name, create );
}


template <>
void PluginManager::registerPlugin<ILight>(
        const std::string& name,
        ILight* (*create)( Context* ctx, const Parameters& params ) )
{
    m_impl->registerLight( name, create );
}


template <>
void PluginManager::registerPlugin<ISurface>(
        const std::string& name,
        ISurface* (*create)( Context* ctx, const Parameters& params ) )
{
    m_impl->registerSurface( name, create );
}


template <>
ICamera* PluginManager::create<ICamera>(
        const std::string& plugin_name,
        const Parameters& params )
{
    return m_impl->createCamera( plugin_name, params );
}

template <>
IFilm* PluginManager::create<IFilm>(
        const std::string& plugin_name,
        const Parameters& params )
{
    return m_impl->createFilm( plugin_name, params );
}



template <>
IGeometry* PluginManager::create<IGeometry>(
        const std::string& plugin_name,
        const Parameters& params )
{
    return m_impl->createGeometry( plugin_name, params );
}



template <>
ILight* PluginManager::create<ILight>(
        const std::string& plugin_name,
        const Parameters& params )
{
    return m_impl->createLight( plugin_name, params );
}



template <>
ISurface* PluginManager::create<ISurface>(
        const std::string& plugin_name,
        const Parameters& params )
{
    return m_impl->createSurface( plugin_name, params );
}