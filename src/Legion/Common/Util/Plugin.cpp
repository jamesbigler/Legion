
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
    typedef ICamera*       (*CameraCreator       )( const Parameters& );
    typedef IFilm*         (*FilmCreator         )( const Parameters& );
    typedef IGeometry*     (*GeometryCreator     )( const Parameters& );
    typedef ILight*        (*LightCreator        )( const Parameters& );
    typedef ISurfaceShader*(*SurfaceShaderCreator)( const Parameters& );

    Impl() {}
    ~Impl() {}

    void registerCamera       ( const std::string& name, CameraCreator        );
    void registerFilm         ( const std::string& name, FilmCreator          );
    void registerGeometry     ( const std::string& name, GeometryCreator      );
    void registerLight        ( const std::string& name, LightCreator         );
    void registerSurfaceShader( const std::string& name, SurfaceShaderCreator );

    ICamera*  createCamera    ( const std::string& name, const Parameters& p );
    IFilm* createFilm         ( const std::string& name, const Parameters& p );
    IGeometry* createGeometry ( const std::string& name, const Parameters& p );
    ILight* createLight       ( const std::string& name, const Parameters& p );
    ISurfaceShader* createSurfaceShader( const std::string& name,
                                         const Parameters& p );

private:
    typedef std::map<std::string, CameraCreator>          CameraCreators;
    typedef std::map<std::string, FilmCreator>            FilmCreators;    
    typedef std::map<std::string, GeometryCreator>        GeometryCreators;
    typedef std::map<std::string, LightCreator>           LightCreators;
    typedef std::map<std::string, SurfaceShaderCreator>   SurfaceShaderCreators;

    CameraCreators        m_camera_creators;
    FilmCreators          m_film_creators;
    GeometryCreators      m_geometry_creators;
    LightCreators         m_light_creators;
    SurfaceShaderCreators m_surface_shader_creators;
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


void PluginManager::Impl::registerSurfaceShader(
        const std::string& name,
        SurfaceShaderCreator creator )
{
    LEGION_ASSERT( creator != 0 ); 
    LEGION_ASSERT( m_surface_shader_creators.find( name ) != 
                   m_surface_shader_creators.end() );

    m_surface_shader_creators.insert( std::make_pair( name, creator ) );
}


ICamera* PluginManager::Impl::createCamera( 
        const std::string& name, 
        const Parameters& p )
{
    CameraCreators::iterator it = m_camera_creators.find( name );
    LEGION_ASSERT( it != m_camera_creators.end() ); 

    return it->second( p );
}


IFilm* PluginManager::Impl::createFilm( 
        const std::string& name, 
        const Parameters& p )
{
    FilmCreators::iterator it = m_film_creators.find( name );
    LEGION_ASSERT( it != m_film_creators.end() ); 

    return it->second( p );
}


IGeometry* PluginManager::Impl::createGeometry(
        const std::string& name,
        const Parameters& p )
{
    GeometryCreators::iterator it = m_geometry_creators.find( name );
    LEGION_ASSERT( it != m_geometry_creators.end() ); 

    return it->second( p );
}


ILight* PluginManager::Impl::createLight(
        const std::string& name,
        const Parameters& p )
{
    LightCreators::iterator it = m_light_creators.find( name );
    LEGION_ASSERT( it != m_light_creators.end() ); 

    return it->second( p );
}


ISurfaceShader* PluginManager::Impl::createSurfaceShader(
        const std::string& name,
        const Parameters& p )
{
    SurfaceShaderCreators::iterator it = m_surface_shader_creators.find( name );
    LEGION_ASSERT( it != m_surface_shader_creators.end() ); 

    return it->second( p );
}


//-----------------------------------------------------------------------------
//
// PluginManager 
//
//-----------------------------------------------------------------------------
PluginManager::PluginManager()
    : m_impl( new Impl )
{
}


PluginManager::~PluginManager()
{
}


template <>
void PluginManager::registerPlugin<ICamera>( 
        const std::string& name,
        ICamera* (*create)( const Parameters& params ) )
{
    m_impl->registerCamera( name, create );
}


template <>
void PluginManager::registerPlugin<IFilm>(
        const std::string& name,
        IFilm* (*create)( const Parameters& params ) )
{
    m_impl->registerFilm( name, create );
}


template <>
void PluginManager::registerPlugin<IGeometry>(
        const std::string& name,
        IGeometry* (*create)( const Parameters& params ) )
{
    m_impl->registerGeometry( name, create );
}


template <>
void PluginManager::registerPlugin<ILight>(
        const std::string& name,
        ILight* (*create)( const Parameters& params ) )
{
    m_impl->registerLight( name, create );
}


template <>
void PluginManager::registerPlugin<ISurfaceShader>(
        const std::string& name,
        ISurfaceShader* (*create)( const Parameters& params ) )
{
    m_impl->registerSurfaceShader( name, create );
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
ISurfaceShader* PluginManager::create<ISurfaceShader>(
        const std::string& plugin_name,
        const Parameters& params )
{
    return m_impl->createSurfaceShader( plugin_name, params );
}
