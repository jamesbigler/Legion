
// Copyright (C) 2011 R. Keith Morley 
// 
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


#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Common/Util/Parameters.hpp>
#include <Legion/Common/Util/Plugin.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>
#include <Legion/Common/Util/Timer.hpp>
#include <Legion/Core/Context.hpp>
#include <Legion/Core/Exception.hpp>
#include <Legion/Core/PluginContext.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Objects/Renderer/IRenderer.hpp>
#include <Legion/Renderer/OptiXScene.hpp>
#include <fstream>


// TODO: For object registration.  Better way to do this???  Perhaps a static
//       object which registers plugins?

#include <Legion/Objects/Camera/ThinLens.hpp>
#include <Legion/Objects/Display/ImageFileDisplay.hpp>
#include <Legion/Objects/Environment/ConstantEnvironment.hpp>
#include <Legion/Objects/Geometry/Parallelogram.hpp>
#include <Legion/Objects/Geometry/Sphere.hpp>
#include <Legion/Objects/Geometry/TriMesh.hpp>
#include <Legion/Objects/Renderer/ProgressiveRenderer.hpp>
#include <Legion/Objects/Surface/DiffuseEmitter.hpp>
#include <Legion/Objects/Surface/Lambertian.hpp>
#include <Legion/Objects/Surface/Ward.hpp>
#include <Legion/Objects/Texture/ConstantTexture.hpp>
#include <Legion/Objects/Texture/ImageTexture.hpp>
#include <Legion/Objects/Texture/PerlinTexture.hpp>



using namespace legion;

#define CHECK_NULL( f, ptr )  \
    if( !ptr ) throw Exception( std::string( f ) + ": " #ptr " param is NULL" );


//------------------------------------------------------------------------------
//
//  Context::Impl class 
//
//------------------------------------------------------------------------------

class Context::Impl
{
public:
    Impl( Context* context );
    ~Impl();
    
    ICamera*      createCamera     ( const char* name, const Parameters& p );
    IDisplay*     createDisplay    ( const char* name, const Parameters& p );
    IEnvironment* createEnvironment( const char* name, const Parameters& p );
    IGeometry*    createGeometry   ( const char* name, const Parameters& p );
    ILight*       createLight      ( const char* name, const Parameters& p );
    IRenderer*    createRenderer   ( const char* name, const Parameters& p );
    ISurface*     createSurface    ( const char* name, const Parameters& p );
    ITexture*     createTexture    ( const char* name, const Parameters& p );

    void setRenderer   ( IRenderer* renderer );
    void setCamera     ( ICamera* camera );
    void setEnvironment( IEnvironment* environment );
    void addGeometry   ( IGeometry* geometry );
    void addLight      ( ILight* light );

    void addAssetPath( const std::string& path );

    void render();

    PluginContext& getPluginContext();

private:
    PluginManager   m_plugin_mgr;
    OptiXScene      m_optix_scene;
    PluginContext   m_plugin_context;
    std::ofstream   m_log_file;

    IRenderer*              m_renderer;
    ICamera*                m_camera;
    std::vector<IGeometry*> m_geometry;
};



Context::Impl::Impl( Context* context ) 
    : m_plugin_mgr( context ),
      m_optix_scene(),
      m_plugin_context( m_optix_scene.getOptiXContext() ),
      m_log_file( "legion.log" )
{
    Log::setStream( m_log_file );
    PluginManager& pm = m_plugin_mgr;
    pm.registerPlugin( "ThinLens",            &ThinLens::create            );
    pm.registerPlugin( "ImageFileDisplay",    &ImageFileDisplay::create    );
    pm.registerPlugin( "ConstantEnvironment", &ConstantEnvironment::create );
    pm.registerPlugin( "Sphere",              &Sphere::create              );
    pm.registerPlugin( "TriMesh",             &TriMesh::create             );
    pm.registerPlugin( "Parallelogram"      , &Parallelogram::create       );
    pm.registerPlugin( "ProgressiveRenderer", &ProgressiveRenderer::create );
    pm.registerPlugin( "DiffuseEmitter",      &DiffuseEmitter::create      );
    pm.registerPlugin( "Lambertian",          &Lambertian::create          );
    pm.registerPlugin( "Ward",                &Ward::create                );
    pm.registerPlugin( "ConstantTexture",     &ConstantTexture::create     );
    pm.registerPlugin( "ImageTexture",        &ImageTexture::create        );
    pm.registerPlugin( "PerlinTexture",       &PerlinTexture::create       );

  /*
    m_plugin_mgr.registerPlugin<IGeometry>( "Sphere", &Sphere::create );

    Parameters params;
    IGeometry* geo = m_plugin_mgr.create<IGeometry>( "Sphere", params );
    LLOG_INFO << "\tSphere: " << geo;
    //Geometry* geo = ctx->create<Geometry>( "Sphere", properties );
  */
}


Context::Impl::~Impl() 
{
}


ICamera* Context::Impl::createCamera( const char* name, const Parameters& p )
{
    return m_plugin_mgr.create<ICamera>( name, p );
}


IDisplay* Context::Impl::createDisplay( const char* name, const Parameters& p )
{
    return m_plugin_mgr.create<IDisplay>( name, p );
}


IEnvironment* Context::Impl::createEnvironment( const char* name, const Parameters& p )
{
    return m_plugin_mgr.create<IEnvironment>( name, p );
}


IGeometry* Context::Impl::createGeometry( const char* name, const Parameters& p )
{
    return m_plugin_mgr.create<IGeometry>( name, p );
}


ILight* Context::Impl::createLight( const char* name, const Parameters& p )
{
    return m_plugin_mgr.create<ILight>( name, p );
}


IRenderer* Context::Impl::createRenderer( const char* name, const Parameters& p )
{
    return m_plugin_mgr.create<IRenderer>( name, p );
}


ISurface* Context::Impl::createSurface( const char* name, const Parameters& p )
{
    return m_plugin_mgr.create<ISurface>( name, p );
}


ITexture* Context::Impl::createTexture( const char* name, const Parameters& p )
{
    return m_plugin_mgr.create<ITexture>( name, p );
}


void Context::Impl::setRenderer( IRenderer* renderer )
{
    m_renderer = renderer;
    m_optix_scene.setRenderer( renderer );
}


void Context::Impl::setCamera( ICamera* camera )
{
    m_camera = camera;
    m_optix_scene.setCamera( camera );
}


void Context::Impl::setEnvironment( IEnvironment* environment )
{
    m_optix_scene.setEnvironment( environment );
}


void Context::Impl::addGeometry( IGeometry* geometry )
{
    m_optix_scene.addGeometry( geometry );
}


void Context::Impl::addLight( ILight* light )
{
    m_optix_scene.addLight( light );
}


void Context::Impl::addAssetPath( const std::string& path )
{
    m_optix_scene.addAssetPath( path );
}


void Context::Impl::render()
{
    m_optix_scene.sync();

    LLOG_INFO << "Rendering frame ... ";
    Timer timer;
    timer.start();
    optix::Program ray_gen_program = 
      m_optix_scene.getOptiXContext()->getRayGenerationProgram( 0 );
    VariableContainer vc( ray_gen_program.get() );
    //VariableContainer vc( m_optix_scene.getOptiXContext().get() );
    m_renderer->render( vc );
    timer.stop();
    LLOG_INFO << "Frame complete ... (" << timer.getTimeElapsed() << "s)";
}


PluginContext& Context::Impl::getPluginContext()
{
    return m_plugin_context; 
}

//------------------------------------------------------------------------------
//
//  Context class -- public API simply forwards to Context::Impl
//
//------------------------------------------------------------------------------

Context::Context() 
    : m_impl( new Impl( this ) )
{
}


Context::~Context()
{
}

void Context::setRenderer( IRenderer* renderer )
{
    m_impl->setRenderer( renderer );
}


void Context::setCamera( ICamera* camera )
{
    m_impl->setCamera( camera );
}


void Context::setEnvironment( IEnvironment* environment )
{
    m_impl->setEnvironment( environment );
}


void Context::addGeometry( IGeometry* geometry )
{
    m_impl->addGeometry( geometry );
}


void Context::addLight( ILight* light )
{
    m_impl->addLight( light );
}


void Context::addAssetPath( const std::string& path )
{
    m_impl->addAssetPath( path );
}


void Context::render()
{
    return m_impl->render();
}


PluginContext& Context::getPluginContext()
{
    return m_impl->getPluginContext();
}


ICamera* Context::createCamera( const char* name, const Parameters& p )
{
    return m_impl->createCamera( name, p );
}


IDisplay* Context::createDisplay( const char* name, const Parameters& p )
{
    return m_impl->createDisplay( name, p );
}


IEnvironment* Context::createEnvironment( const char* name, const Parameters& p)
{
    return m_impl->createEnvironment( name, p );
}


IGeometry* Context::createGeometry( const char* name, const Parameters& p )
{
    return m_impl->createGeometry( name, p );
}


ILight* Context::createLight( const char* name, const Parameters& p )
{
    return m_impl->createLight( name, p );
}


IRenderer* Context::createRenderer ( const char* name, const Parameters& p )
{
    return m_impl->createRenderer( name, p );
}


ISurface* Context::createSurface( const char* name, const Parameters& p )
{
    return m_impl->createSurface( name, p );
}


ITexture* Context::createTexture( const char* name, const Parameters& p )
{
    return m_impl->createTexture( name, p );
}
