
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
#include <Legion/Objects/Geometry/Sphere.hpp>
#include <Legion/Objects/Renderer/IRenderer.hpp>
#include <Legion/Renderer/OptiXScene.hpp>
#include <fstream>


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

    void setRenderer   ( IRenderer* renderer );

    void setCamera     ( ICamera* camera );

    void setEnvironment( IEnvironment* environment );

    void addGeometry   ( IGeometry* geometry );

    void addLight( ILight* light );

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
