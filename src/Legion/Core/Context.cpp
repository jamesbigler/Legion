

#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Scene/Camera/ICamera.hpp>
#include <Legion/Scene/Film/IFilm.hpp>
#include <Legion/Scene/SurfaceShader/ISurfaceShader.hpp>
#include <Legion/Scene/LightShader/ILightShader.hpp>
#include <Legion/Core/Context.hpp>
#include <Legion/Core/config.hpp>


using namespace legion;


/******************************************************************************\
 *                                                                            *
 *                                                                            *
\******************************************************************************/

Context::Context( const std::string& name ) 
    : APIBase( this, name )
{
}


Context::~Context()
{
}


optix::Buffer Context::createOptixBuffer()
{
    return m_renderer.createBuffer();
}


void Context::addMesh( Mesh* mesh )
{
    // TODO: add NULL check to all of these
    LLOG_INFO << "Adding mesh <" << mesh->getName() << ">";
    m_meshes.push_back( mesh );

    m_renderer.addMesh( mesh );
}


void Context::addLight( const ILightShader* light_shader )
{
    Light light;
    light.shader   = light_shader;
    light.geometry = 0u;
    LLOG_INFO << "Adding light <" << light.getName() << ">";
    m_lights.push_back( light );
}


void Context::addLight( const ILightShader* light_shader,
                        const Mesh* light_geometry )
{

    Light light;
    light.shader   = light_shader;
    light.geometry = light_geometry;
    LLOG_INFO << "Adding light <" << light.getName() << ">";
    m_lights.push_back( light );
}


void Context::setActiveCamera( ICamera* camera )
{
  
    LLOG_INFO << "Adding camera <" << camera->getName() << ">";
    m_renderer.setCamera( camera );
}


void Context::setActiveFilm( IFilm* film )
{
    LLOG_INFO << "Adding film <" << film->getName() << ">";
    m_renderer.setFilm( film );
}


void Context::render()
{

    LLOG_INFO << "rendering ....";
    m_renderer.render();
}

