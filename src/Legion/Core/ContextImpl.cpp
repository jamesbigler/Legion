
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Core/ContextImpl.hpp>
#include <Legion/Scene/Camera/ICamera.hpp>
#include <Legion/Scene/Film/IFilm.hpp>
#include <Legion/Scene/LightShader/ILightShader.hpp>
#include <Legion/Scene/Mesh/Mesh.hpp>


using namespace legion;

Context::Impl::Impl()
{
    LLOG_INFO << "Creating Context::Impl";
}


Context::Impl::~Impl()
{
    LLOG_INFO << "Destroying Context::Impl";
}


void Context::Impl::addMesh( const Mesh* mesh )
{
    // TODO: add NULL check to all of these
    LLOG_INFO << "Adding mesh <" << mesh->getName() << ">";
    m_meshes.push_back( mesh );
}


void Context::Impl::addLight( const ILightShader* light_shader )
{
    Light light;
    light.shader   = light_shader;
    light.geometry = 0u;
    LLOG_INFO << "Adding light <" << light.getName() << ">";
    m_lights.push_back( light );
}


void Context::Impl::addLight( const ILightShader* light_shader, const Mesh* light_geometry )
{
    Light light;
    light.shader   = light_shader;
    light.geometry = light_geometry;
    LLOG_INFO << "Adding light <" << light.getName() << ">";
    m_lights.push_back( light );
}


void Context::Impl::setActiveCamera( const ICamera* camera )
{
    LLOG_INFO << "Adding camera <" << camera->getName() << ">";
    m_camera = camera;
}


void Context::Impl::setActiveFilm( const IFilm* film )
{
    LLOG_INFO << "Adding film <" << film->getName() << ">";
    m_film = film;
}


void Context::Impl::preprocess()
{
}


void Context::Impl::doRender()
{
}


void Context::Impl::postprocess()
{
}

void Context::Impl::render()
{
    LLOG_INFO << "rendering ....";

    preprocess();

    doRender();

    postprocess();



    const Index2  image_dims  = m_film->getDimensions();
}



