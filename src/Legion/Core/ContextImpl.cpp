
#include <Legion/Core/ContextImpl.hpp>
#include <Legion/Scene/Camera/ICamera.hpp>
#include <Legion/Scene/Film/IFilm.hpp>
#include <Legion/Scene/LightShader/ILightShader.hpp>
#include <Legion/Scene/Mesh/Mesh.hpp>

#include <iostream>

using namespace legion;

Context::Impl::Impl()
{
    std::cerr << "Creating Context::Impl" << std::endl;
}


Context::Impl::~Impl()
{
    std::cerr << "Destroying Context::Impl" << std::endl;
}


void Context::Impl::addMesh( const Mesh* mesh )
{
    // TODO: add NULL check to all of these
    std::cerr << "Adding mesh <" << mesh->getName() << ">" << std::endl;
    m_meshes.push_back( mesh );
}


void Context::Impl::addLight( const ILightShader* light_shader )
{
    Light light;
    light.shader   = light_shader;
    light.geometry = 0u;
    std::cerr << "Adding light <" << light.getName() << ">" << std::endl;
    m_lights.push_back( light );
}


void Context::Impl::addLight( const ILightShader* light_shader, const Mesh* light_geometry )
{
    Light light;
    light.shader   = light_shader;
    light.geometry = light_geometry;
    std::cerr << "Adding light <" << light.getName() << ">" << std::endl;
    m_lights.push_back( light );
}


void Context::Impl::setActiveCamera( const ICamera* camera )
{
    std::cerr << "Adding camera <" << camera->getName() << ">" << std::endl;
    m_camera = camera;
}


void Context::Impl::setActiveFilm( const IFilm* film )
{
    std::cerr << "Adding film <" << film->getName() << ">" << std::endl;
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
    std::cerr << "rendering ...." << std::endl;

    preprocess();

    doRender();

    postprocess();



    const Index2  image_dims  = m_film->getDimensions();
}



