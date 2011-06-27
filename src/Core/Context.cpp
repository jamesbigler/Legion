

#include <Core/Context.hpp>
#include <Core/Mesh.hpp>
#include <Interface/ILightShader.hpp>
#include <Interface/ICamera.hpp>
#include <Interface/IFilm.hpp>
#include <vector>
#include <iostream>

using namespace legion;

/******************************************************************************\
 *                                                                            *
 *                                                                            *
\******************************************************************************/

namespace legion
{

class Context::Impl
{
public:
    Impl();
    ~Impl();

    void addMesh ( const Mesh* mesh );
    void addLight( const ILightShader* light_shader );
    void addLight( const ILightShader* light_shader, const Mesh* light_geometry );

    void setActiveCamera( const ICamera* camera );
    void setActiveFilm( const IFilm* film );

    void render();

private:
    struct Light
    {
        // TODO: Flesh this out, move to shared place if needed
        std::string    getName() { return shader->getName() + ":" + 
                                          ( geometry ? geometry->getName() : "NULL" ); }
        const ILightShader* shader;
        const Mesh*         geometry;
    };

    std::vector<const Mesh*> m_meshes;
    std::vector<Light> m_lights;
    const ICamera*           m_camera;
    const IFilm*             m_film;

};

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


void Context::Impl::render()
{
    std::cerr << "rendering ...." << std::endl;
}



}

/******************************************************************************\
 *                                                                            *
 *                                                                            *
\******************************************************************************/
Context::Context( const std::string& name )
  : APIBase( name ), m_impl( new Impl() )
{
}


Context::~Context()
{
}


void Context::addMesh( const Mesh* mesh )
{
    m_impl->addMesh( mesh );
}


void Context::addLight( const ILightShader* light_shader )
{
    m_impl->addLight( light_shader );
}


void Context::addLight( const ILightShader* light_shader, const Mesh* light_geometry )
{
    m_impl->addLight( light_shader, light_geometry );
}


void Context::setActiveCamera( const ICamera* camera )
{
    m_impl->setActiveCamera( camera );
}


void Context::setActiveFilm( const IFilm* film )
{
    m_impl->setActiveFilm( film );
}


void Context::render()
{
    m_impl->render();
}

