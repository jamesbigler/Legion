

#include <Legion/Core/Context.hpp>
#include <Legion/Core/ContextImpl.hpp>


using namespace legion;


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

