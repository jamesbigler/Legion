

#include <Core/Context.hpp>

using namespace legion;


Context::Context( const std::string& name )
  : APIBase( name )
{
}


Context::~Context()
{
}


void Context::addMesh( const Mesh* mesh )
{
}


void Context::setActiveCamera( const ICamera* camera )
{
}


void Context::setActiveFilm( const IFilm* camera )
{
}


void Context::render()
{
}
