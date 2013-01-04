
#include <Legion/Objects/Surface/ISurface.hpp>

using namespace legion;

ISurface::ISurface( Context* context, const std::string& name )
    : APIBase( context, name )
{
}


ISurface::~ISurface()
{
}
