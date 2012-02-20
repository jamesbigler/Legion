
#include <Legion/Scene/SurfaceShader/ISurfaceShader.hpp>

using namespace legion;

ISurfaceShader::ISurfaceShader( Context* context, const std::string& name )
    : APIBase( context, name )
{
}


ISurfaceShader::~ISurfaceShader()
{
}
