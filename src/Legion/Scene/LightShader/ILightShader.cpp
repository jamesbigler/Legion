
#include <Legion/Scene/LightShader/ILightShader.hpp>

using namespace legion;


ILightShader::ILightShader( Context* context, const std::string& name )
    : APIBase( context, name )
{
}


ILightShader::~ILightShader()
{
}
