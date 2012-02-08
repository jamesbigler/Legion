

#include <LightShader/PointLightShader.hpp>
#include <Core/Color.hpp>

using namespace legion;


PointLightShader::PointLightShader( const std::string& name )
  : ILightShader( name )
{
}


PointLightShader::~PointLightShader()
{
}


void PointLightShader::sample( const LocalGeometry& p, Vector3& w_in, float& pdf )
{
}


float PointLightShader::pdf( const LocalGeometry& p, const Vector3& w_in )
{
    return 0.0f;
}


float PointLightShader::getPower()const
{
    return 0.0f;
}


Color PointLightShader::getRadiance( const LocalGeometry& p, const Vector3& w_in )
{
    return Color( 0.0f, 0.0f, 0.0f );
}
    

void PointLightShader::setRadiantFlux( const Color& kd )
{
}


void PointLightShader::setPosition( const Vector3& kd )
{
}
