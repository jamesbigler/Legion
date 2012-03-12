

#include <Legion/Scene/LightShader/PointLightShader.hpp>
#include <Legion/Core/Color.hpp>

using namespace legion;


PointLightShader::PointLightShader( Context* context, const std::string& name )
  : ILightShader( context, name ),
    m_rflux( 0.0f ),
    m_position( 0.0f )

{
}


PointLightShader::~PointLightShader()
{
}


void PointLightShader::sample( const Vector2& seed,
                               const LocalGeometry& p,
                               Vector3& on_light,
                               float& pdf )const
{
    on_light =  m_position;
    pdf = 1.0f;
}


float PointLightShader::pdf( const LocalGeometry& p, const Vector3& w_in )const
{
    return 0.0f;
}


float PointLightShader::getPower()const
{
    return 0.0f;
}


Color PointLightShader::getRadiance( const LocalGeometry& p, const Vector3& w_in )const
{
    return rflux;
}
    

void PointLightShader::setRadiantFlux( const Color& rflux )
{
    m_rflux = rflux;
}


void PointLightShader::setPosition( const Vector3& position )
{
    m_position = position;
}
