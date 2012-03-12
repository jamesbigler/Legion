

#include <Legion/Scene/LightShader/PointLightShader.hpp>
#include <Legion/Core/Color.hpp>
#include <Legion/Common/Math/Math.hpp>

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


Color PointLightShader::getPower()const
{
    return 4.0f * legion::PI * m_rflux;
}


Color PointLightShader::getRadiance( const LocalGeometry& p,
                                     const Vector3& w_in )const
{
    return m_rflux;
}
    

void PointLightShader::setRadiantFlux( const Color& rflux )
{
    m_rflux = rflux;
}


void PointLightShader::setPosition( const Vector3& position )
{
    m_position = position;
}
