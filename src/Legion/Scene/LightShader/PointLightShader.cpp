

#include <Legion/Scene/LightShader/PointLightShader.hpp>
#include <Legion/Core/Color.hpp>
#include <Legion/Common/Math/Math.hpp>

using namespace legion;


PointLightShader::PointLightShader( Context* context, const std::string& name )
  : ILightShader( context, name ),
    m_intensity( 0.0f ),
    m_position( 0.0f )

{
}


PointLightShader::~PointLightShader()
{
}


void PointLightShader::sample( const Vector2& seed,
                               Vector3& on_light,
                               float& pdf )const
{
    on_light =  m_position;
    pdf = 1.0f;
}


bool PointLightShader::isSingular()const
{
    return true;
}


Color PointLightShader::power()const
{
    return 4.0f * legion::PI * m_intensity;
}


Color PointLightShader::emittance( const LocalGeometry& light_geom,
                                   const Vector3& w_in )const
{
    return m_intensity;
}
    

void PointLightShader::setIntensity( const Color& intensity )
{
    m_intensity = intensity;
}


void PointLightShader::setPosition( const Vector3& position )
{
    m_position = position;
}
