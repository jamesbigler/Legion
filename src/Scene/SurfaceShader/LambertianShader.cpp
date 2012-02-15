

#include <Scene/SurfaceShader/LambertianShader.hpp>

using namespace legion;


LambertianShader::LambertianShader( const std::string& name )
  : ISurfaceShader( name )
{
}


LambertianShader::~LambertianShader()
{
}


void LambertianShader::sampleBSDF( const Vector2& seed, 
                                   const Vector3& w_out,
                                   const LocalGeometry& p,
                                   Vector3& w_in,
                                   float& pdf )
{
}


float LambertianShader::pdf( const Vector3& w_out, const LocalGeometry& p, const Vector3& w_in )
{
    return 0.0f;
}


Color LambertianShader::evaluateBSDF( const Vector3& w_out,
                                      const LocalGeometry& p,
                                      const Vector3& w_in )
{
    return Color( 0.0f, 0.0f, 0.0f );
}
    

void LambertianShader::setKd( const Color& kd )
{
    m_kd = kd;
}
