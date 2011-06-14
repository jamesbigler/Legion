

#include <Shader/LambertianShader.hpp>

using namespace legion;


LambertianShader::LambertianShader( const std::string& name )
  : IShader( name )
{
}


LambertianShader::~LambertianShader()
{
}


void  LambertianShader::sample( const Vector3& w_out, const Shader::Geometry& p, Vector3& w_in, float& pdf )
{
}


float LambertianShader::pdf( const Vector3& w_out, const Shader::Geometry& p, const Vector3& w_in )
{
}


Color LambertianShader::evaluate( const Vector3& w_out, const Shader::Geometry& p, const Vector3& w_in )
{
}
    

void LambertianShader::setKd( const Color& kd )
{
}
