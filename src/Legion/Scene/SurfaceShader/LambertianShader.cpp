

#include <Legion/Scene/SurfaceShader/LambertianShader.hpp>
#include <Legion/Renderer/Cuda/Shared.hpp>

using namespace legion;


LambertianShader::LambertianShader( Context* context, const std::string& name )
  : ISurfaceShader( context, name )
{
}


LambertianShader::~LambertianShader()
{
}


void LambertianShader::setKd( const Color& kd )
{
    m_kd = kd;
}
    


void LambertianShader::sampleBSDF( const Vector2& seed, 
                                   const Vector3& w_out,
                                   const LocalGeometry& p,
                                   Vector3& w_in,
                                   float& pdf )const
{
}


float LambertianShader::pdf( const Vector3& w_out,
                             const LocalGeometry& p,
                             const Vector3& w_in )const
{
    return 0.0f;
}


Color LambertianShader::evaluateBSDF( const Vector3& w_out,
                                      const LocalGeometry& p,
                                      const Vector3& w_in )const
{
    return Color( 0.0f, 0.0f, 0.0f );
}
    

bool LambertianShader::emits()const
{
    return true;
}


Color LambertianShader::emission( const Vector3& w_out,
                                  const LocalGeometry& p )const
{
    Vector3 normal = Vector3( p.shading_normal.x,
                              p.shading_normal.y,
                              p.shading_normal.z );
    Vector3 in_gammut = ( normal + Vector3( 1.0f ) ) * 0.5f;
    return Color( in_gammut.x(), in_gammut.y(), in_gammut.z() );
}
    
        
