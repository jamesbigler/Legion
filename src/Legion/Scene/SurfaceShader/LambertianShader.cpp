

#include <Legion/Scene/SurfaceShader/LambertianShader.hpp>
#include <Legion/Renderer/Cuda/Shared.hpp>
#include <Legion/Common/Math/Math.hpp>
#include <Legion/Common/Math/ONB.hpp>
#include <Legion/Common/Util/TypeConversion.hpp>

using namespace legion;

// TODO: rename surfaceshader to material
// TODO: rename LambertianShader to Lambertian

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
                                   Color& f_over_pdf )const
{
    // sample hemisphere with cosine density by uniformly sampling
    // unit disk and projecting up to hemisphere
    // TODO: use concentric disc map and project up
    float phi = TWO_PI * seed.x();
    float r   = sqrtf( seed.y() );
    float x   = r * cosf( phi );
    float y   = r * sinf( phi );
    float z   = 1.0f - x*x -y*y;
    z         = z > 0.0f ? sqrtf( z ) : 0.0f;

    ONB onb( toVector3( p.shading_normal ) );
    w_in = onb.inverseTransform( Vector3( x, y, z ) );

    // calculate pdf
    float pdf_inv  = 1.0f / z * ONE_DIV_PI;
    f_over_pdf = pdf_inv * ONE_DIV_PI * m_kd;
}


bool LambertianShader::isSingular()const
{
    return false;
}


float LambertianShader::pdf( const Vector3& w_out,
                             const LocalGeometry& p,
                             const Vector3& w_in )const
{
    float cosine = std::max( 0.0f, dot( w_in, toVector3( p.shading_normal ) ) );
    return cosine * ONE_DIV_PI;
}


Color LambertianShader::evaluateBSDF( const Vector3& w_out,
                                      const LocalGeometry& p,
                                      const Vector3& w_in )const
{
    
    float cosine = std::max( 0.0f, dot( w_in, toVector3( p.shading_normal ) ) );
    return cosine * ONE_DIV_PI * m_kd;
}
    
