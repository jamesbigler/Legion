

#include <Legion/Scene/SurfaceShader/Lambertian.hpp>
#include <Legion/Renderer/Cuda/Shared.hpp>
#include <Legion/Common/Math/Math.hpp>
#include <Legion/Common/Math/ONB.hpp>
#include <Legion/Common/Util/TypeConversion.hpp>

using namespace legion;

// TODO: rename surfaceshader to material

Lambertian::Lambertian( Context* context, const std::string& name )
  : ISurfaceShader( context, name )
{
}


Lambertian::~Lambertian()
{
}


void Lambertian::setKd( const Color& kd )
{
    m_kd = kd;
}
    


void Lambertian::sampleBSDF( const Vector2& seed, 
                                   const Vector3& w_out,
                                   const LocalGeometry& p,
                                   Vector3& w_in,
                                   Color& f_over_pdf )const
{
    // sample hemisphere with cosine density by uniformly sampling
    // unit disk and projecting up to hemisphere
    Vector2 on_disk( squareToDisk( seed ) );
    const float x = on_disk.x();
    const float y = on_disk.y();
          float z = 1.0f - x*x -y*y;

    z = z > 0.0f ? sqrtf( z ) : 0.0f;

    // Transform into world space
    ONB onb( p.shading_normal );
    w_in = onb.inverseTransform( Vector3( x, y, z ) );

    // calculate pdf
    float pdf_inv  = 1.0f / z * ONE_DIV_PI;
    f_over_pdf = pdf_inv * ONE_DIV_PI * m_kd;
}


bool Lambertian::isSingular()const
{
    return false;
}


float Lambertian::pdf( const Vector3& w_out,
                             const LocalGeometry& p,
                             const Vector3& w_in )const
{
    float cosine = std::max( 0.0f, dot( w_in, p.shading_normal ) );
    return cosine * ONE_DIV_PI;
}


Color Lambertian::evaluateBSDF( const Vector3& w_out,
                                      const LocalGeometry& p,
                                      const Vector3& w_in )const
{
    
    float cosine = std::max( 0.0f, dot( w_in, p.shading_normal ) );
    return cosine * ONE_DIV_PI * m_kd;
}
    
