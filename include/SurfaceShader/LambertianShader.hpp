
#ifndef LEGION_LAMBERTIAN_SHADER_H_
#define LEGION_LAMBERTIAN_SHADER_H_


#include <Interface/ISurfaceShader.hpp>


namespace legion
{


class LambertianShader : public ISurfaceShader
{
public:
    LambertianShader( const std::string& name );
    virtual ~LambertianShader();

    void  sample  ( const Vector2& seed, const Vector3& w_out, const Shader::Geometry& p, Vector3& w_in, float& pdf );
    float pdf     ( const Vector3& w_out, const Shader::Geometry& p, const Vector3& w_in );
    Color evaluate( const Vector3& w_out, const Shader::Geometry& p, const Vector3& w_in );

    void setKd( const Color& kd );
        
private:
    Color m_kd;
};


}

#endif // LEGION_LAMBERTIAN_SHADER_H_
