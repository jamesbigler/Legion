
#ifndef LEGION_LAMBERTIAN_SHADER_H_
#define LEGION_LAMBERTIAN_SHADER_H_


#include <Interface/IShader.hpp>


namespace legion
{


class LambertianShader : public IShader
{
public:
    LambertianShader( const std::string& name );
    virtual ~LambertianShader();

    void  sample  ( const Vector3& w_out, const Shader::Geometry& p, Vector3& w_in, float& pdf );
    float pdf     ( const Vector3& w_out, const Shader::Geometry& p, const Vector3& w_in );
    Color evaluate( const Vector3& w_out, const Shader::Geometry& p, const Vector3& w_in );

    void setKd( const Color& kd );
        
private:
    class Impl;
    std::tr1::shared_ptr<Impl> m_impl;
};


}

#endif // LEGION_LAMBERTIAN_SHADER_H_
