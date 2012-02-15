
#ifndef LEGION_LAMBERTIAN_SHADER_H_
#define LEGION_LAMBERTIAN_SHADER_H_


#include <Interface/ISurfaceShader.hpp>
#include <Core/Color.hpp>


namespace legion
{


class LocalGeometry;


class LambertianShader : public ISurfaceShader
{
public:
    LambertianShader( const std::string& name );
    virtual ~LambertianShader();

    virtual void   sampleBSDF( const Vector2& seed,
                               const Vector3& w_out,
                               const LocalGeometry& p,
                               Vector3& w_in,
                               float& pdf );


    virtual float   pdf( const Vector3& w_out,
                         const LocalGeometry& p,
                         const Vector3& w_in );

    virtual Color   evaluateBSDF( const Vector3& w_out,
                                  const LocalGeometry& p,
                                  const Vector3& w_in );

    void setKd( const Color& kd );
        
private:
    Color m_kd;
};


}

#endif // LEGION_LAMBERTIAN_SHADER_H_
