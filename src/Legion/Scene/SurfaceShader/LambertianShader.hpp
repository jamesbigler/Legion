
#ifndef LEGION_SCENE_SURFACE_SHADER_LAMBERTIANSHADER_H_
#define LEGION_SCENE_SURFACE_SHADER_LAMBERTIANSHADER_H_


#include <Legion/Scene/SurfaceShader/ISurfaceShader.hpp>
#include <Legion/Core/Color.hpp>


namespace legion
{


class LocalGeometry;


class LambertianShader : public ISurfaceShader
{
public:
    LambertianShader( Context* context, const std::string& name );
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

#endif // LEGION_SCENE_SURFACE_SHADER_LAMBERTIANSHADER_H_