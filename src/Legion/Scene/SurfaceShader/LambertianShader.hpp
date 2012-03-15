
#ifndef LEGION_SCENE_SURFACE_SHADER_LAMBERTIANSHADER_H_
#define LEGION_SCENE_SURFACE_SHADER_LAMBERTIANSHADER_H_


#include <Legion/Scene/SurfaceShader/ISurfaceShader.hpp>
#include <Legion/Core/Color.hpp>


namespace legion
{


struct LocalGeometry;


class LambertianShader : public ISurfaceShader
{
public:
    LambertianShader( Context* context, const std::string& name );
    ~LambertianShader();
    
    
    void setKd( const Color& kd );


    void   sampleBSDF( const Vector2& seed,
                       const Vector3& w_out,
                       const LocalGeometry& p,
                       Vector3& w_in,
                       Color& f_over_pdf )const;

    bool isSingular()const;


    float   pdf( const Vector3& w_out,
                 const LocalGeometry& p,
                 const Vector3& w_in )const;


    Color   evaluateBSDF( const Vector3& w_out,
                          const LocalGeometry& p,
                          const Vector3& w_in )const;

private:
    Color m_kd;
};


}

#endif // LEGION_SCENE_SURFACE_SHADER_LAMBERTIANSHADER_H_
