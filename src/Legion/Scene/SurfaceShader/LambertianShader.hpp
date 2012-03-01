
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
    ~LambertianShader();
    
    
    void setKd( const Color& kd );


    void   sampleBSDF( const Vector2& seed,
                       const Vector3& w_out,
                       const LocalGeometry& p,
                       Vector3& w_in,
                       float& pdf )const;


    float   pdf( const Vector3& w_out,
                 const LocalGeometry& p,
                 const Vector3& w_in )const;


    Color   evaluateBSDF( const Vector3& w_out,
                          const LocalGeometry& p,
                          const Vector3& w_in )const;
    

    bool    emits()const;


    Color   emission( const Vector3& w_out, const LocalGeometry& p )const;
        
private:
    Color m_kd;
};


}

#endif // LEGION_SCENE_SURFACE_SHADER_LAMBERTIANSHADER_H_
