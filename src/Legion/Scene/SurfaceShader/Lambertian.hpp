
#ifndef LEGION_SCENE_SURFACE_SHADER_LAMBERTIANSHADER_H_
#define LEGION_SCENE_SURFACE_SHADER_LAMBERTIANSHADER_H_


#include <Legion/Scene/SurfaceShader/ISurfaceShader.hpp>
#include <Legion/Core/Color.hpp>


namespace legion
{


class VariableContainer;


class Lambertian : public ISurfaceShader
{
public:
    Lambertian();
    ~Lambertian();
    
    
    void setKd( const Color& kd );
    
    void setVariables( VariableContainer& container ) const ;


    /*
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

    */
private:
    Color m_kd;
};


}

#endif // LEGION_SCENE_SURFACE_SHADER_LAMBERTIANSHADER_H_
