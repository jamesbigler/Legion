
#ifndef LEGION_OBJECTS_SURFACE_LAMBERTIANSHADER_H_
#define LEGION_OBJECTS_SURFACE_LAMBERTIANSHADER_H_


#include <Legion/Objects/Surface/ISurface.hpp>
#include <Legion/Core/Color.hpp>


namespace legion
{


class VariableContainer;


class Lambertian : public ISurface
{
public:
    Lambertian( Context* context );
    ~Lambertian();
    
    void setReflectance( const Color& reflectance );
    
    const char* name()const;
    const char* sampleBSDFFunctionName()const;
    const char* evaluateBSDFFunctionName()const;
    const char* pdfFunctionName()const;

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
    Color m_reflectance;
};


}

#endif // LEGION_OBJECTS_SURFACE_LAMBERTIANSHADER_H_
