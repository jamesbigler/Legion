
#ifndef LEGION_OBJECTS_SURFACE_WARD_HPP_
#define LEGION_OBJECTS_SURFACE_WARD_HPP_


#include <Legion/Objects/Surface/ISurface.hpp>
#include <Legion/Core/Color.hpp>


namespace legion
{

class VariableContainer;

class Ward : public ISurface
{
public:
    static ISurface* create( Context* context, const Parameters& params );

    Ward( Context* context );
    ~Ward();
    
    void setDiffuseReflectance( const Color& reflectance );
    void setSpecularReflectance( const Color& reflectance );
    void setAlphaU( float alpha_u );
    void setAlphaV( float alpha_v );
    
    const char* name()const;
    const char* sampleBSDFFunctionName()const;
    const char* evaluateBSDFFunctionName()const;
    const char* pdfFunctionName()const;
    const char* emissionFunctionName()const;

    void setVariables( VariableContainer& container ) const ;

private:
    Color m_diffuse_reflectance;
    Color m_specular_reflectance;
    float m_alpha_u;
    float m_alpha_v;

};

}

#endif // LEGION_OBJECTS_SURFACE_WARD_HPP_
