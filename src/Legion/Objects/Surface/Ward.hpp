
#ifndef LEGION_OBJECTS_SURFACE_WARD_HPP_
#define LEGION_OBJECTS_SURFACE_WARD_HPP_


#include <Legion/Objects/Surface/ISurface.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>
#include <Legion/Core/Color.hpp>


namespace legion
{

class VariableContainer;

class LCLASSAPI Ward : public ISurface
{
public:
    LAPI static ISurface* create( Context* context, const Parameters& params );

    LAPI Ward( Context* context );
    LAPI ~Ward();

    LAPI void setDiffuseReflectance( const Color& reflectance );
    LAPI void setSpecularReflectance( const Color& reflectance );
    LAPI void setAlphaU( float alpha_u );
    LAPI void setAlphaV( float alpha_v );

    LAPI const char* name()const;
    LAPI const char* sampleBSDFFunctionName()const;
    LAPI const char* evaluateBSDFFunctionName()const;
    LAPI const char* pdfFunctionName()const;
    LAPI const char* emissionFunctionName()const;

    LAPI void setVariables( VariableContainer& container ) const ;

private:
    Color m_diffuse_reflectance;
    Color m_specular_reflectance;
    float m_alpha_u;
    float m_alpha_v;

};

}

#endif // LEGION_OBJECTS_SURFACE_WARD_HPP_
