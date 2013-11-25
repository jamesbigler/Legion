
#ifndef LEGION_OBJECTS_SURFACE_DIELECTRIC_HPP_
#define LEGION_OBJECTS_SURFACE_DIELECTRIC_HPP_


#include <Legion/Objects/Surface/ISurface.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>
#include <Legion/Core/Color.hpp>


namespace legion
{

class VariableContainer;
class ITexture;

class LAPI Dielectric : public ISurface
{
public:
    LAPI static ISurface* create( Context* context, const Parameters& params );

    LAPI Dielectric( Context* context );
    LAPI ~Dielectric();
    LAPI 
    LAPI void setIOROut( float ior_out );
    LAPI void setIORIn ( float ior_in  );

    LAPI void setAbsorption   ( const Color& absorption    );
    LAPI void setReflectance  ( const Color& reflectance   );
    LAPI void setTransmittance( const Color& transmittance );

    LAPI 
    LAPI const char* name()const;
    LAPI const char* sampleBSDFFunctionName()const;
    LAPI const char* evaluateBSDFFunctionName()const;
    LAPI const char* pdfFunctionName()const;
    LAPI const char* emissionFunctionName()const;

    LAPI void setVariables( VariableContainer& container ) const ;

private:
    float m_ior_out;
    float m_ior_in;
    Color m_absorption;
    Color m_reflectance;
    Color m_transmittance;
};

}

#endif // LEGION_OBJECTS_SURFACE_DIELECTRIC_HPP_
