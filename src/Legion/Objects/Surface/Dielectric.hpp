
#ifndef LEGION_OBJECTS_SURFACE_DIELECTRIC_HPP_
#define LEGION_OBJECTS_SURFACE_DIELECTRIC_HPP_


#include <Legion/Objects/Surface/ISurface.hpp>
#include <Legion/Core/Color.hpp>


namespace legion
{

class VariableContainer;
class ITexture;

class Dielectric : public ISurface
{
public:
    static ISurface* create( Context* context, const Parameters& params );

    Dielectric( Context* context );
    ~Dielectric();
    
    void setIOROut( float ior_out );
    void setIORIn( float ior_in );
    void setAbsorption( const Color& absorption );

    
    const char* name()const;
    const char* sampleBSDFFunctionName()const;
    const char* evaluateBSDFFunctionName()const;
    const char* pdfFunctionName()const;
    const char* emissionFunctionName()const;

    void setVariables( VariableContainer& container ) const ;

private:
    float m_ior_out;
    float m_ior_in;
    Color m_absorption;
    Color m_reflectance;
    Color m_transmittance;
};

}

#endif // LEGION_OBJECTS_SURFACE_DIELECTRIC_HPP_
