

#include <Legion/Objects/Surface/Dielectric.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Common/Util/Parameters.hpp>
#include <Legion/Common/Util/Logger.hpp>


using namespace legion;

    
ISurface* Dielectric::create( Context* context, const Parameters& params )
{
    Dielectric* dielectric = new Dielectric( context );
    dielectric->setIOROut( params.get( "ior_out", 1.0f ) );
    dielectric->setIORIn ( params.get( "ior_in",  1.5f ) );
    dielectric->setAbsorption   ( params.get( "absorption",  Color( 1.0f ) ) );
    dielectric->setReflectance  ( params.get( "reflectance", Color( 1.0f ) ) );
    dielectric->setTransmittance( params.get( "transmitance",Color( 1.0f ) ) );
    params.reportUnused( std::cerr ); // TODO: need to fix logger to handle this
    return dielectric;
}



Dielectric::Dielectric( Context* context )
    : ISurface( context ),
      m_ior_out( 1.0f ),
      m_ior_in( 1.5f ),
      m_absorption   ( 1.0f, 1.0f, 1.0f ),
      m_reflectance  ( 1.0f, 1.0f, 1.0f ),
      m_transmittance( 1.0f, 1.0f, 1.0f )
{
}


Dielectric::~Dielectric()
{
}


void Dielectric::setIOROut( float ior_out )
{
    m_ior_out = ior_out;
}


void Dielectric::setIORIn( float ior_in )
{
    m_ior_in = ior_in;
}


void Dielectric::setAbsorption( const Color& absorption )
{
    m_absorption = absorption;
}


void Dielectric::setReflectance( const Color& reflectance)
{
    m_reflectance = reflectance;
}


void Dielectric::setTransmittance( const Color& transmittance)
{
    m_transmittance = transmittance;
}


const char* Dielectric::name()const
{
    return "Dielectric";
}


const char* Dielectric::sampleBSDFFunctionName()const
{
    return "dielectricSampleBSDF";
}


const char* Dielectric::evaluateBSDFFunctionName()const
{
    return "dielectricEvaluateBSDF";
}


const char* Dielectric::pdfFunctionName()const
{
    return "dielectricPDF";
}
    

const char* Dielectric::emissionFunctionName()const
{
    return "nullSurfaceEmission";
}


void Dielectric::setVariables( VariableContainer& container ) const
{
    container.setFloat( "ior_out",    m_ior_out );
    container.setFloat( "ior_in",     m_ior_in  );
    container.setFloat( "absorption", m_absorption);
    container.setFloat( "reflectance", m_reflectance );
    container.setFloat( "transmittance", m_transmittance );
}
