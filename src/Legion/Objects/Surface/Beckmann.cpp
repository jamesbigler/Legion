

#include <Legion/Objects/Surface/Beckmann.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Common/Util/Parameters.hpp>


using namespace legion;

    
ISurface* Beckmann::create( Context* context, const Parameters& params )
{
    Beckmann* beckmann = new Beckmann( context );
    beckmann->setReflectance( 
            params.get( "reflectance", static_cast<ITexture*>( 0 ) )
            );
    return beckmann;
}



Beckmann::Beckmann( Context* context )
    : ISurface( context ),
      m_reflectance( 0 )
{
}


Beckmann::~Beckmann()
{
}


void Beckmann::setReflectance( const ITexture* reflectance )
{
    m_reflectance = reflectance;
}
    

const char* Beckmann::name()const
{
    return "Beckmann";
}


const char* Beckmann::sampleBSDFFunctionName()const
{
    return "beckmannSampleBSDF";
}


const char* Beckmann::evaluateBSDFFunctionName()const
{
    return "beckmannEvaluateBSDF";
}


const char* Beckmann::pdfFunctionName()const
{
    return "beckmannPDF";
}
    

const char* Beckmann::emissionFunctionName()const
{
    return "nullSurfaceEmission";
}


void Beckmann::setVariables( VariableContainer& container ) const
{
    container.setFloat( "reflectance", Color( 0.5f ) );
}
