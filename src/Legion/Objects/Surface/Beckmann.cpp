

#include <Legion/Objects/Surface/Beckmann.hpp>
#include <Legion/Objects/VariableContainer.hpp>
#include <Legion/Common/Util/Parameters.hpp>


using namespace legion;

    
ISurface* Beckmann::create( Context* context, const Parameters& params )
{
    Beckmann* beckmann = new Beckmann( context );

    ITexture* reflectance;
    if( !params.get( "reflectance", reflectance ) )
        throw Exception( "Beckmann::create: no reflectance texture found" );
    beckmann->setReflectance( reflectance );

    ITexture* alpha;
    if( !params.get( "alpha", alpha) )
        throw Exception( "Beckmann::create: no alpha texture found" );
    beckmann->setAlpha( alpha ); 

    return beckmann;
}



Beckmann::Beckmann( Context* context )
    : ISurface( context ),
      m_reflectance( 0 ),
      m_alpha( 0 )
{
}


Beckmann::~Beckmann()
{
}


void Beckmann::setReflectance( const ITexture* reflectance )
{
    m_reflectance = reflectance;
}
    

void Beckmann::setAlpha( const ITexture* alpha )
{
    m_alpha = alpha;
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
    /*
    container.setFloat( "reflectance", Color( 0.5f ) );
    container.setFloat( "alpha",       0.02f         );
    */
    container.setTexture( "reflectance", m_reflectance );
    container.setTexture( "alpha",       m_alpha );
}
