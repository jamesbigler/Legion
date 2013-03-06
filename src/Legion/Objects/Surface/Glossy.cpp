

#include <Legion/Objects/Surface/Glossy.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Common/Util/Parameters.hpp>


using namespace legion;

    
ISurface* Glossy::create( Context* context, const Parameters& params )
{
    Glossy* glossy = new Glossy( context );

    ITexture* reflectance;
    if( !params.get( "reflectance", reflectance ) )
        throw Exception( "Glossy::create: no reflectance texture found" );
    glossy->setReflectance( reflectance );

    ITexture* alpha;
    if( !params.get( "alpha", alpha) )
        throw Exception( "Glossy::create: no alpha texture found" );
    glossy->setAlpha( alpha ); 

    ITexture* eta;
    if( !params.get( "eta", eta) )
        throw Exception( "Glossy::create: no eta texture found" );
    glossy->setEta( eta ); 

    return glossy;
}



Glossy::Glossy( Context* context )
    : ISurface( context ),
      m_reflectance( 0 ),
      m_alpha( 0 )
{
}


Glossy::~Glossy()
{
}


void Glossy::setReflectance( const ITexture* reflectance )
{
    m_reflectance = reflectance;
}
    

void Glossy::setAlpha( const ITexture* alpha )
{
    m_alpha = alpha;
}
    

void Glossy::setEta( const ITexture* eta )
{
    m_eta = eta;
}
    

const char* Glossy::name()const
{
    return "Glossy";
}


const char* Glossy::sampleBSDFFunctionName()const
{
    return "glossySampleBSDF";
}


const char* Glossy::evaluateBSDFFunctionName()const
{
    return "glossyEvaluateBSDF";
}


const char* Glossy::pdfFunctionName()const
{
    return "glossyPDF";
}
    

const char* Glossy::emissionFunctionName()const
{
    return "nullSurfaceEmission";
}


void Glossy::setVariables( VariableContainer& container ) const
{
    container.setTexture( "reflectance", m_reflectance );
    container.setTexture( "alpha",       m_alpha       );
    container.setTexture( "eta",         m_eta         );
}
