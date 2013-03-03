

#include <Legion/Objects/Surface/Metal.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Common/Util/Parameters.hpp>


using namespace legion;

    
ISurface* Metal::create( Context* context, const Parameters& params )
{
    Metal* metal = new Metal( context );

    ITexture* reflectance;
    if( !params.get( "reflectance", reflectance ) )
        throw Exception( "Metal::create: no reflectance texture found" );
    metal->setReflectance( reflectance );

    ITexture* alpha;
    if( !params.get( "alpha", alpha) )
        throw Exception( "Metal::create: no alpha texture found" );
    metal->setAlpha( alpha ); 

    ITexture* eta;
    if( !params.get( "eta", eta) )
        throw Exception( "Metal::create: no eta texture found" );
    metal->setEta( eta ); 

    ITexture* k;
    if( !params.get( "k", k) )
        throw Exception( "Metal::create: no k texture found" );
    metal->setK( k ); 

    return metal;
}



Metal::Metal( Context* context )
    : ISurface( context ),
      m_reflectance( 0 ),
      m_alpha( 0 ),
      m_eta( 0 ),
      m_k( 0 )
{
}


Metal::~Metal()
{
}


void Metal::setReflectance( const ITexture* reflectance )
{
    m_reflectance = reflectance;
}
    

void Metal::setAlpha( const ITexture* alpha )
{
    m_alpha = alpha;
}
    

void Metal::setEta( const ITexture* eta )
{
    m_eta = eta;
}
    

void Metal::setK( const ITexture* k )
{
    m_k = k;
}
    

const char* Metal::name()const
{
    return "Metal";
}


const char* Metal::sampleBSDFFunctionName()const
{
    return "metalSampleBSDF";
}


const char* Metal::evaluateBSDFFunctionName()const
{
    return "metalEvaluateBSDF";
}


const char* Metal::pdfFunctionName()const
{
    return "metalPDF";
}
    

const char* Metal::emissionFunctionName()const
{
    return "nullSurfaceEmission";
}


void Metal::setVariables( VariableContainer& container ) const
{
    container.setTexture( "reflectance", m_reflectance );
    container.setTexture( "alpha",       m_alpha       );
    container.setTexture( "eta",         m_eta         );
    container.setTexture( "k",           m_k           );
}
