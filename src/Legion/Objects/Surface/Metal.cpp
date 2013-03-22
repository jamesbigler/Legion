

#include <Legion/Objects/Surface/Metal.hpp>
#include <Legion/Objects/Texture/ConstantTexture.hpp>
#include <Legion/Objects/VariableContainer.hpp>
#include <Legion/Core/Color.hpp>
#include <Legion/Common/Util/Parameters.hpp>


using namespace legion;

    
Metal::MetalLookup Metal::s_metal_lookup;

    
ISurface* Metal::create( Context* context, const Parameters& params )
{
    Metal::initializeMetalLookup();

    Metal* metal = new Metal( context );

    std::string metal_type;
    if( params.get( "preset_eta_k", metal_type ) )
    {
        metal->setMetalType( metal_type );
    }
    else
    {
        ITexture* eta;
        if( !params.get( "eta", eta) )
            throw Exception( "Metal::create: no eta texture found" );
        metal->setEta( eta ); 

        ITexture* k;
        if( !params.get( "k", k) )
            throw Exception( "Metal::create: no k texture found" );
        metal->setK( k ); 
    }

    ITexture* reflectance;
    if( !params.get( "reflectance", reflectance ) )
        throw Exception( "Metal::create: no reflectance texture found" );
    metal->setReflectance( reflectance );

    ITexture* alpha;
    if( !params.get( "alpha", alpha) )
        throw Exception( "Metal::create: no alpha texture found" );
    metal->setAlpha( alpha ); 

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


void Metal::setMetalType( const std::string& type )
{
    if( !s_metal_lookup.count( type ) )
        throw Exception( "Metal::setMetalType: unknown type '" + type + "'" );

    std::pair<Color, Color> eta_k = s_metal_lookup[ type ];
    m_metal_type_eta.reset( new ConstantTexture( getContext() ) );
    m_metal_type_k.reset( new ConstantTexture( getContext() ) );

    m_metal_type_eta->set( eta_k.first  ); 
    m_metal_type_k->set( eta_k.second );

    setEta( m_metal_type_eta.get() );
    setK( m_metal_type_k.get() );
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

void Metal::addMetalLookup(
        const std::string& n,
        const Color& eta,
        const Color& k )
{
    s_metal_lookup.insert( std::make_pair( n, std::make_pair( eta, k ) ) );
}


void Metal::initializeMetalLookup()
{
    static bool initialized = false;
    if( initialized )
        return;
    initialized = true;

    addMetalLookup( "aluminum", Color( 1.661,  0.8814, 0.5211 ),
                                Color( 9.224,  6.273,  4.753  ) );
    addMetalLookup( "copper",   Color( 0.2129, 0.9196, 1.103  ),
                                Color( 3.918,  2.46,   2.137  ) );
    addMetalLookup( "gold",     Color( 0.1723, 0.3827, 1.437  ),
                                Color( 3.976,  2.381,  1.605  ) );
    addMetalLookup( "nickel",   Color( 2.361,  1.663,  1.468  ),
                                Color( 4.498,  3.051,  2.344  ) );
    addMetalLookup( "platinum", Color( 2.83,   1.996,  1.646  ),
                                Color( 5.001,  3.493,  2.796  ) );
    addMetalLookup( "silver",   Color( 0.1555, 0.1169, 0.1384 ),
                                Color( 4.822,  3.123,  2.146  ) );
}
