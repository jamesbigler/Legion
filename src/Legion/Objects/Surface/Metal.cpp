

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

    addMetalLookup( "aluminum", Color( 1.661f,  0.8814f, 0.5211f ),
                                Color( 9.224f,  6.273f,  4.753f  ) );
    addMetalLookup( "copper",   Color( 0.2129f, 0.9196f, 1.103f  ),
                                Color( 3.918f,  2.46f,   2.137f  ) );
    addMetalLookup( "gold",     Color( 0.1723f, 0.3827f, 1.437f  ),
                                Color( 3.976f,  2.381f,  1.605f  ) );
    addMetalLookup( "nickel",   Color( 2.361f,  1.663f,  1.468f  ),
                                Color( 4.498f,  3.051f,  2.344f  ) );
    addMetalLookup( "platinum", Color( 2.83f,   1.996f,  1.646f  ),
                                Color( 5.001f,  3.493f,  2.796f  ) );
    addMetalLookup( "silver",   Color( 0.1555f, 0.1169f, 0.1384f ),
                                Color( 4.822f,  3.123f,  2.146f  ) );
}
