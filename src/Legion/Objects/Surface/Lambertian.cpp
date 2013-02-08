

#include <Legion/Objects/Surface/Lambertian.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Common/Util/Parameters.hpp>


using namespace legion;

    
ISurface* Lambertian::create( Context* context, const Parameters& params )
{
    Lambertian* lambertian = new Lambertian( context );
    lambertian->setReflectance( 
            params.get( "reflectance", static_cast<ITexture*>( 0 ) )
            );
    return lambertian;
}



Lambertian::Lambertian( Context* context )
    : ISurface( context ),
      m_reflectance( 0 )
{
}


Lambertian::~Lambertian()
{
}


void Lambertian::setReflectance( const ITexture* reflectance )
{
    m_reflectance = reflectance;
}
    

const char* Lambertian::name()const
{
    return "Lambertian";
}


const char* Lambertian::sampleBSDFFunctionName()const
{
    return "lambertianSampleBSDF";
}


const char* Lambertian::evaluateBSDFFunctionName()const
{
    return "lambertianEvaluateBSDF";
}


const char* Lambertian::pdfFunctionName()const
{
    return "lambertianPDF";
}
    

const char* Lambertian::emissionFunctionName()const
{
    return "nullSurfaceEmission";
}


void Lambertian::setVariables( VariableContainer& container ) const
{
    container.setTexture( "reflectance", m_reflectance );
}
