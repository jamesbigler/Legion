

#include <Legion/Objects/Surface/Lambertian.hpp>
#include <Legion/Core/VariableContainer.hpp>


using namespace legion;


Lambertian::Lambertian( Context* context )
    : ISurface( context )
{
}


Lambertian::~Lambertian()
{
}


void Lambertian::setReflectance( const Color& reflectance )
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


void Lambertian::setVariables( const VariableContainer& container ) const
{
    container.setFloat( "reflectance", m_reflectance );
}
