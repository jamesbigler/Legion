

#include <Legion/Objects/Surface/Ward.hpp>
#include <Legion/Core/VariableContainer.hpp>


using namespace legion;


Ward::Ward( Context* context )
    : ISurface( context )
{
}


Ward::~Ward()
{
}


void Ward::setReflectance( const Color& reflectance )
{
    m_reflectance = reflectance;
}
    

const char* Ward::name()const
{
    return "Ward";
}


const char* Ward::sampleBSDFFunctionName()const
{
    return "wardSampleBSDF";
}


const char* Ward::evaluateBSDFFunctionName()const
{
    return "wardEvaluateBSDF";
}


const char* Ward::pdfFunctionName()const
{
    return "wardPDF";
}
    

const char* Ward::emissionFunctionName()const
{
    return "nullSurfaceEmission";
}


void Ward::setVariables( const VariableContainer& container ) const
{
    container.setFloat( "reflectance", m_reflectance );
}
