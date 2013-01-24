

#include <Legion/Objects/Surface/DiffuseEmitter.hpp>
#include <Legion/Core/VariableContainer.hpp>


using namespace legion;


DiffuseEmitter::DiffuseEmitter( Context* context )
    : ISurface( context )
{
}


DiffuseEmitter::~DiffuseEmitter()
{
}


void DiffuseEmitter::setRadiance( const Color& radiance )
{
    m_radiance = radiance;
}
    

const char* DiffuseEmitter::name()const
{
    return "DiffuseEmitter";
}


const char* DiffuseEmitter::sampleBSDFFunctionName()const
{
    return "nullSurfaceSampleBSDF";
}


const char* DiffuseEmitter::evaluateBSDFFunctionName()const
{
    return "nullSurfaceEvaluateBSDF";
}


const char* DiffuseEmitter::pdfFunctionName()const
{
    return "nullSurfacePDF";
}
    

const char* DiffuseEmitter::emissionFunctionName()const
{
    return "diffuseEmitterEmission";
}


void DiffuseEmitter::setVariables( VariableContainer& container ) const
{
    container.setFloat( "radiance", m_radiance );
}
