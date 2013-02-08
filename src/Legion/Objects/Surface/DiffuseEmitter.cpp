

#include <Legion/Objects/Surface/DiffuseEmitter.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Common/Util/Parameters.hpp>


using namespace legion;


ISurface* DiffuseEmitter::create( Context* context, const Parameters& params )
{
    DiffuseEmitter* de = new DiffuseEmitter( context );
    de->setRadiance( params.get( "radiance", Color( 1.0f, 1.0f, 1.0f ) ) );
    return de;
}


DiffuseEmitter::DiffuseEmitter( Context* context )
    : ISurface( context ),
      m_radiance( 1.0f, 1.0f, 1.0f )
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
