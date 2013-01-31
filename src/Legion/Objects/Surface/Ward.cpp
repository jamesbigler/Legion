

#include <Legion/Objects/Surface/Ward.hpp>
#include <Legion/Core/VariableContainer.hpp>


using namespace legion;


Ward::Ward( Context* context )
    : ISurface( context ),
      m_diffuse_reflectance( 0.5f, 0.5f, 0.5f ),
      m_specular_reflectance( 0.5f, 0.5f, 0.5f ),
      m_alpha_u( 0.0f ),
      m_alpha_v( 0.0f )
{
}


Ward::~Ward()
{
}


void Ward::setDiffuseReflectance( const Color& reflectance )
{
    m_diffuse_reflectance = reflectance;
}
    

void Ward::setSpecularReflectance( const Color& reflectance )
{
    m_specular_reflectance = reflectance;
}
    

void Ward::setAlphaU( float alpha_u )
{
    m_alpha_u = alpha_u;
}

void Ward::setAlphaV( float alpha_v )
{
    m_alpha_v = alpha_v;
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
    container.setFloat( "diff_reflectance", m_diffuse_reflectance  );
    container.setFloat( "spec_reflectance", m_specular_reflectance );
    container.setFloat( "alpha_u",          m_alpha_u              );
    container.setFloat( "alpha_v",          m_alpha_v              );
    container.setFloat( "diffuse_weight", 
        m_diffuse_reflectance.luminance() / 
        ( m_diffuse_reflectance.luminance() + 
          m_specular_reflectance.luminance() ) );
}
