

#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Common/Util/Optix.hpp>
#include <Legion/Scene/Camera/ICamera.hpp>
#include <Legion/Scene/Film/IFilm.hpp>
#include <Legion/Scene/SurfaceShader/ISurfaceShader.hpp>
#include <Legion/Scene/LightShader/ILightShader.hpp>
#include <Legion/Core/Context.hpp>


using namespace legion;


/******************************************************************************\
 *                                                                            *
 *                                                                            *
\******************************************************************************/

Context::Context( const std::string& name ) 
    : APIBase( this, name )
{
    LLOG_INFO << "Creating Context::Impl";

    std::vector<std::string> paths;
    paths.push_back( "/Users/keithm/Code/Legion/build_debug/src/Legion/" );
    m_optix.setProgramSearchPath( paths );
    m_optix.loadProgram( "cuda_compile_ptx_generated_hit_programs.cu.ptx",
                         "XXclosestHit" );
}


Context::~Context()
{
    LLOG_INFO << "Destroying Context::Impl";
}


void Context::addMesh( const Mesh* mesh )
{
    // TODO: add NULL check to all of these
    LLOG_INFO << "Adding mesh <" << mesh->getName() << ">";
    m_meshes.push_back( mesh );
}


void Context::addLight( const ILightShader* light_shader )
{
    Light light;
    light.shader   = light_shader;
    light.geometry = 0u;
    LLOG_INFO << "Adding light <" << light.getName() << ">";
    m_lights.push_back( light );
}


void Context::addLight( const ILightShader* light_shader, const Mesh* light_geometry )
{
    Light light;
    light.shader   = light_shader;
    light.geometry = light_geometry;
    LLOG_INFO << "Adding light <" << light.getName() << ">";
    m_lights.push_back( light );
}


void Context::setActiveCamera( const ICamera* camera )
{
    LLOG_INFO << "Adding camera <" << camera->getName() << ">";
    m_camera = camera;
}


void Context::setActiveFilm( const IFilm* film )
{
    LLOG_INFO << "Adding film <" << film->getName() << ">";
    m_film = film;
}


void Context::preprocess()
{
}


void Context::doRender()
{
}


void Context::postprocess()
{
}

void Context::render()
{
    LLOG_INFO << "rendering ....";

    preprocess();

    doRender();

    postprocess();



    const Index2  image_dims  = m_film->getDimensions();
}


