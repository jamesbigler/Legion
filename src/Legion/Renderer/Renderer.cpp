
#include <Legion/Core/Ray.hpp>
#include <Legion/Renderer/RayScheduler.hpp>
#include <Legion/Renderer/Renderer.hpp>
#include <Legion/Scene/Camera/ICamera.hpp>
#include <Legion/Scene/Film/IFilm.hpp>
#include <Legion/Common/Util/Logger.hpp>

using namespace legion;


Renderer::Renderer()
{
}


void Renderer::setCamera( ICamera* camera )
{
    m_camera = camera;
    m_ray_scheduler.setCamera( camera );
}


void Renderer::setFilm( IFilm* film )
{
    m_film = film;
    m_ray_scheduler.setFilm( film );
}


void Renderer::render()
{
    m_film->shutterOpen();

    std::vector<RayScheduler::PixelID> pixel_ids;
    std::vector<Ray>                   rays;

    m_ray_scheduler.setSamplesPerPixel( Index2( 5, 5 ) );
    //m_ray_scheduler.setSamplesPerPixel( Index2( 1, 1 ) );

    //double compilation_time = 0.0;

    while( !m_ray_scheduler.finished() )
    {
        m_ray_scheduler.getPass( rays, pixel_ids );
        m_ray_tracer.traceRays( RayTracer::CLOSEST_HIT, rays );
        const LocalGeometry* trace_results = m_ray_tracer.getResults();

        m_shading_engine.shade( rays, trace_results );

        const std::vector<Color>& shading_results=m_shading_engine.getResults();

        for( unsigned int i = 0; i < pixel_ids.size(); ++i )
        {
            m_film->addSample( pixel_ids[i].pixel,
                               pixel_ids[i].weight,
                               shading_results[i] );
        }

        m_film->passComplete();
    }

    m_film->shutterClose();
}


void Renderer::addLight( const ILightShader* light_shader )
{
    m_shading_engine.addLight( light_shader );
}


void Renderer::addMesh( Mesh* mesh )
{
    m_ray_tracer.addMesh( mesh );
    m_shading_engine.addSurfaceShader( mesh->getShader() );
}


void Renderer::removeMesh( Mesh* mesh )
{
    m_ray_tracer.removeMesh( mesh );
}


optix::Buffer Renderer::createBuffer()
{
    return m_ray_tracer.createBuffer();
}

