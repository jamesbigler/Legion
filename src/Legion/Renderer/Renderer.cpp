
#include <Legion/Renderer/Renderer.hpp>
#include <Legion/Renderer/RayScheduler.hpp>
#include <Legion/Scene/Film/IFilm.hpp>
#include <Legion/Scene/Camera/ICamera.hpp>

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

    /// TODO: this vec needs to be persistant
    std::vector<RayScheduler::PixelID> pixel_ids;
    optix::Buffer rays = m_ray_tracer.getRayBuffer();

    m_ray_scheduler.getPass( rays, pixel_ids );
    m_ray_tracer.traceRays( RayTracer::CLOSEST_HIT );
    optix::Buffer trace_results = m_ray_tracer.getResults();

    m_shading_engine.shade( pixel_ids.size(),
                            static_cast<const Ray*>( rays->map() ), 
                            static_cast<const LocalGeometry*>( 
                                trace_results->map() ) );
    trace_results->unmap();

    const ShadingEngine::Results& shading_results =
           m_shading_engine.getResults();

    for( unsigned int i = 0; i < pixel_ids.size(); ++i )
    {
        m_film->addSample( pixel_ids[i].pixel,
                           shading_results[i],
                           pixel_ids[0].weight );
    }

    m_film->passComplete();

    m_film->shutterClose();
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

