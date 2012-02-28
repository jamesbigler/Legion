
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
    /// TODO: this vec needs to be persistant
    std::vector<RayScheduler::PixelID> pixel_ids;
    optix::Buffer rays       = m_ray_tracer.getRayBuffer();

    m_ray_scheduler.getPass( rays, pixel_ids );
    m_ray_tracer.traceRays( RayTracer::CLOSEST_HIT );
}


void Renderer::addMesh( Mesh* mesh )
{
    m_ray_tracer.addMesh( mesh );
}


void Renderer::removeMesh( Mesh* mesh )
{
    m_ray_tracer.removeMesh( mesh );
}


optix::Buffer Renderer::createBuffer()
{
    return m_ray_tracer.createBuffer();
}

