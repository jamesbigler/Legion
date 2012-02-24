
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
}


void Renderer::setFilm( IFilm* film )
{
    m_film = film;
}


void Renderer::render()
{
    Index2   film_dims  = m_film->getDimensions();
    unsigned num_pixels = film_dims.x() * film_dims.y();
    Ray*     rays       = m_ray_tracer.getRayData( num_pixels );

    //
    //
    //
    //
    //
    //
    /// TODO: uhnhh, leak
    //
    //
    //
    //
    //
    //
    //
    RayScheduler::PixelID* pixel_ids = new RayScheduler::PixelID[ num_pixels ];

    m_ray_scheduler.getPass( num_pixels, rays, pixel_ids );
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

