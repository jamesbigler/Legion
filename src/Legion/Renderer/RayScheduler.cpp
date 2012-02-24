
#include <Legion/Renderer/RayScheduler.hpp>
#include <Legion/Renderer/RayTracer.hpp>
#include <Legion/Scene/Film/IFilm.hpp>
#include <Legion/Common/Util/Assert.hpp>

using namespace legion;

RayScheduler::RayScheduler() 
{
}

void RayScheduler::setFilm( IFilm* film )
{
    m_film = film;
}

void RayScheduler::setCamera( ICamera* camera )
{
    m_camera = camera;
}


void RayScheduler::getPass( unsigned num_rays, Ray* rays, PixelID* pixel_ids )
{
    LEGION_TODO();
}

