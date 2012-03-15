
#include <Legion/Core/Ray.hpp>
#include <Legion/Renderer/RayScheduler.hpp>
#include <Legion/Renderer/Renderer.hpp>
#include <Legion/Renderer/Cuda/Shared.hpp>
#include <Legion/Scene/Camera/ICamera.hpp>
#include <Legion/Scene/Film/IFilm.hpp>
#include <Legion/Common/Util/AutoTimerHelpers.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Common/Util/Timer.hpp>


using namespace legion;


Renderer::Renderer()
    : m_shading_engine( m_ray_tracer ),
      m_spp( 1, 1 )
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


void Renderer::setSamplesPerPixel( const Index2& spp  )
{
    m_spp = spp;
}


void Renderer::render()
{
    m_film->shutterOpen();

    std::vector<RayScheduler::PixelID> pixel_ids;
    std::vector<Ray>                   rays;

    m_ray_scheduler.setSamplesPerPixel( m_spp );

    m_shading_engine.reset();

    LoopTimerInfo   scheduling_time( "    Scheduling rays" );
    LoopTimerInfo   shading_time   ( "    Shading        " ); 
    LoopTimerInfo   film_time      ( "    Film splatting " ); 

    Timer total_time;
    total_time.start();
    m_ray_tracer.preprocess();

    while( !m_ray_scheduler.finished() )
    {
        {
            AutoTimerRef<LoopTimerInfo> schedule_timer( scheduling_time );
            m_ray_scheduler.getPass( rays, pixel_ids );
        }

        {
            AutoTimerRef<LoopTimerInfo> shading_timer( shading_time );
            m_shading_engine.shade( rays );
        }

        {
            AutoTimerRef<LoopTimerInfo> shading_timer( film_time );
            const ShadingEngine::Results& results = m_shading_engine.getResults();

            for( unsigned int i = 0; i < pixel_ids.size(); ++i )
            {
                m_film->addSample( pixel_ids[i].pixel,
                                   pixel_ids[i].weight,
                                   results[i] );
            }
            m_film->passComplete();
        }
    }
    total_time.stop();

    LLOG_INFO << "Total render time: " << total_time.getTimeElapsed();
    scheduling_time.log();
    shading_time.log();
    m_shading_engine.logTimerInfo();
    film_time.log();
    LLOG_INFO << "    Summed totals: " << scheduling_time.total_time  + 
                                          shading_time.total_time     + 
                                          film_time.total_time;
    LLOG_INFO << "    Summed avgs  : " << scheduling_time.averageTime()  + 
                                          shading_time.averageTime()     + 
                                          film_time.averageTime();

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

