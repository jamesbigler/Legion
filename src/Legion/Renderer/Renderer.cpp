
#include <Legion/Core/Ray.hpp>
#include <Legion/Renderer/RayScheduler.hpp>
#include <Legion/Renderer/Renderer.hpp>
#include <Legion/Renderer/Cuda/Shared.hpp>
#include <Legion/Scene/Camera/ICamera.hpp>
#include <Legion/Scene/Film/IFilm.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Common/Util/Timer.hpp>


using namespace legion;


namespace
{

struct TimerInfo
{
    TimerInfo( const std::string& name ) 
        : name( name ),
          iterations( 0u ),
          max_time( 0.0 ),
          total_time( 0.0 )
    {}

    void operator()( double time_elapsed )
    { 
        ++iterations;
        max_time = std::max( max_time, time_elapsed );
        total_time += time_elapsed;
    }

    void log()
    {
        LLOG_INFO << std::fixed
                  << name << "  max: " << max_time << " avg: "
                  << averageTime();
    }

    double averageTime()const
    { 
        return total_time / static_cast<double>( iterations );
    }

    std::string name;
    unsigned    iterations; 
    double      max_time;
    double      total_time;
};

}



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

    //m_ray_scheduler.setSamplesPerPixel( Index2( 5, 5 ) );
    m_ray_scheduler.setSamplesPerPixel( m_spp );

    //double compilation_time = 0.0;

    TimerInfo   scheduling_time( "    Scheduling rays" );
    TimerInfo   trace_time     ( "    Tracing rays   " );
    TimerInfo   shading_time   ( "    Shading        " ); 
    TimerInfo   film_time      ( "    Film splatting " ); 

    Timer total_time;
    total_time.start();
    m_ray_tracer.preprocess();

    while( !m_ray_scheduler.finished() )
    {
        {
            AutoTimerRef<TimerInfo> schedule_timer( scheduling_time );
            m_ray_scheduler.getPass( rays, pixel_ids );
        }

        {
            AutoTimerRef<TimerInfo> trace_timer( trace_time );
            m_ray_tracer.trace( RayTracer::CLOSEST_HIT, rays );
            m_ray_tracer.join(); // Finish async tracing
        }

        {
            AutoTimerRef<TimerInfo> shading_timer( shading_time );
            std::vector<LocalGeometry> trace_results;
            m_ray_tracer.getResults( trace_results );
            m_shading_engine.shade( rays, trace_results );
        }

        {
            AutoTimerRef<TimerInfo> shading_timer( film_time );
            const std::vector<Color>& shading_results = 
                m_shading_engine.getResults();

            for( unsigned int i = 0; i < pixel_ids.size(); ++i )
            {
                m_film->addSample( pixel_ids[i].pixel,
                                   pixel_ids[i].weight,
                                   shading_results[i] );
            }
            m_film->passComplete();
        }
    }
    total_time.stop();

    LLOG_INFO << "Total render time: " << total_time.getTimeElapsed();
    scheduling_time.log();
    trace_time.log();
    shading_time.log();
    film_time.log();
    LLOG_INFO << "    Summed totals: " << scheduling_time.total_time  + 
                                          trace_time.total_time       + 
                                          shading_time.total_time     + 
                                          film_time.total_time;
    LLOG_INFO << "    Summed avgs  : " << scheduling_time.averageTime()  + 
                                          trace_time.averageTime()       + 
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

