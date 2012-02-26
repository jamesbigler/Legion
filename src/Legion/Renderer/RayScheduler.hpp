
#ifndef LEGION_RENDERER_RAYSCHEDULER_HPP_
#define LEGION_RENDERER_RAYSCHEDULER_HPP_

#include <vector>
#include <optixu/optixpp_namespace.h>
#include <Legion/Core/Vector.hpp>

namespace legion
{

class IFilm;
class ICamera;
class Ray;

class RayScheduler
{
public:
    struct PixelID
    {
        Index2  pixel;
        float   weight;
    };

    RayScheduler();

    bool finished()const;

    void setSamplesPerPixel( const Index2& spp ); 
    void setTimeInterval( const Vector2& time_interval ); 
    void setFilm( IFilm* film );
    void setCamera( ICamera* camera );

    void getPass( optix::Buffer rays, std::vector<PixelID>& pixel_ids );
    
private:

    IFilm*    m_film;
    ICamera*  m_camera;

    Index2 m_spp;
    Index2 m_current_sample;
    Vector2 m_time_interval;
};

}
#endif // LEGION_RENDERER_RAYSCHEDULER_HPP_