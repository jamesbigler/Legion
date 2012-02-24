
#ifndef LEGION_RENDERER_RAYSCHEDULER_HPP_
#define LEGION_RENDERER_RAYSCHEDULER_HPP_

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
        unsigned pixel;
        float    weight;
    };

    RayScheduler();
    
    void setFilm( IFilm* film );
    void setCamera( ICamera* camera );

    void getPass( unsigned num_rays, Ray* rays, PixelID* pixel_ids );
    
private:

    IFilm*  m_film;
    ICamera*  m_camera;
};

}
#endif // LEGION_RENDERER_RAYSCHEDULER_HPP_
