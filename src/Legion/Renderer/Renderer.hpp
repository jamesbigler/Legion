
#ifndef LEGION_RENDERER_RENDERER_HPP_
#define LEGION_RENDERER_RENDERER_HPP_

#include <Legion/Common/Util/Noncopyable.hpp>
#include <Legion/Renderer/LightSet.hpp>
#include <Legion/Renderer/RayScheduler.hpp>
#include <Legion/Renderer/RayTracer.hpp>
#include <Legion/Renderer/ShadingEngine.hpp>

namespace legion
{

class Mesh;
class IFilm;
class ICamera;
class ILightShader;

class Renderer : public Noncopyable
{
public:
    Renderer();

    void render();

    void setFilm( IFilm* film );

    void setCamera( ICamera* camera);


    void setSamplesPerPixel( const Index2& spp  );

    void addLight( const ILightShader* light_shader );

    void addMesh( Mesh* mesh );

    void removeMesh( Mesh* mesh );

    void     setMaxRayDepth( unsigned );
    unsigned maxRayDepth()const;

    optix::Buffer createBuffer();
private:

    RayScheduler   m_ray_scheduler;
    RayTracer      m_ray_tracer;
    ShadingEngine  m_shading_engine;
    LightSet       m_light_set;

    Index2         m_spp;

    IFilm*         m_film;
    ICamera*       m_camera;
};

}

#endif // LEGION_RENDERER_RENDERER_HPP_
