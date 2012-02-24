

#ifndef LEGION_CORE_CONTEXT_H_
#define LEGION_CORE_CONTEXT_H_

#include <Legion/Common/Util/Noncopyable.hpp>
#include <Legion/Core/APIBase.hpp>
#include <Legion/Core/Light.hpp>
#include <Legion/RayTracer/RayTracer.hpp>


namespace legion
{


class ICamera;
class IFilm;
class ILightShader;
class Mesh;

class Context : public APIBase 
{
public:
    //
    // External interface -- will be wrapped via Pimpl
    //
    explicit   Context( const std::string& name );
    ~Context();

    void addMesh ( Mesh* mesh );
    void addLight( const ILightShader* light_shader );
    void addLight( const ILightShader* light_shader, const Mesh* light_geometry );

    void setActiveCamera( const ICamera* camera );
    void setActiveFilm( const IFilm* film );


    void render();

    //
    // Internal interface -- will exist only in Pimpl class
    //
    optix::Buffer createOptixBuffer();

private:

    void preprocess();
    void doRender();
    void postprocess();

    std::vector<const Mesh*> m_meshes;
    std::vector<Light>       m_lights;
    const ICamera*           m_camera;
    const IFilm*             m_film;

    RayTracer          m_ray_tracer;
};


}
#endif // LEGION_CORE_CONTEXT_H_
