

#ifndef LEGION_CORE_CONTEXT_H_
#define LEGION_CORE_CONTEXT_H_

#include <Legion/Common/Util/Noncopyable.hpp>
#include <Legion/Core/APIBase.hpp>
#include <Legion/Core/Light.hpp>
#include <Legion/Renderer/Renderer.hpp>






#include <iostream>

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

    void addMesh( Mesh* mesh );

    // TODO: * create baseclass MeshLight < ILightShader implements sample()
    //       * Add internal sample() method to mesh
    //       * Add internal setLightShader() method to mesh
    void addLight( const ILightShader* light_shader );

    void setActiveCamera( ICamera* camera );

    void setActiveFilm  ( IFilm* film );

    void render();

    //
    // Internal interface -- will exist only in Pimpl class
    //
    optix::Buffer createOptixBuffer();

private:


    std::vector<const Mesh*> m_meshes;
    std::vector<Light>       m_lights;
    const ICamera*           m_camera;
    const IFilm*             m_film;

    Renderer                 m_renderer;
};


}
#endif // LEGION_CORE_CONTEXT_H_
