
#ifndef LEGION_ENGINE_CONTEXT_IMPL_HPP_
#define LEGION_ENGINE_CONTEXT_IMPL_HPP_

#include <Legion/Core/Light.hpp>
#include <Legion/Core/Context.hpp>

#include <vector>

namespace legion
{
    class ICamera;
    class IFilm;
    class Mesh;

    class Context::Impl
    {
    public:
        Impl();
        ~Impl();

        void addMesh ( const Mesh* mesh );
        void addLight( const ILightShader* light_shader );
        void addLight( const ILightShader* light_shader, const Mesh* light_geometry );

        void setActiveCamera( const ICamera* camera );
        void setActiveFilm( const IFilm* film );


        void render();

    private:

        void preprocess();
        void doRender();
        void postprocess();


        std::vector<const Mesh*> m_meshes;
        std::vector<Light>       m_lights;
        const ICamera*           m_camera;
        const IFilm*             m_film;
    };
 
}

#endif // LEGION_ENGINE_CONTEXT_IMPL_HPP_
