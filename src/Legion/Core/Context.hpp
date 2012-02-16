

#ifndef LEGION_CONTEXT_H_
#define LEGION_CONTEXT_H_

#include <Legion/Core/APIBase.hpp>
#include <tr1/memory>

namespace legion
{


class ICamera;
class IFilm;
class ILightShader;
class Mesh;

class Context : public APIBase 
{
public:
    explicit Context( const std::string& name );
    ~Context();

    void addMesh ( const Mesh* mesh );
    void addLight( const ILightShader* light_shader );
    void addLight( const ILightShader* light_shader, const Mesh* light_geometry );

    void setActiveCamera( const ICamera* camera );
    void setActiveFilm( const IFilm* film );

    void render();

private:

    class Impl;
    std::tr1::shared_ptr<Impl> m_impl;
};


}
#endif // LEGION_CONTEXT_H_
