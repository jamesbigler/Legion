

#ifndef LEGION_CONTEXT_H_
#define LEGION_CONTEXT_H_

#include <private/APIBase.hpp>

namespace legion
{


class Camera;
class Mesh;
class Film;

class Context : public APIBase 
{
public:
    explicit Context( const std::string& name );
    ~Context();

    void addMesh( const Mesh& mesh );

    void setActiveCamera( const Camera& camera );
    void setActiveFilm( const Film& camera );

    void render();

private:

    class Impl;
    std::shared_ptr<Impl> m_impl;
};


}
#endif // LEGION_CONTEXT_H_
