

#ifndef LEGION_CONTEXT_H_
#define LEGION_CONTEXT_H_

#include <Private/APIBase.hpp>

namespace legion
{


class ICamera;
class IFilm;
class Mesh;

class Context : public APIBase 
{
public:
    explicit Context( const std::string& name );
    ~Context();

    void addMesh( const Mesh* mesh );

    void setActiveCamera( const ICamera* camera );
    void setActiveFilm( const IFilm* film );

    void render();

private:

    class Impl;
    std::tr1::shared_ptr<Impl> m_impl;
};


}
#endif // LEGION_CONTEXT_H_
