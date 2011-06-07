
#ifndef LEGION_CONTEXT_H_
#define LEGION_CONTEXT_H_


// TODO:
//   - Remove non-copyable semantics and make Impl* member be shared_ptr?

namespace legion
{

class Camera;
class Mesh;

class Context : public NonCopyable 
{
public:
    Context( const std::string& name );
    ~Context();

    void addMesh( const Mesh& mesh );

    void setActiveCamera( const Camera& camera );

    void render();

private:

    class Impl;
    Impl* impl;
};

}
#endif // LEGION_CONTEXT_H_
