
#ifndef LEGION_CORE_MESH_HPP_
#define LEGION_CORE_MESH_HPP_


#include <Legion/Core/APIBase.hpp>
#include <Legion/Core/Vector.hpp>
#include <tr1/memory>

#include <optixu/optixpp_namespace.h>

namespace legion
{

class ISurfaceShader;
class Matrix4x4;

class Mesh : public APIBase
{
public:

    //
    // External interface -- will be wrapped via Pimpl
    //

    struct Vertex
    {
        Vertex() {}
        Vertex( const Vector3& p, const Vector3& n, const Vector3& t )
            : position( p ), normal( n ), texcoord( t ) {}
        Vector3 position;
        Vector3 normal;
        Vector3 texcoord;
    };

    Mesh( Context* context, const std::string& name );
    ~Mesh();


    
    void setTransform( const Matrix4x4& transform );
    void setTransform( unsigned num_samples, const Matrix4x4* transform );

    void setVertices( unsigned num_vertices, const Vertex* vertices );

    void setVertices( unsigned num_samples,  const float* times,
                      unsigned num_vertices, const Vertex** vertices );

    void setFaces( unsigned num_faces,
                   const Index3* tris,
                   const ISurfaceShader* shader );

    void setFaces( unsigned num_faces,
                   const Index4* quads,
                   const ISurfaceShader* shader );

    void enableSubdivision();
    void disableSubdivision();

    //
    // Internal interface
    //
    void setVertexBuffer  ( optix::Buffer buffer );
    void setFaceBuffer    ( optix::Buffer buffer );

    bool subdvisionEnabled()const;

    const std::vector<Matrix4x4>& getTransform()const;

    bool verticesChanged()const;
    bool facesChanged()const;

private:

    bool                     m_subdivision_enabled;
    std::vector<Matrix4x4>   m_transform;

    bool                     m_vertices_changed;
    bool                     m_faces_changed;

    optix::Buffer            m_vertices;
    optix::Buffer            m_faces;
};

}
#endif // LEGION_SCENE_MESH_MESH_HPP_ 
