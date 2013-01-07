
#ifndef LEGION_CORE_MESH_HPP_
#define LEGION_CORE_MESH_HPP_


#include <Legion/Core/APIBase.hpp>
#include <Legion/Common/Math/Vector.hpp>
#include <tr1/memory>

#include <optixu/optixpp_namespace.h>

namespace legion
{

// TODO: need to work out ownership of mesh's surfaceshader and MeshLightShader
class MeshLight;
class ISurface;
class Matrix;
struct LocalGeometry;

class Mesh : public APIBase
{
public:

    //
    // External interface -- will be wrapped via Pimpl
    //

    struct Vertex
    {
        Vertex() {}
        Vertex( const Vector3& p, const Vector3& n, const Vector2& t )
            : position( p ), normal( n ), texcoord( t ) {}

        Vector3 position;
        Vector3 normal;
        Vector2 texcoord;
    };

    Mesh( Context* context, const std::string& name );
    ~Mesh();


    
    void setTransform( const Matrix& transform );
    void setTransform( unsigned num_samples, const Matrix* transform );

    void setVertices( unsigned num_vertices, const Vertex* vertices );

    void setVertices( unsigned num_samples,  const float* times,
                      unsigned num_vertices, const Vertex** vertices );

    void setFaces( unsigned              num_faces,
                   const Index3*         tris,
                   const ISurface* sshader,
                   MeshLight*            lshader = 0u );

    void setFaces( unsigned              num_faces,
                   const Index4*         quads,
                   const ISurface* sshader,
                   MeshLight*            lshader = 0u );

    unsigned  getVertexCount();
    unsigned  getFaceCount();

    void enableSubdivision();
    void disableSubdivision();

    //
    // Internal interface
    //
    optix::Buffer getVertexBuffer();
    optix::Buffer getFaceBuffer();

    bool subdvisionEnabled()const;

    const ISurface* getShader()const;
  
    const std::vector<Matrix>& getTransform()const;

    bool verticesChanged()const;
    bool facesChanged()const;
    void acceptChanges();

    void sample( const Vector2&       seed,
                 const LocalGeometry& p,
                 Vector3&             on_light,
                 float&               pdf );

    float getArea()const;

private:

    bool                     m_subdivision_enabled;

    std::vector<Matrix>      m_transform;

    optix::Buffer            m_vertices;
    optix::Buffer            m_faces;
    bool                     m_vertices_changed;
    bool                     m_faces_changed;

    const ISurface*    m_shader;

    float                    m_area;
    std::vector<float>       m_area_cdf;

    std::vector<Vector3>     m_vertex_data;
    std::vector<Index3>      m_face_data;

};

}
#endif // LEGION_OBJECTS_MESH_MESH_HPP_ 