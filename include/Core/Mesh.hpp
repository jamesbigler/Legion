
#ifndef LEGION_MESH_H_
#define LEGION_MESH_H_

#include <private/APIBase.h>

namespace legion
{

class Shader;

class Mesh : public APIBase
{
public:
    enum Type
    {
        TYPE_TRIANGLE = 0,
        TYPE_CATMULL_CLARK,
        TYPE_BUILTIN_COUNT
    };

    explicit Mesh( const std::string& name, Type type, unsigned vertex_count );
    ~Mesh();

    void setTime( float time );
    void setVertices( const Point3*  vertices );
    void setNormals(  const Vector3* normals );
    void setTextureCoordinates( const Vector2* tex_coords );

    void addTris( int num_faces, const TriIndices* tris, const Shader& shader );
    void addQuads( int num_faces, const QuadIndices* quads, const Shader& shader );

    void setTransform( const Matrix4x4& transform );

private:

    class Impl;
    std::shared_ptr<Impl> m_impl;
};

}
#endif // LEGION_MESH_H_
