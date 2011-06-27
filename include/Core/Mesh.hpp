
#ifndef LEGION_CORE_MESH_HPP_
#define LEGION_CORE_MESH_HPP_

#include <Private/APIBase.hpp>
#include <Core/Vector.hpp>
#include <Core/Index.hpp>
#include <Core/Matrix.hpp>

namespace legion
{

class ISurfaceShader;

class Mesh : public APIBase
{
public:
    enum Type
    {
        TYPE_POLYGONAL = 0,
        TYPE_CATMULL_CLARK,
        TYPE_BUILTIN_COUNT
    };

    Mesh( const std::string& name, Type type, unsigned vertex_count );
    ~Mesh();

    void setTime( float time );
    void setVertices( const Vector3* vertices );
    void setNormals( const Vector3* normals );
    void setTextureCoordinates( const Vector2* tex_coords );
    void setTransform( const Matrix4x4& transform );

    void addTriangles( unsigned num_faces, const Index3* tris, const ISurfaceShader& shader );
    void addQuads( unsigned num_faces, const Index4* quads, const ISurfaceShader& shader );


private:

    class Impl;
    std::tr1::shared_ptr<Impl> m_impl;
};

}
#endif // LEGION_CORE_MESH_HPP_ 
