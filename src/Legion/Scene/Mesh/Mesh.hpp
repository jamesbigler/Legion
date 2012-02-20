
#ifndef LEGION_CORE_MESH_HPP_
#define LEGION_CORE_MESH_HPP_

#include <Legion/Core/APIBase.hpp>
#include <Legion/Core/Vector.hpp>
#include <tr1/memory>

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
    Mesh( Context* context, const std::string& name, unsigned vertex_count );
    ~Mesh();

    void setTime( float time );

    // TODO: add templated iterator setters???
    void setVertices( const Vector3* vertices );
    void setNormals( const Vector3* normals );
    void setTextureCoordinates( const Vector2* tex_coords );
    void setTransform( const Matrix4x4& transform );

    void addTriangles( unsigned num_faces,
                       const Index3* tris,
                       const ISurfaceShader* shader );

    void addQuads( unsigned num_faces,
                   const Index4* quads,
                   const ISurfaceShader* shader );

    void enableSubdivision();
    void disableSubdivision();

    //
    // Internal interface
    //

private:

};

}
#endif // LEGION_SCENE_MESH_MESH_HPP_ 
