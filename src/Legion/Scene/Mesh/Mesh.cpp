
#include <Legion/Scene/Mesh/Mesh.hpp>
#include <Legion/Common/Util/Optix.hpp>

using namespace legion;


Mesh::Mesh( Context* context, const std::string& name, unsigned vertex_count )
    : APIBase( context, name )
{
}


Mesh::~Mesh()
{
}


void Mesh::setTime( float time )
{
}


void Mesh::setVertices( const Vector3* vertices )
{



}


void Mesh::setNormals(  const Vector3* normals )
{
}


void Mesh::setTransform( const Matrix4x4& transform )
{
}


void Mesh::setTextureCoordinates( const Vector2* tex_coords )
{
}


void Mesh::addTriangles( unsigned num_faces,
                               const Index3* triangles,
                               const ISurfaceShader* shader )
{
}


void Mesh::addQuads( unsigned num_faces,
                           const Index4* quads,
                           const ISurfaceShader* shader )
{
}


void Mesh::enableSubdivision()
{
}


void Mesh::disableSubdivision()
{
}



