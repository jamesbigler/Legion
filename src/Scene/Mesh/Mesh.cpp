
#include <Core/Mesh.hpp>

using namespace legion;

Mesh::Mesh( const std::string& name, Type type, unsigned vertex_count )
    : APIBase( name )
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


void Mesh::addTriangles( unsigned num_triangles, const Index3* triangles, const ISurfaceShader& shader )
{
}


void Mesh::addQuads( unsigned num_quads, const Index4* quads, const ISurfaceShader& shader )
{
}


