
#include <Mesh.hpp>


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


void Mesh::setVertices( const Point3*  vertices )
{
}


void Mesh::setNormals(  const Vector3* normals )
{
}


void Mesh::setTextureCoordinates( const Vector2* tex_coords )
{
}


void Mesh::addTris( int num_faces, const TriIndices* tris, const Shader& shader )
{
}


void Mesh::addQuads( int num_faces, const QuadIndices* quads, const Shader& shader )
{
}


void Mesh::setTransform( const Matrix4x4& transform )
{
}
