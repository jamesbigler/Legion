
#include <Legion/Scene/Mesh/Mesh.hpp>

using namespace legion;


class Mesh::Impl
{
public:
    void setTime( float time );

    void setVertices( const Vector3* vertices );
    void setNormals( const Vector3* normals );
    void setTextureCoordinates( const Vector2* tex_coords );
    void setTransform( const Matrix4x4& transform );

    void addTriangles( unsigned num_faces, const Index3* tris,  const ISurfaceShader* shader );
    void addQuads    ( unsigned num_faces, const Index4* quads, const ISurfaceShader* shader );

    void enableSubdivision();
    void disableSubdivision();
private:
};

void Mesh::Impl::setTime( float time )
{
}


void Mesh::Impl::setVertices( const Vector3* vertices )
{
}


void Mesh::Impl::setNormals(  const Vector3* normals )
{
}


void Mesh::Impl::setTransform( const Matrix4x4& transform )
{
}


void Mesh::Impl::setTextureCoordinates( const Vector2* tex_coords )
{
}


void Mesh::Impl::addTriangles( unsigned num_faces, const Index3* triangles, const ISurfaceShader* shader )
{
}


void Mesh::Impl::addQuads( unsigned num_faces, const Index4* quads, const ISurfaceShader* shader )
{
}


void Mesh::Impl::enableSubdivision()
{
}


void Mesh::Impl::disableSubdivision()
{
}



//------------------------------------------------------------------------------
//
// Mesh class facade
//
//------------------------------------------------------------------------------
Mesh::Mesh( const std::string& name, unsigned vertex_count )
    : APIBase( name )
{
}


Mesh::~Mesh()
{
}


void Mesh::setTime( float time )
{
    m_impl->setTime( time );
}


void Mesh::setVertices( const Vector3* vertices )
{
    m_impl->setVertices( vertices );
}


void Mesh::setNormals(  const Vector3* normals )
{
    m_impl->setNormals( normals );
}


void Mesh::setTransform( const Matrix4x4& transform )
{
    m_impl->setTransform( transform );
}


void Mesh::setTextureCoordinates( const Vector2* tex_coords )
{
    m_impl->setTextureCoordinates( tex_coords );
}


void Mesh::addTriangles( unsigned num_faces, const Index3* triangles, const ISurfaceShader* shader )
{
    m_impl->addTriangles( num_faces, triangles, shader  );
}


void Mesh::addQuads( unsigned num_faces, const Index4* quads, const ISurfaceShader* shader )
{
    m_impl->addQuads( num_faces, quads, shader  );
}

    
void Mesh::enableSubdivision()
{
    m_impl->enableSubdivision();
}


void Mesh::disableSubdivision()
{
    m_impl->disableSubdivision();
}

