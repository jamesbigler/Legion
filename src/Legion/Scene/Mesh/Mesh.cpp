
#include <Legion/Core/Matrix.hpp>
#include <Legion/Scene/Mesh/Mesh.hpp>
#include <Legion/Common/Util/Optix.hpp>

using namespace legion;


Mesh::Mesh( Context* context, const std::string& name )
    : APIBase( context, name ),
      m_subdivision_enabled( false ),
      m_vertices_changed( true ),
      m_faces_changed( true )
{
}


Mesh::~Mesh()
{
}
   

void Mesh::setTransform( const Matrix4x4& transform )
{
}


void Mesh::setTransform( unsigned num_samples, const Matrix4x4* transform )
{
}


void Mesh::setVertices( unsigned num_vertices, const Vertex* vertices )
{
}


void Mesh::setVertices( unsigned num_samples,  const float* times,
                        unsigned num_vertices, const Vertex** vertices )
{
}


void Mesh::setFaces( unsigned num_faces,
                     const Index3* tris,
                     const ISurfaceShader* shader )
{
}


void Mesh::setFaces( unsigned num_faces,
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


void Mesh::setVertexBuffer( optix::Buffer buffer )
{
}


void Mesh::setTriangleBuffer( optix::Buffer buffer )
{
}


void Mesh::setQuadBuffer( optix::Buffer buffer )
{
}


bool Mesh::subdvisionEnabled()const
{
}


const std::vector<Matrix4x4>& Mesh::getTransform()const
{
}


bool Mesh::verticesChanged()const
{
}


bool Mesh::facesChanged()const
{
}
