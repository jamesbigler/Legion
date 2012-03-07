
#include <Legion/Core/Matrix.hpp>
#include <Legion/Core/Context.hpp>
#include <Legion/Scene/Mesh/Mesh.hpp>
#include <Legion/Renderer/RayTracer.hpp>

using namespace legion;


Mesh::Mesh( Context* context, const std::string& name )
    : APIBase( context, name ),
      m_subdivision_enabled( false ),
      m_vertices_changed( true ),
      m_faces_changed( true ),
      m_shader( 0u )
{
    m_vertices = context->createOptixBuffer();
    m_faces    = context->createOptixBuffer();
}


Mesh::~Mesh()
{
    m_vertices->destroy();
    m_faces->destroy();
}
   

void Mesh::setTransform( const Matrix4x4& transform )
{
    m_transform.clear();
    m_transform.push_back( transform );

}


void Mesh::setTransform( unsigned num_samples, const Matrix4x4* transform )
{
    m_transform.assign( transform, transform+num_samples );
}


void Mesh::setVertices( unsigned num_vertices, const Vertex* vertices )
{
    RayTracer::updateVertexBuffer( m_vertices, num_vertices, vertices );
    m_vertices_changed = true;
}


void Mesh::setVertices( unsigned num_samples,  const float* times,
                        unsigned num_vertices, const Vertex** vertices )
{
}


void Mesh::setFaces( unsigned num_faces,
                     const Index3* tris,
                     const ISurfaceShader* shader )
{
    RayTracer::updateFaceBuffer( m_faces, num_faces, tris, shader );
    m_faces_changed = true;
    m_shader = shader;
}


void Mesh::setFaces( unsigned num_faces,
                     const Index4* quads,
                     const ISurfaceShader* shader )
{
}


optix::Buffer Mesh::getVertexBuffer()
{
    return m_vertices;
}


optix::Buffer Mesh::getFaceBuffer()
{
    return m_faces;
}


unsigned Mesh::getVertexCount()
{
    RTsize count;
    m_vertices->getSize( count );
    return count;
}


unsigned Mesh::getFaceCount()
{
    RTsize count;
    m_faces->getSize( count );
    return count;
}


void Mesh::enableSubdivision()
{
    m_subdivision_enabled = true;
}


void Mesh::disableSubdivision()
{
    m_subdivision_enabled = false;
}



bool Mesh::subdvisionEnabled()const
{
    return m_subdivision_enabled;
}

    
const ISurfaceShader* Mesh::getShader()const
{
    return m_shader;
}


const std::vector<Matrix4x4>& Mesh::getTransform()const
{
    return m_transform;
}


bool Mesh::verticesChanged()const
{
    return m_vertices_changed;
}


bool Mesh::facesChanged()const
{
    return m_faces_changed;
}

void Mesh::acceptChanges()
{
    m_vertices_changed = m_faces_changed = false;
}
