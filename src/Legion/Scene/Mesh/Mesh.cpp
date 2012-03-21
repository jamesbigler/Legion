
// Copyright (C) 2011 R. Keith Morley 
// 
// (MIT/X11 License)
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include <Legion/Common/Math/Matrix.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Common/Util/Stream.hpp>
#include <Legion/Core/Context.hpp>
#include <Legion/Renderer/RayTracer.hpp>
#include <Legion/Scene/LightShader/MeshLight.hpp>
#include <Legion/Scene/Mesh/Mesh.hpp>

using namespace legion;

namespace
{
    float getArea( const std::vector<Vector3>& vertices, const Index3& tris )
    {
        const Vector3 p0 = vertices[ tris.x() ];
        const Vector3 p1 = vertices[ tris.y() ];
        const Vector3 p2 = vertices[ tris.z() ];

        return length( cross( p1-p0, p2-p0) ) * 0.5f;
    }
}



Mesh::Mesh( Context* context, const std::string& name )
    : APIBase( context, name ),
      m_subdivision_enabled( false ),
      m_vertices_changed( true ),
      m_faces_changed( true ),
      m_shader( 0u ),
      m_area( 0.0f )
{
    m_vertices = context->createOptixBuffer();
    m_faces    = context->createOptixBuffer();
}


Mesh::~Mesh()
{
    m_vertices->destroy();
    m_faces->destroy();
}
   

void Mesh::setTransform( const Matrix& transform )
{
    m_transform.clear();
    m_transform.push_back( transform );

}


void Mesh::setTransform( unsigned num_samples, const Matrix* transform )
{
    m_transform.assign( transform, transform+num_samples );
}


void Mesh::setVertices( unsigned num_vertices, const Vertex* vertices )
{
    RayTracer::updateVertexBuffer( m_vertices, num_vertices, vertices );
    m_vertex_data.resize( num_vertices );
    for( unsigned i = 0u; i < num_vertices; ++i )
        m_vertex_data[i] = vertices[i].position;

    m_vertices_changed = true;
}


void Mesh::setVertices( unsigned num_samples,  const float* times,
                        unsigned num_vertices, const Vertex** vertices )
{
}


void Mesh::setFaces( unsigned              num_faces,
                     const Index3*         tris,
                     const ISurfaceShader* sshader,
                     MeshLight*            lshader)
{

    m_faces_changed = true;
    m_area          = 0.0f;
    m_face_data.assign( tris, tris+num_faces );

    if( lshader ) 
    {
        lshader->setMesh( this );
        getContext().addLight( lshader ); 
        for( unsigned i = 0; i < num_faces; ++i )
            m_area += getArea( m_vertex_data, tris[i] );
    }

    RayTracer::updateFaceBuffer( m_faces, num_faces, tris, sshader, lshader );

    m_shader = sshader;
}


void Mesh::setFaces( unsigned              num_faces,
                     const Index4*         quads,
                     const ISurfaceShader* shader,
                     MeshLight*            lshader)
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


const std::vector<Matrix>& Mesh::getTransform()const
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


void Mesh::sample( const Vector2& seed,
                   Vector3&       on_light,
                   float&         pdf )
{
    // TODO: need seed for object selection
    // TODO: need to build up CDF over area of tris and sample that
    //       pdf is then 1.0f / m_area;
    const float    oseed     = drand48();
    const unsigned num_faces = m_face_data.size();
    unsigned       idx       = oseed * num_faces;
    idx = std::min( idx, num_faces-1 );

    const float t = seed.x() == 0.0f ? 0.0f : sqrtf( seed.x() );
    const float alpha = 1.0f - t;
    const float beta  = seed.y() * t;
    const float gamma = 1.0f - alpha - beta;

    const Index3  tri = m_face_data[ idx ];
    on_light = alpha * m_vertex_data[ tri.x() ] +
               beta  * m_vertex_data[ tri.y() ] +
               gamma * m_vertex_data[ tri.z() ];
    
    pdf = 1.0f / ( static_cast<float>( num_faces ) * 
                 getArea( m_vertex_data, Index3( tri.x(), tri.y(), tri.z() ) ) );
}
