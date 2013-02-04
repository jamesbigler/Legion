
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

/// \file TriMesh.cpp

#include <Legion/Objects/Geometry/TriMesh.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Common/Util/Stream.hpp>
#include <algorithm>


using namespace legion;













#include <iostream>
    










TriMesh::TriMesh( Context* context )
    : IGeometry( context ),
      m_vertex_type( P ),
      m_area( 0.0f ),
      m_transform( Matrix::identity() ),
      m_surface( 0 )
{
}


const char* TriMesh::name()const
{
    return "TriMesh";
}


const char* TriMesh::intersectionFunctionName()const
{
    switch( m_vertex_type )
    {
        case P:    return "triMeshIntersectP";
        case PN:   return "triMeshIntersectPN";
        case PUV:  return "triMeshIntersectPUV";
        case PNUV: return "triMeshIntersectPNUV";
    };
}


const char* TriMesh::boundingBoxFunctionName()const
{
    switch( m_vertex_type )
    {
        case P:    return "triMeshBoundingBoxP";
        case PN:   return "triMeshBoundingBoxPN";
        case PUV:  return "triMeshBoundingBoxPUV";
        case PNUV: return "triMeshBoundingBoxPNUV";
    };
}


const char* TriMesh::sampleFunctionName()const
{
    return "triMeshSample";
}

const char* TriMesh::pdfFunctionName()const
{
    return "triMeshPDF";
}

unsigned TriMesh::numPrimitives()const
{
    RTsize triangle_count;
    m_triangles->getSize( triangle_count );
    return triangle_count;
}


float TriMesh::area()const
{
    LEGION_TODO();
    return m_area;
}


void TriMesh::setTransform( const Matrix& transform )
{
    m_transform = transform;
}


Matrix TriMesh::getTransform() const
{
    return m_transform;
}


void TriMesh::setSurface( ISurface* surface )
{
    m_surface = surface;
}


ISurface* TriMesh::getSurface()const
{
    return m_surface;
}





















void TriMesh::setTriangles( const std::vector<Vector3>& vertices,
                            const std::vector<Index3>& triangles )
{
    m_vertex_type = P;

    // Free old data
    if( m_vertices  ) m_vertices->destroy();
    if( m_triangles ) m_triangles->destroy();

    m_triangles = createOptiXBuffer( RT_BUFFER_INPUT,
                                     RT_FORMAT_UNSIGNED_INT3,
                                     triangles.size() );

    m_vertices  = createOptiXBuffer( RT_BUFFER_INPUT,
                                     RT_FORMAT_FLOAT3,
                                     vertices.size() );

    std::copy( triangles.begin(),
               triangles.end(),
               reinterpret_cast<Index3*>( m_triangles->map() ) );
    m_triangles->unmap();

    std::copy( vertices.begin(),
               vertices.end(),
               reinterpret_cast<Vector3*>( m_vertices->map() ) );
    m_vertices->unmap();

    /*
    float* f = (float*)m_vertices->map();
    for( unsigned i = 0; i < vertices.size(); ++i )
    {
        std::cerr << "v " << f[i*3+0] << " " << f[i*3+1] << " " << f[i*3+2] 
                  << std::endl;
    }
    m_vertices->unmap();

    unsigned* u = (unsigned*)m_triangles->map();
    for( unsigned i = 0; i < triangles.size(); ++i )
    {
        std::cerr << "f " << u[i*3+0] << " " << u[i*3+1] << " " << u[i*3+2] 
                  << std::endl;
    }
    m_triangles->unmap();
    exit(0);
    */
}


void TriMesh::setTriangles( const std::vector<VertexUV>& /*vertices*/,
                            const std::vector<Index3>& /*triangles*/ )
{
    m_vertex_type = PUV;
    LEGION_TODO();
}


void TriMesh::setTriangles( const std::vector<VertexN>& /*vertices*/,
                            const std::vector<Index3>& /*triangles*/ )
{
    m_vertex_type = PN;
    LEGION_TODO();
}


void TriMesh::setTriangles( const std::vector<Vertex>& /*vertices*/,
                            const std::vector<Index3>& /*triangles*/ )
{
    m_vertex_type = PNUV;
    LEGION_TODO();
}


void TriMesh::setVariables( const VariableContainer& container ) const
{

    switch( m_vertex_type )
    {
        case P:    container.setBuffer( "vertices_p",    m_vertices );
                   break;
        case PN:   container.setBuffer( "vertices_pn",   m_vertices );
                   break;
        case PUV:  container.setBuffer( "vertices_puv",  m_vertices );
                   break;
        case PNUV: container.setBuffer( "vertices_pnuv", m_vertices );
                   break;
    };
    container.setBuffer( "triangles", m_triangles );
}
