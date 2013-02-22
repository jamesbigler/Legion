
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

#include <Legion/Common/Util/Parameters.hpp>
#include <Legion/Common/Util/Stream.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Objects/Geometry/TriMesh.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>


using namespace legion;

namespace legion
{
    std::istream& operator>>( std::istream& in, Vector3& v )
    {
        in >> (v[0]);
        in >> (v[1]) >> (v[2]);
        return in;
    }

    std::istream& operator>>( std::istream& in, Index3& i )
    {
        in >> i[0] >> i[1] >> i[2];
        return in;
    }
    
    std::istream& operator>>( std::istream& in, TriMesh::VertexN& v )
    {
        in >> v.p >> v.n;
        return in;
    }
    
    std::ostream& operator<<( std::ostream& out, TriMesh::VertexN& v )
    {
        out << v.p << " " << v.n;
        return out;
    }
}

IGeometry* TriMesh::create( Context* context, const Parameters& params )
{
    TriMesh* trimesh = new TriMesh( context );
    const std::string datafile = params.get( "datafile",
                                             std::string( "NO_DATAFILE" ) );
    loadMeshData( datafile, trimesh );
    return trimesh;
}


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
        default  : return "triMeshIntersectNUV";
    };
}


const char* TriMesh::boundingBoxFunctionName()const
{
    switch( m_vertex_type )
    {
        case P   : return "triMeshBoundingBoxP";
        case PN  : return "triMeshBoundingBoxPN";
        case PUV : return "triMeshBoundingBoxPUV";
        case PNUV: return "triMeshBoundingBoxPNUV";
        default  : return "triMeshBoundingBoxPNUV";
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
}


void TriMesh::setTriangles( const std::vector<VertexUV>& /*vertices*/,
                            const std::vector<Index3>& /*triangles*/ )
{
    m_vertex_type = PUV;
    LEGION_TODO();
}


void TriMesh::setTriangles( const std::vector<VertexN>& vertices,
                            const std::vector<Index3>&  triangles )
{
    m_vertex_type = PN;

    // Free old data
    if( m_vertices  ) m_vertices->destroy();
    if( m_triangles ) m_triangles->destroy();

    m_triangles = createOptiXBuffer( RT_BUFFER_INPUT,
                                     RT_FORMAT_UNSIGNED_INT3,
                                     triangles.size() );

    m_vertices  = createOptiXBuffer( RT_BUFFER_INPUT,
                                     RT_FORMAT_USER,
                                     vertices.size() );
    m_vertices->setElementSize( sizeof( VertexN ) );

    std::copy( triangles.begin(),
               triangles.end(),
               reinterpret_cast<Index3*>( m_triangles->map() ) );
    m_triangles->unmap();

    std::copy( vertices.begin(),
               vertices.end(),
               reinterpret_cast<VertexN*>( m_vertices->map() ) );
    m_vertices->unmap();
}


void TriMesh::setTriangles( const std::vector<Vertex>& /*vertices*/,
                            const std::vector<Index3>& /*triangles*/ )
{
    m_vertex_type = PNUV;
    LEGION_TODO();
}


void TriMesh::setVariables( VariableContainer& container ) const
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
    

void TriMesh::loadMeshData( const std::string& filename, TriMesh* mesh )
{
    LLOG_INFO << "Reading mesh '" << filename << "'";
    const std::string path = 
        "/Users/keith/Code/Legion/src/Standalone/lr/scenes/quadbot/" + filename;
    std::ifstream in( path.c_str() );
    if( !in )
        throw Exception( "Failed to open file '" + path + "' for reading" );

    std::string token;
    unsigned vertcount;

    in >> token >> vertcount;
    LLOG_INFO << "\t'" << token << "': " << vertcount;
    std::vector<VertexN> vertices( vertcount );

    for( int i = 0; i < vertcount; ++i )
        in >> vertices[i];
    LLOG_INFO << "\tvertices[n-1]: " << vertices[vertcount-1];

    unsigned polycount;
    in >> token >> polycount;
    LLOG_INFO << "\tpolycount: " << polycount;
    std::vector<Index3> triangles;

    for( unsigned i = 0u; i < polycount; ++i )
    {
        unsigned poly_vertcount, v0, v1, v2;
        in >> poly_vertcount >> v0 >> v1 >> v2;
        triangles.push_back( Index3( v0, v1, v2 ) );
        for( unsigned j = 3u; j < poly_vertcount; ++j ) 
        {
            v1 =  v2;
            in >> v2;
            triangles.push_back( Index3( v0, v1, v2 ) );
        }

    }

    LLOG_INFO << "\ttriangles[n-1]: " << triangles.back();
    
    
    mesh->setTriangles( vertices, triangles );
}

