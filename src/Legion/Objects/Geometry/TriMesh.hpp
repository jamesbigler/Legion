
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

/// \file TriMesh.hpp


#ifndef LEGION_OBJECTS_GEOMETRY_TRI_MESH_HPP_
#define LEGION_OBJECTS_GEOMETRY_TRI_MESH_HPP_

#include <Legion/Objects/Geometry/IGeometry.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>

namespace legion
{

class VariableContainer;

class LCLASSAPI TriMesh : public IGeometry
{
public:
    LAPI static IGeometry* create( Context* context, const Parameters& params );

    LAPI TriMesh( Context* context );

    struct Vertex
    {
        Vector3 p;
        Vector3 n;
        Vector2 uv;
    };
    
    struct VertexN
    {
        Vector3 p;
        Vector3 n;
    };

    struct VertexUV
    {
        Vector3 p;
        Vector2 uv;
    };


    LAPI const char* name()const;
    LAPI const char* intersectionFunctionName()const;
    LAPI const char* boundingBoxFunctionName()const;
    LAPI const char* sampleFunctionName()const;
    LAPI const char* pdfFunctionName()const;

    LAPI unsigned    numPrimitives()const;
    LAPI float       area()const;

    LAPI void        setTransform( const Matrix& transform );
    LAPI Matrix      getTransform() const;

    LAPI void        setSurface( ISurface* surface );
    LAPI ISurface*   getSurface()const;

    LAPI void        setTriangles( const std::vector<Vector3>& vertices,
                                   const std::vector<Index3>& triangles );

    LAPI void        setTriangles( const std::vector<VertexUV>& vertices,
                                   const std::vector<Index3>& triangles );

    LAPI void        setTriangles( const std::vector<VertexN>& vertices,
                                   const std::vector<Index3>& triangles );

    LAPI void        setTriangles( const std::vector<Vertex>& vertices,
                                   const std::vector<Index3>& triangles );

    LAPI void setVariables( VariableContainer& container ) const;
private:
    enum VertexType
    {
        P=0,
        PN,
        PUV,
        PNUV
    };
    
    static void loadMeshData( const std::string& filename, TriMesh* mesh );
    
    optix::Buffer    m_triangles;    // uint3
    optix::Buffer    m_vertices;     // float3, Vertex, VertexN, or VertexUV
    VertexType       m_vertex_type;
    float            m_area;
    
    Matrix           m_transform;
    ISurface*        m_surface;
};


}


#endif // LEGION_OBJECTS_GEOMETRY_TRI_MESH_HPP_
