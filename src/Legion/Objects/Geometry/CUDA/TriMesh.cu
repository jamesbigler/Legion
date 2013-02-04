
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

#include <Legion/Objects/cuda_common.hpp>
#include <Legion/Objects/Light/CUDA/Light.hpp>

rtDeclareVariable( legion::LocalGeometry, local_geom, attribute local_geom, ); 
rtDeclareVariable( optix::Ray,            ray,   rtCurrentRay, );

rtBuffer<uint3>  triangles;

//-----------------------------------------------------------------------------
//
// Helpers 
//
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//
// triMesh with position vertex data 
//
//-----------------------------------------------------------------------------
rtBuffer<float3>  vertices_p;

RT_PROGRAM void triMeshIntersectP( int prim_idx )
{
    uint3 triangle = triangles[ prim_idx ];

    const float3 p0 = vertices_p[ triangle.x ];
    const float3 p1 = vertices_p[ triangle.y ];
    const float3 p2 = vertices_p[ triangle.z ];

    // Intersect ray with triangle
    float3 n;
    float  t, beta, gamma;
    if( optix::intersect_triangle( ray, p0, p1, p2, n, t, beta, gamma ) )
    {
        if(  rtPotentialIntersection( t ) ) 
        {
            const float3 normal = optix::normalize( n );

            // Fill in a localgeometry
            legion::LocalGeometry lg;
            lg.position         = ray.origin + t*ray.direction;
            lg.geometric_normal = normal;
            lg.shading_normal   = normal;
            lg.texcoord         = make_float2( 0.0f );
            local_geom = lg;

            rtReportIntersection( 0u );
        }
    }
}


RT_PROGRAM void triMeshBoundingBoxP( int prim_idx, float result[6] )
{
    uint3 triangle = triangles[ prim_idx ];

    const float3 p0 = vertices_p[ triangle.x ];
    const float3 p1 = vertices_p[ triangle.y ];
    const float3 p2 = vertices_p[ triangle.z ];

    const float  area = optix::length( optix::cross( p1-p0, p2-p0 ) );

    optix::Aabb* aabb = reinterpret_cast<optix::Aabb*>( result );

    if( area > 0.0f && !isinf(area) )
    {
        aabb->m_min = fminf( fminf( p0, p1), p2 );
        aabb->m_max = fmaxf( fmaxf( p0, p1), p2 );
    } else {
        aabb->invalidate();
    }
}


//-----------------------------------------------------------------------------
//
//
//
//-----------------------------------------------------------------------------

RT_CALLABLE_PROGRAM
legion::LightSample triMeshSample( 
        float2 sample_seed, 
        float3 shading_point, 
        float3 shading_normal )
{
     legion::LightSample sample;
     sample.pdf      = 0.0f;
     sample.distance = 0.0f; 
     sample.w_in     = make_float3( 0.0f ); 
     sample.normal   = make_float3( 0.0f ); 
     return sample;
}


RT_CALLABLE_PROGRAM
float triMeshPDF( float3 w_in, float3 shading_point )
{
    return 0.0f;
}
