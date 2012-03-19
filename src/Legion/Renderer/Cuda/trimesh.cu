
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <Legion/Renderer/Cuda/Shared.hpp>


// TODO: interleave all time-vertex sets in single buffer and add buffer of
// times.  if times.size() = 3 then vertices will have 3xnumverts size and be
// formatted as
// [ v0_time0_vertex, v0_time1_vertex, v0_time2_vertex, v1_time0_vertex ... ]

rtBuffer<legion::Vertex> vertices;     
rtBuffer<optix::int4>    triangles;

rtDeclareVariable( legion::LocalGeometry, lgeom,    attribute surface_info, );
rtDeclareVariable( optix::float3,         on_plane, attribute on_plane, );
rtDeclareVariable( optix::Ray,            ray,   rtCurrentRay, );

RT_PROGRAM void polyMeshIntersect( int prim_index )
{
  const optix::int4 triangle = triangles[ prim_index ];

  const legion::Vertex v0 = vertices[ triangle.x ];
  const legion::Vertex v1 = vertices[ triangle.y ];
  const legion::Vertex v2 = vertices[ triangle.z ];

  // Intersect ray with triangle
  optix::float3 geometric_normal;
  float t, beta, gamma;
  if( intersect_triangle( ray, v0.position, v1.position, v2.position,
                          geometric_normal, t, beta, gamma ) )
  {
      if(  rtPotentialIntersection( t ) )
      {
          const float alpha = 1.0f - beta - gamma;
          legion::LocalGeometry lg;
          lg.position_object  = ray.origin + t*ray.direction,
          lg.geometric_normal = geometric_normal; 
          lg.shading_normal   = alpha*v0.normal +
                                beta*v1.normal  +
                                gamma*v2.normal;
          lg.texcoord         = alpha*v0.tex    +
                                beta*v1.tex     +
                                gamma*v2.tex;
          lg.material_id      = triangle.w & 0x0000FFFF;
          lg.light_id         = triangle.w >> 16;
          lgeom = lg;

          on_plane = v0.position;

          /*
          refine_and_offset_hitpoint( lg.object_p,
                                      ray.direction, 
                                      lg.geometric_normal,
                                      v0.position,
                                      lg.front_p,
                                      lg.back_p );
                                      */
          rtReportIntersection( 0 );
      }
  }
}


RT_PROGRAM void polyMeshBounds( int prim_index, float bbox[6] )

{  
    const optix::int4   triangle = triangles[ prim_index ];
    const optix::float3 v0       = vertices[ triangle.x ].position;
    const optix::float3 v1       = vertices[ triangle.y ].position;
    const optix::float3 v2       = vertices[ triangle.z ].position;

    const float  area = optix::length( optix::cross( v1-v0, v2-v0 ) );
    optix::Aabb* aabb = (optix::Aabb*)bbox;

    if( area > 0.0f && !isinf( area ) )
    {
        aabb->m_min = fminf( fminf( v0, v1), v2 );
        aabb->m_max = fmaxf( fmaxf( v0, v1), v2 );
    }
    else 
    {
        aabb->invalidate();
    }
}

