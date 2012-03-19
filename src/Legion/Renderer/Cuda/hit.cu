

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <Legion/Renderer/Cuda/Shared.hpp>
#include <Legion/Renderer/Cuda/intersection_refinement.hpp>

rtDeclareVariable( optix::Ray,            ray,   rtCurrentRay, );
rtDeclareVariable( float,                 t_hit, rtIntersectionDistance, );
rtDeclareVariable( legion::LocalGeometry, prd,   rtPayload, );
rtDeclareVariable( legion::LocalGeometry, lgeom, attribute surface_info, );
rtDeclareVariable( optix::float3,         on_plane, attribute on_plane, );



RT_PROGRAM void anyHit()
{
    prd.material_id = lgeom.material_id;
    rtTerminateRay();
}

RT_PROGRAM void closestHit()
{
    prd.position_object = lgeom.position_object;
    prd.texcoord        = lgeom.texcoord;
    prd.material_id     = lgeom.material_id;
    prd.light_id        = lgeom.light_id;

    float3 snorm = rtTransformNormal(RT_OBJECT_TO_WORLD,lgeom.shading_normal);
    prd.shading_normal = optix::normalize( snorm );
    
    float3 gnorm = rtTransformNormal(RT_OBJECT_TO_WORLD,lgeom.geometric_normal);
    prd.geometric_normal= optix::normalize( gnorm );
    
    float3 on_plane_world = rtTransformPoint(RT_OBJECT_TO_WORLD, on_plane);
    
    // Refine hitpoint
    float3 back_hit  = optix::make_float3( 0.0f );
    float3 front_hit = optix::make_float3( 0.0f );
    refine_and_offset_hitpoint( ray.origin + t_hit*ray.direction,
                                ray.direction,
                                prd.geometric_normal,
                                on_plane_world,
                                back_hit,
                                front_hit );

    prd.position = front_hit;
    //prd.position = ray.origin + t_hit*ray.direction;
}
