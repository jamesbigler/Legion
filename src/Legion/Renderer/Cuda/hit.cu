

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <Legion/Renderer/Cuda/Shared.hpp>

rtDeclareVariable( optix::Ray,            ray,   rtCurrentRay, );
rtDeclareVariable( float,                 t_hit, rtIntersectionDistance, );
rtDeclareVariable( legion::LocalGeometry, prd,   rtPayload, );
rtDeclareVariable( legion::LocalGeometry, lgeom, attribute surface_info, );



RT_PROGRAM void anyHit()
{
    prd.material_id = lgeom.material_id;
    rtTerminateRay();
}

RT_PROGRAM void closestHit()
{
    prd.position        = ray.origin + t_hit * ray.direction;
    prd.position_object = lgeom.position_object;
    prd.texcoord        = lgeom.texcoord;
    prd.material_id     = lgeom.material_id;
    prd.light_id        = lgeom.light_id;

    float3 snorm = rtTransformNormal(RT_OBJECT_TO_WORLD,lgeom.shading_normal);
    prd.shading_normal = optix::normalize( snorm );
    
    float3 gnorm = rtTransformNormal(RT_OBJECT_TO_WORLD,lgeom.geometric_normal);
    prd.geometric_normal= optix::normalize( gnorm );
}
