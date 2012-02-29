

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <Legion/Renderer/Cuda/Shared.hpp>

rtDeclareVariable( optix::Ray,          ray,   rtCurrentRay, );
rtDeclareVariable( float,               t_hit, rtIntersectionDistance, );
rtDeclareVariable( legion::SurfaceInfo, prd,   rtPayload, );
rtDeclareVariable( legion::SurfaceInfo, sinfo, attribute surface_info, );



RT_PROGRAM void anyHit()
{
    //prd = surface_info;
    prd.material_id = 0;
    rtTerminateRay();
}

RT_PROGRAM void closestHit()
{
    prd.position        = ray.origin + t_hit * ray.direction;
    prd.position_object = sinfo.position_object;
    prd.texcoord        = sinfo.texcoord;
    prd.material_id     = sinfo.material_id;

    float3 snorm = rtTransformNormal(RT_OBJECT_TO_WORLD,sinfo.shading_normal);
    prd.shading_normal = optix::normalize( snorm );
    
    float3 gnorm = rtTransformNormal(RT_OBJECT_TO_WORLD,sinfo.geometric_normal);
    prd.geometric_normal= optix::normalize( gnorm );
}
