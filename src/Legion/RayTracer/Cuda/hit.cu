

#include <optix.h>
#include <Legion/RayTracer/Cuda/Shared.hpp>

rtDeclareVariable( legion::SurfaceInfo, prd,          rtPayload, );
rtDeclareVariable( legion::SurfaceInfo, surface_info, attribute surface_info, );

RT_PROGRAM void anyHit()
{
    prd = surface_info;
    rtTerminateRay();
}

RT_PROGRAM void closestHit()
{
    prd = surface_info;
}
