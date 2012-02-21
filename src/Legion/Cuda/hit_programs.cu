

#include <optix.h>
#include <Legion/Cuda/Shared.hpp>

rtDeclareVariable( legion::SurfaceInfo, prd,          rtPayload, );
rtDeclareVariable( legion::SurfaceInfo, surface_info, attribute surface_info, );

RT_PROGRAM void closestHit()
{
    prd = surface_info;
}
