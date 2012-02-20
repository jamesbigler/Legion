
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <Legion/Cuda/Shared.hpp>


rtDeclareVariable( unsigned, launch_index, rtLaunchIndex, );
rtDeclareVariable( rtObject, top_object, , );

rtBuffer<legion::SurfaceInfo, 1> results;
rtBuffer<legion::RayInfo,     1> rays;

const unsigned RAY_TYPE=0u;


RT_PROGRAM void traceRays()
{
    legion::RayInfo ray_info = rays[ launch_index ];

    optix::Ray ray( ray_info.origin,
                    ray_info.direction, 
                    RAY_TYPE,
                    ray_info.tmin,
                    ray_info.tmax );

    legion::SurfaceInfo prd( -1 );
    rtTrace( top_object, ray, prd );

    results[ launch_index ] = prd;
}
