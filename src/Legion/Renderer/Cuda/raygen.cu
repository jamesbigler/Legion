
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <Legion/Renderer/Cuda/Shared.hpp>


rtDeclareVariable( unsigned, launch_index, rtLaunchIndex, );
rtDeclareVariable( rtObject, top_object, , );
rtDeclareVariable( unsigned, ray_type, , );

rtBuffer<legion::SurfaceInfo, 1> results;
rtBuffer<legion::RayInfo,     1> rays;


RT_PROGRAM void traceRays()
{
    legion::RayInfo ray_info = rays[ launch_index ];

    optix::Ray ray( ray_info.origin,
                    ray_info.direction, 
                    ray_type,
                    ray_info.tmin,
                    ray_info.tmax );

    legion::SurfaceInfo prd( -1 );
    rtTrace( top_object, ray, prd );

    results[ launch_index ] = prd;
}
