
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <Legion/Renderer/Cuda/Shared.hpp>


rtDeclareVariable( unsigned, launch_index, rtLaunchIndex, );
rtDeclareVariable( rtObject, top_object, , );
rtDeclareVariable( unsigned, ray_type, , );

rtBuffer<legion::LocalGeometry, 1> results;
rtBuffer<legion::RayInfo,       1> rays;


RT_PROGRAM void traceRays()
{
    legion::RayInfo ray_info = rays[ launch_index ];

    legion::LocalGeometry prd( -1 );

    if( ray_info.direction.x != 0.0f || 
        ray_info.direction.y != 0.0f || 
        ray_info.direction.z != 0.0f ) 
    {
        optix::Ray ray( ray_info.origin,
                        ray_info.direction, 
                        ray_type,
                        0.0001f,    // TODO: use 0.0f, iterative intersection
                        ray_info.tmax );

        rtTrace( top_object, ray, prd );
    }

    results[ launch_index ] = prd;
}
