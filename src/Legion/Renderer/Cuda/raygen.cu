
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





    /*
    legion::SurfaceInfo si;
    si.material_id = -1;
    si.position        = optix::make_float3( 1 );
    si.position_object = optix::make_float3( 2 );
    si.geometric_normal= optix::make_float3( 3 );
    si.shading_normal  = optix::make_float3( 4 );
    si.texcoord        = optix::make_float2( 5 );
    results[ launch_index ] = si; 
    */
}
