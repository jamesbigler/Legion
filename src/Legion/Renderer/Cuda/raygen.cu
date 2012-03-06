
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

    optix::Ray ray( ray_info.origin,
                    ray_info.direction, 
                    ray_type,
                    0.0001f,    // TODO: use 0.0f, use iterative intersection
                    ray_info.tmax );

    legion::LocalGeometry prd( -1 );
    prd.texcoord = optix::make_float2( 0.2f );
      rtPrintf( "sending ray %f %f %f, %f %f %f, %f\n",
          ray.origin.x,
          ray.origin.y,
          ray.origin.z,
          ray.direction.x,
          ray.direction.y,
          ray.direction.z,
          ray.tmax
          ); 
    rtTrace( top_object, ray, prd );

    results[ launch_index ] = prd;
}
