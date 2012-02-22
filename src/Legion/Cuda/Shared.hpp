
#ifndef LEGION_CUDA_SHARED_HPP_
#define LEGION_CUDA_SHARED_HPP_

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <Legion/Cuda/Shared.hpp>

#define HOST_DEVICE __host__ __device__

namespace legion
{

struct Vertex
{
    optix::float3   position;
    optix::float3   normal;
    optix::float3   tex;
};


struct SurfaceInfo 
{
    HOST_DEVICE SurfaceInfo() {} 

    HOST_DEVICE SurfaceInfo( int material_id ) 
        : material_id( material_id )
    { 
        geometric_normal = shading_normal = texcoord = optix::make_float3(0.0f);
    }

    optix::float3   geometric_normal;
    optix::float3   shading_normal;
    optix::float3   texcoord;
    int             material_id;
};


struct RayInfo
{
    optix::float3   origin;
    optix::float3   direction;
    float           tmin;
    float           tmax;
};


}

#endif // LEGION_CUDA_SHARED_HPP_
