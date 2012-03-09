
#ifndef LEGION_CUDA_SHARED_HPP_
#define LEGION_CUDA_SHARED_HPP_

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#define HOST_DEVICE __host__ __device__

namespace legion
{

/// Must match layout and size of legion::Mesh::Vertex
struct Vertex
{
    optix::float3   position;
    optix::float3   normal;
    optix::float2   tex;
};


struct LocalGeometry 
{
    HOST_DEVICE LocalGeometry() {} 

    HOST_DEVICE LocalGeometry( int material_id ) 
        : material_id( material_id ) 
    {
#if 1
        position = position_object 
                 = geometric_normal 
                 = shading_normal 
                 = optix::make_float3( -1.0f );
        texcoord = optix::make_float2( -1.0f );
        padding  = ~0;
#endif
    }

    bool isValidHit()const { return material_id != -1; }

    optix::float3   position;
    optix::float3   position_object;
    optix::float3   geometric_normal;
    optix::float3   shading_normal;
    optix::float2   texcoord;
    int             material_id;
    unsigned        padding;
};


struct RayInfo
{
    optix::float3   origin;
    optix::float3   direction;
    float           tmax;
    float           time;
};


}

#endif // LEGION_CUDA_SHARED_HPP_
