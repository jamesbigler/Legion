
#ifndef LEGION_CUDA_SHARED_HPP_
#define LEGION_CUDA_SHARED_HPP_

#include <Legion/Common/Math/Vector.hpp>

#ifdef __CUDACC__
#    define HOST_DEVICE __host__ __device__
#else
#    define HOST_DEVICE
#endif

namespace legion
{

/// Must match layout and size of legion::Mesh::Vertex
struct Vertex
{
#ifdef __CUDACC__
    typedef optix::float3   VECTOR3;
    typedef optix::float2   VECTOR2;
#else
    typedef legion::Vector3 VECTOR3; 
    typedef legion::Vector2 VECTOR2; 
#endif

    VECTOR3   position;
    VECTOR3   normal;
    VECTOR2   tex;
};


struct LocalGeometry 
{
#ifdef __CUDACC__
    typedef optix::float3   VECTOR3;
    typedef optix::float2   VECTOR2;
#else
    typedef legion::Vector3 VECTOR3; 
    typedef legion::Vector2 VECTOR2; 
#endif
    HOST_DEVICE void reset()
    { material_id = light_id = -1; } 

    bool isValidHit()const { return material_id != -1; }


    VECTOR3   position;
    VECTOR3   position_object;
    VECTOR3   geometric_normal;
    VECTOR3   shading_normal;
    VECTOR2   texcoord;
    int       material_id; // Can be shorts if necessary
    int       light_id;
};


struct RayInfo
{
#ifdef __CUDACC__
    typedef optix::float3   VECTOR3;
    typedef optix::float2   VECTOR2;
#else
    typedef legion::Vector3 VECTOR3; 
    typedef legion::Vector2 VECTOR2; 
#endif

    VECTOR3   origin;
    VECTOR3   direction;
    float     tmax;
    float     time;
};


}

#endif // LEGION_CUDA_SHARED_HPP_
