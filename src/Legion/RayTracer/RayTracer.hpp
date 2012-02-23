
#ifndef LEGION_RAYTRACER_RAYTRACER_HPP_
#define LEGION_RAYTRACER_RAYTRACER_HPP_

#include <optixu/optixpp_namespace.h>
#include <Legion/RayTracer/Optix.hpp>
#include <Legion/Core/Vector.hpp>

namespace legion
{

class Vertex;
class Mesh;
class Ray;
class SurfaceInfo;


class RayTracer
{
public:
    enum QueryType
    {
        ANY_HIT=0,
        CLOSEST_HIT
    };

    RayTracer();

    void updateVertexBuffer( optix::Buffer buffer,
                             unsigned num_verts,
                             const Vertex* verts );

    void updateFaceBuffer( optix::Buffer buffer,
                           unsigned num_tris,
                           const legion::Index3* tris );


    void addMesh( legion::Mesh* mesh );
    void removeMesh( legion::Mesh* mesh );


    void traceRays( QueryType type, unsigned num_rays, legion::Ray* rays );
    legion::SurfaceInfo* getResults()const;

private:

    Optix   m_optix;

    std::vector<Mesh*>   m_meshes;
    optix::Buffer        m_ray_buffer;;
    optix::Buffer        m_result_buffer;
};

}


#endif // LEGION_RAYTRACER_RAYTRACER_HPP_

