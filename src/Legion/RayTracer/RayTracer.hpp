
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

    optix::Buffer  getResults()const;

private:
    optix::Program createProgram( const std::string& cuda_file,
                                  const std::string name );

    void initializeOptixContext();
      

    static const int     OPTIX_ENTRY_POINT_INDEX = 0u;

    optix::Context       m_optix_context;

    std::vector<Mesh*>   m_meshes;

    optix::Buffer        m_ray_buffer;;
    optix::Buffer        m_result_buffer;

    optix::Program       m_closest_hit;
    optix::Program       m_any_hit;
    optix::Program       m_trace_rays;
    optix::Program       m_pmesh_intersect;
    optix::Program       m_pmesh_bounds;

    optix::Group         m_top_object;
};

}


#endif // LEGION_RAYTRACER_RAYTRACER_HPP_

