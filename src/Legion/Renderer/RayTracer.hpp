
#ifndef LEGION_RAYTRACER_RAYTRACER_HPP_
#define LEGION_RAYTRACER_RAYTRACER_HPP_

#include <optixu/optixpp_namespace.h>
#include <Legion/Core/Vector.hpp>
#include <Legion/Scene/Mesh/Mesh.hpp>

namespace legion
{

class Ray;
class SurfaceInfo;
class ISurfaceShader;


class RayTracer
{
public:
    enum RayType
    {
        ANY_HIT=0,
        CLOSEST_HIT
    };

    RayTracer();

    optix::Buffer createBuffer();

    void addMesh( legion::Mesh* mesh );
    void removeMesh( legion::Mesh* mesh );

    optix::Buffer getRayBuffer();

    void traceRays( RayType type );
    void traceRaysNonBlocking( RayType type );

    optix::Buffer getResults()const;

    static void updateVertexBuffer( optix::Buffer buffer,
                                    unsigned num_verts,
                                    const Mesh::Vertex* verts );

    static void updateFaceBuffer( optix::Buffer buffer,
                                  unsigned num_tris,
                                  const Index3* tris,
                                  const ISurfaceShader* shader);


private:
    typedef std::vector< std::pair<Mesh*, optix::GeometryGroup> > MeshList;
    optix::Program createProgram( const std::string& cuda_file,
                                  const std::string name );

    void initializeOptixContext();
      

    static const int     OPTIX_ENTRY_POINT_INDEX = 0u;

    optix::Context       m_optix_context;

    MeshList             m_meshes;

    optix::Buffer        m_ray_buffer;;
    optix::Buffer        m_result_buffer;

    optix::Program       m_closest_hit;
    optix::Program       m_any_hit;
    optix::Program       m_trace_rays;
    optix::Program       m_pmesh_intersect;
    optix::Program       m_pmesh_bounds;

    optix::Group         m_top_object;
    optix::Material      m_material;
};

}


#endif // LEGION_RAYTRACER_RAYTRACER_HPP_
