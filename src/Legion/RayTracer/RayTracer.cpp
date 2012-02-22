
#include <Legion/RayTracer/RayTracer.hpp>
#include <Legion/Core/config.hpp>


using namespace legion;


RayTracer::RayTracer()
{
  m_optix.setProgramSearchPath( legion::PTX_DIR );

  std::string pre = "cuda_compile_ptx_generated_";
  m_optix.registerProgram( pre+"hit_programs.cu.ptx",   "closestHit"        );
  m_optix.registerProgram( pre+"hit_programs.cu.ptx",   "anyHit"            );
  m_optix.registerProgram( pre+"ray_generation.cu.ptx", "traceRays"         );
  m_optix.registerProgram( pre+"triangle_mesh.cu.ptx",  "polyMeshIntersect" );
  m_optix.registerProgram( pre+"triangle_mesh.cu.ptx",  "polyMeshBounds"    );

}

void RayTracer::updateVertexBuffer( optix::Buffer buffer,
                                    unsigned num_verts,
                                    const Vertex* verts )
{
}


void RayTracer::updateFaceBuffer( optix::Buffer buffer,
                                  unsigned num_tris,
                                  const legion::Index3* tris )
{
}


void RayTracer::addMesh( legion::Mesh* mesh )
{
}


void RayTracer::removeMesh( legion::Mesh* mesh )
{
}


void RayTracer::traceRays( QueryType type,
                           unsigned num_rays,
                           legion::Ray* rays )
{
}


legion::SurfaceInfo* RayTracer::getResults()const
{
}

