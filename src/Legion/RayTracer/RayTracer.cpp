
#include <Legion/RayTracer/RayTracer.hpp>
#include <Legion/Core/config.hpp>
#include <Legion/Core/Ray.hpp>
#include <Legion/Core/Exception.hpp>
#include <Legion/Cuda/Shared.hpp>
#include <Legion/Scene/Mesh/Mesh.hpp>


using namespace legion;

//
//
//
//
//  TODO: looks like Optix class can be folded into this one
//
//
//

RayTracer::RayTracer()
{
    m_optix.setProgramSearchPath( legion::PTX_DIR );

    std::string pre = "cuda_compile_ptx_generated_";
    m_optix.registerProgram( pre+"hit_programs.cu.ptx",   "closestHit"        );
    m_optix.registerProgram( pre+"hit_programs.cu.ptx",   "anyHit"            );
    m_optix.registerProgram( pre+"ray_generation.cu.ptx", "traceRays"         );
    m_optix.registerProgram( pre+"triangle_mesh.cu.ptx",  "polyMeshIntersect" );
    m_optix.registerProgram( pre+"triangle_mesh.cu.ptx",  "polyMeshBounds"    );

    try
    {
        m_ray_buffer = m_optix.getContext()->createBuffer( RT_BUFFER_INPUT );
        m_ray_buffer->setFormat( RT_FORMAT_USER );
        m_ray_buffer->setElementSize( sizeof( Ray ) );

        m_ray_buffer = m_optix.getContext()->createBuffer( RT_BUFFER_INPUT );
        m_ray_buffer->setFormat( RT_FORMAT_USER );
        m_ray_buffer->setElementSize( sizeof( SurfaceInfo ) );
    }
    OPTIX_CATCH_RETHROW;

}

void RayTracer::updateVertexBuffer( optix::Buffer buffer,
                                    unsigned num_verts,
                                    const Vertex* verts )
{
    try
    {
        buffer->setSize( num_verts );
        Vertex* buffer_data = static_cast<Vertex*>( buffer->map() );
        memcpy( buffer_data, verts, num_verts*sizeof( Vertex ) );
        buffer->unmap();
    }
    OPTIX_CATCH_RETHROW;
}


void RayTracer::updateFaceBuffer( optix::Buffer buffer,
                                  unsigned num_faces,
                                  const Index3* tris )
{
    try
    {
        buffer->setSize( num_faces  );
        buffer->setFormat( RT_FORMAT_UNSIGNED_INT3 );
        Index3* buffer_data = static_cast<Index3*>( buffer->map() );
        memcpy( buffer_data, tris, num_faces*sizeof( Index3 ) );
        buffer->unmap();
    }
    OPTIX_CATCH_RETHROW;
}


void RayTracer::addMesh( legion::Mesh* mesh )
{
    m_meshes.push_back( mesh );

    try
    {
        optix::Context optix_context = m_optix.getContext();
        optix::Buffer vbuffer = optix_context->createBuffer( RT_BUFFER_INPUT );
        vbuffer->setFormat( RT_FORMAT_USER );
        vbuffer->setElementSize( sizeof( Vertex ) );
        mesh->setVertexBuffer( vbuffer );

        optix::Buffer ibuffer = optix_context->createBuffer( RT_BUFFER_INPUT );
        ibuffer->setFormat( RT_FORMAT_USER );
        mesh->setFaceBuffer( ibuffer );
    }
    OPTIX_CATCH_RETHROW;
}


void RayTracer::removeMesh( legion::Mesh* mesh )
{
    try
    {
    }
    OPTIX_CATCH_RETHROW;
}


void RayTracer::traceRays( QueryType type,
                           unsigned num_rays,
                           Ray* rays )
{
    try
    {
        m_ray_buffer->setSize( num_rays );
        Ray* buffer_data = static_cast<Ray*>( m_ray_buffer->map() );
        memcpy( buffer_data, rays, num_rays*sizeof( Ray ) );
        m_ray_buffer->unmap();
    }
    OPTIX_CATCH_RETHROW;
}


legion::SurfaceInfo* RayTracer::getResults()const
{
}

