
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Core/Exception.hpp>
#include <Legion/Core/Ray.hpp>
#include <Legion/Core/config.hpp>
#include <Legion/RayTracer/Cuda/Shared.hpp>
#include <Legion/RayTracer/RayTracer.hpp>
#include <Legion/Scene/Mesh/Mesh.hpp>


using namespace legion;

#define OPTIX_CATCH_RETHROW                                                    \
    catch ( optix::Exception& e )                                              \
    {                                                                          \
        throw legion::Exception( std::string("OPTIX_EXCEPTION: ")+e.what() );  \
    }                                                                          \
    catch ( std::exception& e )                                                \
    {                                                                          \
        throw legion::Exception( std::string("OPTIX_EXCEPTION: ")+e.what() );  \
    }                                                                          \
    catch (...)                                                                \
    {                                                                          \
        throw legion::Exception( std::string("OPTIX_EXCEPTION: unknown") );    \
    }


#define OPTIX_CATCH_WARN                                                       \
    catch ( optix::Exception& e )                                              \
    {                                                                          \
        LLOG_WARN << "OPTIX_EXCEPTION: " << e.what();                          \
    }                                                                          \
    catch ( std::exception& e )                                                \
    {                                                                          \
        LLOG_WARN << "OPTIX_EXCEPTION: " << e.what();                          \
    }                                                                          \
    catch (...)                                                                \
    {                                                                          \
        LLOG_WARN << "OPTIX_EXCEPTION: Unknown";                               \
    }



RayTracer::RayTracer()
{
    initializeOptixContext();
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
        optix::Buffer vbuffer = m_optix_context->createBuffer(RT_BUFFER_INPUT);
        vbuffer->setFormat( RT_FORMAT_USER );
        vbuffer->setElementSize( sizeof( Vertex ) );
        mesh->setVertexBuffer( vbuffer );

        optix::Buffer ibuffer = m_optix_context->createBuffer(RT_BUFFER_INPUT);
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

        m_optix_context[ "ray_type" ]->setUint( static_cast<unsigned>( type ) );

        LLOG_INFO << "RayTracer::traceRays(): Compiling OptiX context...";
        m_optix_context->compile();
        LLOG_INFO << "    Finished.";

        LLOG_INFO << "RayTracer::traceRays(): Launching OptiX ...";
        m_optix_context->launch( OPTIX_ENTRY_POINT_INDEX, num_rays );
        LLOG_INFO << "    Finished.";

    }
    OPTIX_CATCH_RETHROW;
}


optix::Buffer RayTracer::getResults()const
{
    return m_result_buffer;
}


optix::Program RayTracer::createProgram( const std::string& cuda_file,
                                         const std::string name )
{
    try
    {
        std::string path = legion::PTX_DIR + "/cuda_compile_ptx_generated_" +
                           cuda_file + ".ptx";

        optix::Program program;
        program = m_optix_context->createProgramFromPTXFile( path, name );

        LLOG_INFO << "Successfully loaded optix program " << name;
        LLOG_INFO << "    from: " << cuda_file;

        return program;
    }
    OPTIX_CATCH_RETHROW;
}


void RayTracer::initializeOptixContext()
{
    // Initialize optix members
    try
    {
        m_optix_context = optix::Context::create();
        
        m_closest_hit     = createProgram( "hit.cu",     "closestHit" );
        m_any_hit         = createProgram( "hit.cu",     "anyHit" );
        m_trace_rays      = createProgram( "raygen.cu",  "traceRays" );
        m_pmesh_intersect = createProgram( "trimesh.cu", "polyMeshIntersect" );
        m_pmesh_bounds    = createProgram( "trimesh.cu", "polyMeshBounds" );

        m_optix_context->setRayTypeCount( 2u );
        m_optix_context->setEntryPointCount( 1u );
        m_optix_context->setRayGenerationProgram( OPTIX_ENTRY_POINT_INDEX,
                                                  m_trace_rays );

        m_ray_buffer = m_optix_context->createBuffer( RT_BUFFER_INPUT );
        m_ray_buffer->setFormat( RT_FORMAT_USER );
        m_ray_buffer->setElementSize( sizeof( Ray ) );
        m_optix_context[ "rays" ]->set( m_ray_buffer );

        m_result_buffer = m_optix_context->createBuffer( RT_BUFFER_INPUT );
        m_result_buffer->setFormat( RT_FORMAT_USER );
        m_result_buffer->setElementSize( sizeof( SurfaceInfo ) );
        m_optix_context[ "results" ]->set( m_result_buffer );

        m_top_object = m_optix_context->createGroup();
        m_optix_context[ "top_object" ]->set( m_top_object );
    }
    OPTIX_CATCH_RETHROW;
}
