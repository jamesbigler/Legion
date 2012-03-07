
#include <Legion/Common/Util/Assert.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Common/Util/Stream.hpp>
#include <Legion/Core/Exception.hpp>
#include <Legion/Core/Ray.hpp>
#include <Legion/Core/config.hpp>
#include <Legion/Renderer/Cuda/Shared.hpp>
#include <Legion/Renderer/RayTracer.hpp>
#include <Legion/Scene/Mesh/Mesh.hpp>
#include <Legion/Scene/SurfaceShader/ISurfaceShader.hpp>


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



namespace
{

struct OptiXLaunch
{
    OptiXLaunch( optix::Context context,
                 unsigned entry_point_index,
                 unsigned num_rays )
        : m_context( context ),
          m_entry_point_index( entry_point_index ),
          m_num_rays( num_rays )
    {}

    void operator()()
    {
        m_context->launch( m_entry_point_index, m_num_rays );
    }

    optix::Context m_context;
    unsigned       m_entry_point_index;
    unsigned       m_num_rays;
};
}


RayTracer::RayTracer()
{
    initializeOptixContext();
}

optix::Buffer RayTracer::createBuffer()
{
    try
    {
        return m_optix_context->createBuffer(RT_BUFFER_INPUT);
    }
    OPTIX_CATCH_RETHROW;
}

void RayTracer::updateVertexBuffer( optix::Buffer buffer,
                                    unsigned num_verts,
                                    const Mesh::Vertex* verts )
{
    LEGION_STATIC_ASSERT( sizeof( Mesh::Vertex ) == sizeof( Vertex ) );
    try
    {
        buffer->setFormat( RT_FORMAT_USER );
        buffer->setElementSize( sizeof( Vertex ) );
        buffer->setSize( num_verts );

        Vertex* buffer_data = static_cast<Vertex*>( buffer->map() );
        memcpy( buffer_data, verts, num_verts*sizeof( Vertex ) );
        buffer->unmap();
    }
    OPTIX_CATCH_RETHROW;
}


void RayTracer::updateFaceBuffer( optix::Buffer buffer,
                                  unsigned num_faces,
                                  const Index3* tris,
                                  const ISurfaceShader* shader )
{
    try
    {
        buffer->setFormat( RT_FORMAT_UNSIGNED_INT4 );
        buffer->setSize( num_faces  );

        unsigned shader_id = shader->getID();
        Index4* buffer_data = static_cast<Index4*>( buffer->map() );
        for( unsigned i = 0; i < num_faces; ++i )
            buffer_data[i] = Index4( tris[i], shader_id );

        buffer->unmap();
    }
    OPTIX_CATCH_RETHROW;
}


void RayTracer::addMesh( legion::Mesh* mesh )
{
    try
    {
        optix::Geometry geom = m_optix_context->createGeometry();
        geom->setBoundingBoxProgram( m_pmesh_bounds );
        geom->setIntersectionProgram( m_pmesh_intersect );
        geom->setPrimitiveCount( mesh->getFaceCount() );
        geom[ "vertices"  ]->set( mesh->getVertexBuffer() );
        geom[ "triangles" ]->set( mesh->getFaceBuffer() );

        optix::GeometryInstance gi = m_optix_context->createGeometryInstance();
        gi->setGeometry( geom );
        gi->addMaterial( m_material );

        optix::Acceleration accel;
        accel = m_optix_context->createAcceleration( "Bvh", "Bvh" );

        optix::GeometryGroup gg = m_optix_context->createGeometryGroup();
        gg->setAcceleration( accel );
        gg->setChildCount( 1u );
        gg->setChild( 0, gi );

        unsigned num_meshes = m_top_object->getChildCount() + 1;
        m_top_object->setChildCount( num_meshes );
        m_top_object->setChild( num_meshes-1, gg );
        m_top_object->getAcceleration()->markDirty();

        m_meshes.push_back( std::make_pair( mesh, gg ) );
    }
    OPTIX_CATCH_RETHROW;
}


void RayTracer::removeMesh( legion::Mesh* mesh )
{
    LEGION_TODO();

    try
    {
    }
    OPTIX_CATCH_RETHROW;
}

    
void RayTracer::preprocess()
{
    for( MeshList::iterator it = m_meshes.begin();
         it != m_meshes.end();
         ++it )
    {
        Mesh* mesh = it->first;
        if( mesh->verticesChanged() || mesh->facesChanged() )
            it->second->getAcceleration()->markDirty();
        mesh->acceptChanges();
    }

    m_ray_server.preprocess();

}


void RayTracer::trace( RayType type, const std::vector<Ray>& rays )
{
    try
    {
        for( MeshList::iterator it = m_meshes.begin();
             it != m_meshes.end();
             ++it )
        {
            Mesh* mesh = it->first;
            if( mesh->verticesChanged() || mesh->facesChanged() )
                it->second->getAcceleration()->markDirty();
            mesh->acceptChanges();
        }
            
        m_optix_context[ "ray_type" ]->setUint( static_cast<unsigned>( type ) );
        m_optix_context->compile();
        m_ray_server.trace( OPTIX_ENTRY_POINT_INDEX, rays );
    }
    OPTIX_CATCH_RETHROW;
}


const LocalGeometry* RayTracer::getResults()
{
    return m_ray_server.getResults();
}


void RayTracer::join()
{
    return m_ray_server.join();
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
        m_ray_buffer->setSize( 0 );
        m_optix_context[ "rays" ]->set( m_ray_buffer );

        m_result_buffer = m_optix_context->createBuffer( RT_BUFFER_OUTPUT );
        m_result_buffer->setFormat( RT_FORMAT_USER );
        m_result_buffer->setElementSize( sizeof( LocalGeometry ) );
        m_result_buffer->setSize( 0 );
        m_optix_context[ "results" ]->set( m_result_buffer );

        optix::Acceleration accel;
        accel = m_optix_context->createAcceleration( "Bvh", "Bvh" );
        m_top_object = m_optix_context->createGroup();
        m_top_object->setAcceleration( accel );
        m_optix_context[ "top_object" ]->set( m_top_object );

        m_material = m_optix_context->createMaterial();
        m_material->setClosestHitProgram( CLOSEST_HIT, m_closest_hit );
        m_material->setAnyHitProgram    ( ANY_HIT,     m_any_hit );


        m_ray_server.setContext( m_optix_context );
        m_ray_server.setRayBufferName( "rays" );
        m_ray_server.setResultBufferName( "results" );
    }
    OPTIX_CATCH_RETHROW;
}
