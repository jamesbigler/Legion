
#include <Legion/Legion.hpp>
#include <Legion/Core/Vector.hpp>
#include <vector>
#include <iomanip>


void generateSphere( std::vector<legion::Mesh::Vertex>& verts,
                     std::vector<legion::Index3>& indices,
                     unsigned lat_div,
                     unsigned long_div,
                     float radius,
                     const legion::Vector3& offset )
{
    using namespace legion;

    const float PIf = static_cast<float>( M_PI );
    const unsigned num_verts = ( lat_div - 1 ) * (long_div+1) + 2;
    const unsigned num_tris  = ( lat_div - 2 ) * long_div * 2 + 2*long_div;
    LLOG_INFO << "sphere - ntris: " << num_tris << " nverts: " << num_verts;

    // generate vertices
    verts.resize( num_verts );
    unsigned vindex = 0u;
    for( unsigned j = 1u; j < lat_div; ++j )
    {
        const float theta     = static_cast<float>( j ) / 
                                static_cast<float>( lat_div ) * PIf;
        const float sin_theta = sinf( theta );
        const float cos_theta = cosf( theta );
        for( unsigned i = 0u; i <= long_div; ++i )
        {
            const float phi     = static_cast<float>( i ) / 
                                  static_cast<float>( long_div ) * PIf * 2.0f;
            const float sin_phi = sinf( phi );
            const float cos_phi = cosf( phi );
            verts[ vindex ].normal = Vector3( sin_theta * cos_phi,
                                              cos_theta,
                                              -sin_theta * sin_phi );
            verts[ vindex ].position = verts[ vindex ].normal*radius + offset;
            verts[ vindex ].texcoord = Vector2( phi / (2.0f*PIf), theta/PIf );
            ++vindex;
        }
    }
    verts[ vindex ].position   = Vector3( 0.0f, radius,  0.0f )+offset;
    verts[ vindex ].normal     = Vector3( 0.0f, 1.0f,  0.0f );
    verts[ vindex ].texcoord   = Vector2( 0.0f, 1.0f );
    verts[ ++vindex ].position = Vector3( 0.0f, -radius,  0.0f )+offset;
    verts[ vindex ].normal     = Vector3( 0.0f, -1.0f,  0.0f );
    verts[ vindex ].texcoord   = Vector2( 0.0f, 0.0f );

    // generate triangle indices
    indices.resize( num_tris );
    unsigned tindex = 0u;
    for( unsigned j = 0u; j < lat_div-2u; ++j )
    {
        for( unsigned i = 0u; i < long_div; ++i )
        {

            indices[ tindex++ ] = Index3( (j+0u)*(long_div+1) + i,
                                          (j+1u)*(long_div+1) + i + 1,
                                          (j+0u)*(long_div+1) + i + 1 );
            indices[ tindex++ ] = Index3( (j+0u)*(long_div+1) + i,
                                          (j+1u)*(long_div+1) + i,
                                          (j+1u)*(long_div+1) + i + 1 );
        }
    }
    for( unsigned i = 0u; i < long_div; ++i )
    {
        indices[ tindex++ ] = Index3( (lat_div-1u)*(long_div+1u), i, i+1 );
        indices[ tindex++ ] = Index3( (lat_div-1u)*(long_div+1u) + 1,
                                      (lat_div-2u)*(long_div+1u) + i + 1,
                                      (lat_div-2u)*(long_div+1u) + i );
    }
}


int main( int argc, char** argv )
{
    unsigned sspp = 2;
    if( argc > 1 )
    {
        sspp = atoi( argv[ 1 ] );
    }

    try
    {
        legion::Context ctx( "legion_simple" );
        legion::Log::setReportingLevel( legion::Log::INFO );

        ctx.setSamplesPerPixel( legion::Index2( sspp, sspp ) );

        LLOG_INFO << "Starting ***********";
        
        // Parameters params;
        // params.add( "Kd", legion::Color( 0.5f, 0.5f, 0.5f ) );
        // legion::createSurfaceShader( "Lambertian", "material", params );
        legion::LambertianShader mtl( &ctx, "material" );
        mtl.setKd( legion::Color(  0.5f, 0.5f, 0.5f ) );
       
        std::vector<legion::Mesh::Vertex> verts;
        std::vector<legion::Index3> indices;
        generateSphere( verts, indices, 10, 20, 0.5f, 
                        legion::Vector3(0.0f, 0.0f, -1.0f) );

        legion::Mesh square( &ctx, "sphere" );
        square.setVertices( verts.size(), &verts[0] );
        square.setTransform( legion::Matrix4x4::identity() );
        square.setFaces( indices.size(), &indices[0], &mtl );
        ctx.addMesh( &square );

        legion::PointLightShader light( &ctx, "lshader" );
        light.setPosition( legion::Vector3( 1.0f, 1.0f, 1.0f ) );
        light.setRadiantFlux( legion::Color( 1.0f, 1.0f, 1.0f ) );
        ctx.addLight( &light );
        //ctx.addAreaLigth(...); 

        legion::ThinLensCamera cam( &ctx, "camera" );
        cam.setViewPlane( -1.0f, 1.0f, 1.0f, -1.0f );
        cam.setShutterOpenClose( 0.0f, 0.005f );
        cam.setFocalDistance( 1.0f );
        cam.setLensRadius( 0.0f );
        ctx.setActiveCamera( &cam );

        legion::ImageFilm film( &ctx, "image" );
        //film.setDimensions( legion::Index2( 1024u, 1024u) );
        film.setDimensions( legion::Index2( 512u, 512u ) );
        //film.setDimensions( legion::Index2( 256u, 256u ) );
        //film.setDimensions( legion::Index2( 3u, 3u ) );
        ctx.setActiveFilm( &film );

        ctx.render();
        LLOG_INFO << "Finished ***********";
    }
    catch( legion::Exception& e )
    {
        LLOG_ERROR << e.what();
    }
}

