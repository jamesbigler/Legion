
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
    const unsigned num_spheres = 4;
    try
    {
        legion::Context ctx( "legion_simple" );
        legion::Log::setReportingLevel( legion::Log::INFO );

        ctx.setSamplesPerPixel( legion::Index2( 2, 2 ) );

        LLOG_INFO << "Starting ***********";
        
        // Parameters params;
        // params.add( "Kd", legion::Color( 0.5f, 0.5f, 0.5f ) );
        // legion::createSurfaceShader( "Lambertian", "material", params );
        legion::LambertianShader mtl( &ctx, "material" );
        mtl.setKd( legion::Color(  0.5f, 0.5f, 0.5f ) );
       
       
        std::vector<legion::Mesh*> meshes( num_spheres+1 );
        for( unsigned i = 0; i < num_spheres; ++i )
        {
            std::vector<legion::Mesh::Vertex> verts;
            std::vector<legion::Index3> indices;
            //generateSphere( verts, indices, 10, 20, 0.5f, 
            generateSphere( verts, indices, 10, 20, 0.25f, 
                            legion::Vector3(-0.75f + i*0.5f, 0.0f, -5.0f) );

            meshes[i] = new legion::Mesh( &ctx, "sphere" );
            meshes[i]->setVertices( verts.size(), &verts[0] );
            meshes[i]->setTransform( legion::Matrix4x4::identity() );
            meshes[i]->setFaces( indices.size(), &indices[0], &mtl );
            ctx.addMesh( meshes[i] );
        }

        // Plane
        std::vector<legion::Mesh::Vertex> verts(4);
        std::vector<legion::Index3> indices(2);
        verts[ 0 ].position = legion::Vector3( -1.5f, -0.25f, -3.0f );
        verts[ 0 ].normal   = legion::Vector3(  0.0f,  1.0f,   0.0f );
        verts[ 0 ].texcoord = legion::Vector2(  0.0f,  0.0f );
        verts[ 1 ].position = legion::Vector3( -1.5f, -0.25f, -7.0f );
        verts[ 1 ].normal   = legion::Vector3(  0.0f,  1.0f,   0.0f );
        verts[ 1 ].texcoord = legion::Vector2(  0.0f,  1.0f );
        verts[ 2 ].position = legion::Vector3(  1.5f, -0.25f, -7.0f );
        verts[ 2 ].normal   = legion::Vector3(  0.0f,  1.0f,   0.0f );
        verts[ 2 ].texcoord = legion::Vector2(  1.0f,  1.0f );
        verts[ 3 ].position = legion::Vector3(  1.5f, -0.25f, -3.0f );
        verts[ 3 ].normal   = legion::Vector3(  0.0f,  1.0f,   0.0f );
        verts[ 3 ].texcoord = legion::Vector2(  1.0f,  0.0f );
        indices[0] = legion::Index3( 0, 2, 1 );
        indices[1] = legion::Index3( 0, 3, 2 );
        meshes[num_spheres] = new legion::Mesh( &ctx, "plane" );
        meshes[num_spheres]->setVertices( verts.size(), &verts[0] );
        meshes[num_spheres]->setTransform( legion::Matrix4x4::identity() );
        meshes[num_spheres]->setFaces( indices.size(), &indices[0], &mtl );
        ctx.addMesh( meshes[num_spheres] );

        legion::PointLightShader light( &ctx, "lshader" );
        light.setPosition( legion::Vector3( 1.0f, 1.0f, 1.0f ) );
        light.setRadiantFlux( legion::Color( 1.0f, 1.0f, 1.0f ) );
        ctx.addLight( &light );
        //ctx.addAreaLigth(...); 

        legion::ThinLensCamera cam( &ctx, "camera" );
        cam.setViewPlane( -1.0f, 1.0f, 0.75f, -0.75f );
        cam.setShutterOpenClose( 0.0f, 0.005f );
        cam.setFocalDistance( 5.0f );
        cam.setLensRadius( 0.0f );
        ctx.setActiveCamera( &cam );

        legion::ImageFilm film( &ctx, "image" );
        film.setDimensions( legion::Index2( 1024u, 768u) );
        //film.setDimensions( legion::Index2( 512u, 512u ) );
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

