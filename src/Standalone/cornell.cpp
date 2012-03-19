
#include <Legion/Legion.hpp>

#include <vector>
#include <iomanip>

void addQuad( std::vector<legion::Mesh::Vertex>& verts,
              std::vector<legion::Index3>&       faces,
              const legion::Vector3& p0,
              const legion::Vector3& p1,
              const legion::Vector3& p2,
              const legion::Vector3& p3 )
{
    using legion::Mesh;

    unsigned sidx = verts.size();

    legion::Vector3 n = legion::normalize( legion::cross( p1-p0, p2-p0 ) );
    verts.push_back( Mesh::Vertex( p0, n, legion::Vector2( 0.0f, 0.0f ) ) );
    verts.push_back( Mesh::Vertex( p1, n, legion::Vector2( 0.0f, 0.0f ) ) );
    verts.push_back( Mesh::Vertex( p2, n, legion::Vector2( 0.0f, 0.0f ) ) );
    verts.push_back( Mesh::Vertex( p3, n, legion::Vector2( 0.0f, 0.0f ) ) );

    faces.push_back( legion::Index3( sidx+0, sidx+2, sidx+1 ) );
    faces.push_back( legion::Index3( sidx+0, sidx+2, sidx+3 ) );
}


int main( int argc, char** argv )
{
    unsigned sspp = 2;
    if( argc > 1 )
    {
        sspp = atoi( argv[ 1 ] );
    }

    unsigned ray_depth = 2;
    if( argc > 2 )
    {
        ray_depth = atoi( argv[ 2 ] );
    }

    try
    {
        legion::Context ctx( "legion_simple" );
        legion::Log::setReportingLevel( legion::Log::INFO );

        ctx.setSamplesPerPixel( legion::Index2( sspp, sspp ) );
        ctx.setMaxRayDepth( ray_depth );

        //
        // Set up materials
        //
        legion::Lambertian black( &ctx, "black" );
        black.setKd( legion::Color( 0.0f, 0.0f, 0.0f ) );

        legion::Lambertian red( &ctx, "red" );
        red.setKd( legion::Color( 0.80f, 0.05f, 0.05f ) );
        
        legion::Lambertian green( &ctx, "green" );
        green.setKd( legion::Color( 0.05f, 0.80f, 0.05f ) );
        
        legion::Lambertian white( &ctx, "white" );
        white.setKd( legion::Color( 0.80f, 0.80f, 0.80f ) );
        
        legion::DiffuseLight diffuse_light( &ctx, "diffuse_light" );
        diffuse_light.setEmittance( legion::Color( 100000.0f ) );

        //
        // Set up geometry
        //
        std::vector<legion::Mesh::Vertex> verts;
        std::vector<legion::Index3>       faces;

        // Light geometry
        legion::Mesh light_mesh( &ctx, "light" );
        addQuad( verts, faces,
                 legion::Vector3( 343.0f, 548.7f, 227.0f ),
                 legion::Vector3( 343.0f, 548.7f, 332.0f ),
                 legion::Vector3( 213.0f, 548.7f, 332.0f ),
                 legion::Vector3( 213.0f, 548.7f, 227.0f ) ); 
        light_mesh.setVertices( verts.size(), &verts[0] );
        light_mesh.setFaces( faces.size(), &faces[0], &black, &diffuse_light );
        ctx.addMesh( &light_mesh );

        // White surfaces
        verts.clear();
        faces.clear();
        legion::Mesh white_mesh( &ctx, "white" );

        // floor
        addQuad( verts, faces,
                 legion::Vector3( 552.8, 0.0f, 0.0f ),
                 legion::Vector3( 0.0f, 0.0f, 0.0f ),
                 legion::Vector3( 0.0f, 0.0f, 559.2 ),
                 legion::Vector3( 549.6, 0.0f, 559.2 ) ); 
        // ceiling
        addQuad( verts, faces,
                 legion::Vector3( 556.0f, 548.8, 0.0f ),
                 legion::Vector3( 556.0f, 548.8, 559.2 ),
                 legion::Vector3( 0.0f, 548.8, 559.2 ),
                 legion::Vector3( 0.0f, 548.8, 0.0f ) ); 
        // back wall
        addQuad( verts, faces,
                legion::Vector3( 549.6, 0.0f, 559.2 ),
                legion::Vector3( 0.0f, 0.0f, 559.2 ),
                legion::Vector3( 0.0f, 548.8, 559.2 ),
                legion::Vector3( 556.0f, 548.8, 559.2 ) ); 

        /*
        // Short block
        addQuad( verts, faces,
                legion::Vector3( 130.0f, 165.0f, 65.0f ),
                legion::Vector3( 82.0f, 165.0f, 225.0f ),
                legion::Vector3( 240.0f, 165.0f, 272.0f ),
                legion::Vector3( 290.0f, 165.0f, 114.0f ) ); 
        addQuad( verts, faces,
                legion::Vector3( 290.0f, 0.0f, 114.0f ),
                legion::Vector3( 290.0f, 165.0f, 114.0f ),
                legion::Vector3( 240.0f, 165.0f, 272.0f ),
                legion::Vector3( 240.0f, 0.0f, 272.0f ) ); 
        addQuad( verts, faces,
                legion::Vector3( 130.0f, 0.0f, 65.0f ),
                legion::Vector3( 130.0f, 165.0f, 65.0f ),
                legion::Vector3( 290.0f, 165.0f, 114.0f ),
                legion::Vector3( 290.0f, 0.0f, 114.0f ) ); 
        addQuad( verts, faces,
                legion::Vector3( 82.0f, 0.0f, 225.0f ),
                legion::Vector3( 82.0f, 165.0f, 225.0f ),
                legion::Vector3( 130.0f, 165.0f, 65.0f ),
                legion::Vector3( 130.0f, 0.0f, 65.0f ) ); 
        addQuad( verts, faces,
                legion::Vector3( 240.0f, 0.0f, 272.0f ),
                legion::Vector3( 240.0f, 165.0f, 272.0f ),
                legion::Vector3( 82.0f, 165.0f, 225.0f ),
                legion::Vector3( 82.0f, 0.0f, 225.0f ) ); 

        // tall block
        addQuad( verts, faces,
                legion::Vector3( 423.0f, 330.0f, 247.0f ),
                legion::Vector3( 265.0f, 330.0f, 296.0f ),
                legion::Vector3( 314.0f, 330.0f, 456.0f ),
                legion::Vector3( 472.0f, 330.0f, 406.0f ) ); 
        addQuad( verts, faces,
                legion::Vector3( 423.0f, 0.0f, 247.0f ),
                legion::Vector3( 423.0f, 330.0f, 247.0f ),
                legion::Vector3( 472.0f, 330.0f, 406.0f ),
                legion::Vector3( 472.0f, 0.0f, 406.0f ) ); 
        addQuad( verts, faces,
                legion::Vector3( 472.0f, 0.0f, 406.0f ),
                legion::Vector3( 472.0f, 330.0f, 406.0f ),
                legion::Vector3( 314.0f, 330.0f, 456.0f ),
                legion::Vector3( 314.0f, 0.0f, 456.0f ) ); 
        addQuad( verts, faces,
                legion::Vector3( 314.0f, 0.0f, 456.0f ),
                legion::Vector3( 314.0f, 330.0f, 456.0f ),
                legion::Vector3( 265.0f, 330.0f, 296.0f ),
                legion::Vector3( 265.0f, 0.0f, 296.0f ) ); 
        addQuad( verts, faces,
                legion::Vector3( 265.0f, 0.0f, 296.0f ),
                legion::Vector3( 265.0f, 330.0f, 296.0f ),
                legion::Vector3( 423.0f, 330.0f, 247.0f ),
                legion::Vector3( 423.0f, 0.0f, 247.0f ) ); 
                */

        white_mesh.setVertices( verts.size(), &verts[0] );
        white_mesh.setFaces( faces.size(), &faces[0], &white );
        ctx.addMesh( &white_mesh );

        // Green wall on rhs
        verts.clear();
        faces.clear();
        legion::Mesh right_wall_mesh( &ctx, "right_wall" );
        addQuad( verts, faces,
                legion::Vector3( 0.0f, 0.0f, 559.2 ),
                legion::Vector3( 0.0f, 0.0f, 0.0f ),
                legion::Vector3( 0.0f, 548.8, 0.0f ),
                legion::Vector3( 0.0f, 548.8, 559.2 ) ); 
        right_wall_mesh.setVertices( verts.size(), &verts[0] );
        right_wall_mesh.setFaces( faces.size(), &faces[0], &green );
        ctx.addMesh( &right_wall_mesh);

        // Red wall on lhs
        verts.clear();
        faces.clear();
        legion::Mesh left_wall_mesh( &ctx, "left_wall" );
        addQuad( verts, faces,
                legion::Vector3( 552.8, 0.0f, 0.0f ),
                legion::Vector3( 549.6, 0.0f, 559.2 ),
                legion::Vector3( 556.0f, 548.8, 559.2 ),
                legion::Vector3( 556.0f, 548.8, 0.0f ) ); 
        left_wall_mesh.setVertices( verts.size(), &verts[0] );
        left_wall_mesh.setFaces( faces.size(), &faces[0], &red);
        ctx.addMesh( &left_wall_mesh);

        legion::ThinLensCamera cam( &ctx, "camera" );
        cam.setViewPlane( -1.0f, 1.0f, -1.0f, 1.0f );
        cam.setShutterOpenClose( 0.0f, 0.005f );
        cam.setFocalDistance( 2.5f );
        cam.setLensRadius( 0.0f );
        legion::Matrix c2w =
            legion::Matrix::translate(legion::Vector3(278.0f, 273.0f, -800.0f))*
            legion::Matrix::rotate( M_PI, legion::Vector3( 0.0f, 1.0f, 0.0f) );
        cam.setTransform( c2w, 0.0f );
        ctx.setActiveCamera( &cam );

        legion::ImageFilm film( &ctx, "image" );
        film.setDimensions( legion::Index2( 512u, 512u ) );
        //film.setDimensions( legion::Index2( 64u, 64u ) );
        //film.setDimensions( legion::Index2( 40u, 40u ) );
        //film.setDimensions( legion::Index2( 10u, 10u ) );
        ctx.setActiveFilm( &film );

        ctx.render();
        LLOG_INFO << "Finished ***********";
    }
    catch( legion::Exception& e )
    {
        LLOG_ERROR << e.what();
    }
}

