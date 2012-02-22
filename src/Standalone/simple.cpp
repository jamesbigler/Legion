
#include <Legion/Legion.hpp>
#include <vector>

int main( int argc, char** argv )
{
    try
    {
        legion::Context ctx( "legion_simple" );
        legion::Log::setReportingLevel( legion::Log::INFO );

        LLOG_INFO << "Starting ***********";
        
        // Parameters params;
        // params.add( "Kd", legion::Color( 0.5f, 0.5f, 0.5f ) );
        // legion::createSurfaceShader( "Lambertian", "material", params );
        legion::LambertianShader mtl( &ctx, "material" );
        mtl.setKd( legion::Color(  0.5f, 0.5f, 0.5f ) );
       
        std::vector<legion::Mesh::Vertex> verts;
        verts.push_back( legion::Mesh::Vertex( legion::Vector3(-0.5f,-0.5f, 0.0f ),
                                         legion::Vector3( 0.0f, 0.0f,-1.0f ),
                                         legion::Vector3( 0.0f, 0.0f, 0.0f )) );
        verts.push_back( legion::Mesh::Vertex( legion::Vector3(-0.5f, 0.5f, 0.0f ),
                                         legion::Vector3( 0.0f, 0.0f,-1.0f ),
                                         legion::Vector3( 0.0f, 0.0f, 0.0f )) );
        verts.push_back( legion::Mesh::Vertex( legion::Vector3( 0.5f, 0.5f, 0.0f ),
                                         legion::Vector3( 0.0f, 0.0f,-1.0f ),
                                         legion::Vector3( 0.0f, 0.0f, 0.0f )) );
        verts.push_back( legion::Mesh::Vertex( legion::Vector3( 0.5f, -0.5f, 0.0f),
                                         legion::Vector3( 0.0f, 0.0f,-1.0f ),
                                         legion::Vector3( 0.0f, 0.0f, 0.0f )) );

        std::vector<legion::Index3> indices;
        indices.push_back( legion::Index3( 0, 1, 3 ) );
        indices.push_back( legion::Index3( 1, 2, 3 ) );

        legion::Mesh square( &ctx, "square" );
        square.setVertices( vertices.size(), &vertices[0] );
        square.setTransform( legion::Matrix4x4::identity() );
        square.addTriangles( indices.size(), &indices[0], &mtl );
        ctx.addMesh( &square );

        legion::PointLightShader light( &ctx, "lshader" );
        light.setPosition( legion::Vector3( 1.0f, 1.0f, 1.0f ) );
        light.setRadiantFlux( legion::Color( 1.0f, 1.0f, 1.0f ) );
        ctx.addLight( &light );
        //ctx.addAreaLigth(...); 

        legion::ThinLensCamera cam( &ctx, "camera" );
        cam.setViewPlane( -1.0f, 1.0f, -0.75f, 0.75f );
        cam.setShutterOpenClose( 0.0f, 0.005f );
        cam.setFocalDistance( 1.0f );
        cam.setLensRadius( 0.0f );
        ctx.setActiveCamera( &cam );

        legion::ImageFilm film( &ctx, "image" );
        film.setDimensions( legion::Index2( 4u, 4u ) );
        ctx.setActiveFilm( &film );

        ctx.render();
        LLOG_INFO << "Finished ***********";
    }
    catch( legion::Exception& e )
    {
        LLOG_ERROR << e.what();
    }
}

