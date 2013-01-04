
#include <Legion/Legion.hpp>
#include <vector>

int main( int argc, char** argv )
{
    try
    {
        legion::Context ctx;

        // TODO: ctx.log.setReportingLevel( legion::Log::INFO );
        //     : each api class can have log func which calls m_ctx.log() 
        legion::Log::setReportingLevel( legion::Log::INFO );

        // TODO: can be defined as ctx.log.info()
        LLOG_INFO << "Starting ***********";
        
        // Parameters params;
        // params.add( "Kd", legion::Color( 0.5f, 0.5f, 0.5f ) );
        // legion::createSurfaceShader( "Lambertian", "material", params );
        legion::Lambertian mtl;
        mtl.setKd( legion::Color(  0.5f, 0.5f, 0.5f ) );
       
        /*
        std::vector<legion::Mesh::Vertex> verts;
        verts.push_back( legion::Mesh::Vertex(
                             legion::Vector3(-0.5f,-0.5f,-1.0f ),
                             legion::Vector3( 0.0f, 0.0f,-1.0f ),
                             legion::Vector2( 0.0f, 0.0f ) ) );
        verts.push_back( legion::Mesh::Vertex(
                             legion::Vector3(-0.5f, 0.5f,-1.0f ),
                             legion::Vector3( 0.0f, 0.0f,-1.0f ),
                             legion::Vector2( 0.0f, 0.0f ) ) );
        verts.push_back( legion::Mesh::Vertex(
                             legion::Vector3( 0.5f, 0.5f,-1.0f ),
                             legion::Vector3( 0.0f, 0.0f,-1.0f ),
                             legion::Vector2( 0.0f, 0.0f ) ) );
        verts.push_back( legion::Mesh::Vertex(
                             legion::Vector3( 0.5f,-0.5f,-1.0f),
                             legion::Vector3( 0.0f, 0.0f,-1.0f ),
                             legion::Vector2( 0.0f, 0.0f ) ) );

        std::vector<legion::Index3> indices;
        indices.push_back( legion::Index3( 0, 1, 2 ) );
        indices.push_back( legion::Index3( 0, 2, 3 ) );
        */

        legion::Sphere sphere( &ctx );
        sphere.setCenter( legion::Vector3( 0.0f, 0.0f, -4.0f ) );
        sphere.setSurface( &mtl );
        /*
        square.setVertices( verts.size(), &verts[0] );
        //square.setVertices( verts.begin(), verts.end() );
        square.setFaces( indices.size(), &indices[0], &mtl );
        //square.setFaces( indices.begin(), indices.end() );
        */
        ctx.addGeometry( &sphere );

        legion::PointLight light( &ctx );
        light.setPosition( legion::Vector3( 1.0f, 1.0f, 1.0f ) );
        light.setIntensity( legion::Color( 1.0f, 1.0f, 1.0f ) );
        ctx.addLight( &light );
        //ctx.addAreaLigth(...); 

        legion::ThinLens cam( &ctx );
        ctx.setCamera( &cam );

        legion::ImageFilm film;
        //film.setDimensions( legion::Index2( 256u, 256u ) );
        ctx.setFilm( &film );

        ctx.render();
        LLOG_INFO << "Finished ***********";
    }
    catch( legion::Exception& e )
    {
        LLOG_ERROR << e.what();
    }
}

