
#include <Legion.hpp>
#include <vector>
#include <iostream>

int main( int argc, char** argv )
{
    try
    {
        std::cerr << " Starting ***********" << std::endl;
        legion::Context ctx( "legion_simple" );
        
        // Parameters params;
        // params.add( "Kd", legion::Color( 0.5f, 0.5f, 0.5f ) );
        // legion::createSurfaceShader( "Lambertian", "material", params );
        legion::LambertianShader mtl( "material" );
        mtl.setKd( legion::Color(  0.5f, 0.5f, 0.5f ) );
       
        std::vector<legion::Vector3> vertices;
        vertices.push_back( legion::Vector3( -0.5f, -0.5f, 0.0f ) );
        vertices.push_back( legion::Vector3( -0.5f,  0.5f, 0.0f ) );
        vertices.push_back( legion::Vector3(  0.5f,  0.5f, 0.0f ) );
        vertices.push_back( legion::Vector3(  0.5f, -0.5f, 0.0f ) );

        std::vector<legion::Index3> indices;
        indices.push_back( legion::Index3( 0, 1, 3 ) );
        indices.push_back( legion::Index3( 1, 2, 3 ) );

        legion::Mesh square = legion::Mesh( "square",
                                            legion::Mesh::TYPE_POLYGONAL,
                                            vertices.size() );
        square.setTime( 0.0f );
        square.setVertices( &vertices[0] );
        square.setTransform( legion::Matrix4x4::identity() );
        square.addTriangles( indices.size(), &indices[0], mtl );
        ctx.addMesh( &square );

        legion::PointLightShader light = legion::PointLightShader( "lshader" );
        light.setPosition( legion::Vector3( 1.0f, 1.0f, 1.0f ) );
        light.setRadiantFlux( legion::Color( 1.0f, 1.0f, 1.0f ) );
        ctx.addLight( &light );
        //ctx.addAreaLigth(...); 

        legion::ThinLensCamera cam( "camera" );
        cam.setViewPlane( -1.0f, 1.0f, -0.75f, 0.75f );
        cam.setShutterOpenClose( 0.0f, 0.005f );
        cam.setFocalDistance( 1.0f );
        cam.setLensRadius( 0.0f );
        ctx.setActiveCamera( &cam );

        legion::ImageFilm film = legion::ImageFilm( "image" );
        film.setDimensions( legion::Index2( 4u, 4u ) );
        ctx.setActiveFilm( &film );

        ctx.render();
        std::cerr << " Finished ***********" << std::endl;
    }
    catch( legion::Exception& e )
    {
        std::cerr << e.what() << std::endl;
    }
}

