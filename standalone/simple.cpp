
#include <Legion.hpp>
#include <vector>
#include <iostream>

int main( int argc, char** argv )
{
    try
    {
        std::cerr << " Starting ***********" << std::endl;
        legion::Context ctx( "legion_simple" );
        
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

        legion::Mesh square = legion::Mesh( "square", legion::Mesh::TYPE_POLYGONAL, vertices.size() );
        square.setTime( 0.0f );
        //square.setVertices( vertices.begin(), vertices.end );
        square.setVertices( &vertices[0] );
        square.setTransform( legion::Matrix4x4::identity() );
        //square.addTriangles( indices.begin(), indices.end() );
        square.addTriangles( indices.size(), &indices[0], mtl );
        ctx.addMesh( &square );

        legion::ThinLensCamera cam( "camera" );
        cam.setViewPlane( -1.0f, 1.0f, -0.75f, 0.75f );
        cam.setShutterOpenClose( 0.0f, 0.005f );
        cam.setFocalDistance( 10.0f );
        cam.setLensRadius( 0.005f );
        ctx.setActiveCamera( &cam );

        legion::ImageFilm film = legion::ImageFilm( "image" );
        film.setDimensions( legion::Index2( 1024u, 756u ) );
        ctx.setActiveFilm( &film );

        ctx.render();
        std::cerr << " Finished ***********" << std::endl;
    }
    catch( legion::Exception& e )
    {
        std::cerr << e.what() << std::endl;
    }
}

