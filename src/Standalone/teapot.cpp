
#include <Legion/Legion.hpp>
#include <vector>
#include <fstream>

void readMesh( std::vector<legion::Index3>&  triangles,
               std::vector<legion::Vector3>& vertices )
{
    std::ifstream in( "../src/Standalone/data/teapot.obj" );
    std::string line;
    float p0, p1, p2;
    std::string token;
    unsigned i0, i1, i2;
    while( in )
    {
        std::getline( in,  line );
        if( !line.empty() && line[0] == 'v' )
        {
            std::istringstream iss( line );
            iss >> token >> p0 >> p1 >> p2;
            vertices.push_back( legion::Vector3( p0, p1, p2 ) );
            //std::cout << token << ": " << p0 << ", " << p1 << ", " << p2 
            //          << std::endl;
        }
        else if( !line.empty() && line[0] == 'f' )
        {
            std::istringstream iss( line );
            iss >> token >> i0 >> i1 >> i2;
            triangles.push_back( legion::Index3( i0-1, i1-1, i2-1 ) );
            //std::cout << token << ": " << i0 << ", " << i1 << ", " << i2 
            //          << std::endl;
        }
    }
}


int main( int , char** )
{
    try
    {
        legion::Context ctx;

        legion::ImageFileDisplay display( &ctx, "teapot.exr" );
        display.beginScene( "teapot" );

        legion::Log::setReportingLevel( legion::Log::INFO );

        LLOG_INFO << "teapot scene ...";
        
        legion::Lambertian lambertian( &ctx );
        lambertian.setReflectance( legion::Color(  0.7f, 0.7f, 0.7f ) );
       
        std::vector<legion::Index3> triangles;
        std::vector<legion::Vector3> vertices;
        readMesh( triangles, vertices );

        legion::TriMesh mesh( &ctx );
        mesh.setTriangles( vertices, triangles );
        mesh.setSurface( &lambertian );
        ctx.addGeometry( &mesh );
        
        /*
        legion::Sphere sphere( &ctx );
        sphere.setRadius( 2.0f );
        sphere.setSurface( &lambertian );
        ctx.addGeometry( &sphere );
        */
        
        legion::Parallelogram pgram( &ctx );
        pgram.setAnchorUV( 
            legion::Vector3( -10.0f, 0.0f,   6.0f ),
            legion::Vector3(  20.0f, 0.0f,   0.0f ),
            legion::Vector3(   0.0f, 0.0f, -20.0f )
            );
        pgram.setSurface( &lambertian );
        ctx.addGeometry( &pgram );

        legion::ThinLens cam( &ctx );
        cam.setCameraToWorld( 
                legion::Matrix::translate( 
                    legion::Vector3( 0.0f, 2.0f, 8.0f ) ) );
        ctx.setCamera( &cam );

        legion::ConstantEnvironment env( &ctx );
        env.setRadiance( legion::Color( 0.529f, 0.808f, 0.922f )*0.5f );
        ctx.setEnvironment( &env );

        legion::ProgressiveRenderer renderer( &ctx );
        renderer.setSamplesPerPixel( 64 );
        renderer.setDisplay( &display );
        ctx.setRenderer( &renderer );

        ctx.render();
    }
    catch( std::exception& e )
    {
        LLOG_ERROR << e.what();
    }
}

