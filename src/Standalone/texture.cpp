
#include <Legion/Legion.hpp>
#include <vector>

int main( int , char** )
{
    try
    {
        legion::Context ctx;

        legion::ImageFileDisplay display( &ctx, "texture.exr" );
        display.beginScene( "texture" );

        legion::Log::setReportingLevel( legion::Log::INFO );

        LLOG_INFO << "texture scene ...";
        
        //
        // Textures
        //
        legion::ConstantTexture white_tex( &ctx );
        white_tex.set( legion::Color(  1.0f, 1.0f, 1.0f ) );
        
        legion::ImageTexture brick_tex( &ctx );
        brick_tex.set( "../src/Standalone/data/bricks.exr" );

        legion::PerlinTexture perlin_tex( &ctx );

        //
        // Surfaces 
        //
        legion::Lambertian white( &ctx );
        white.setReflectance( &white_tex );
        
        legion::Lambertian brick( &ctx );
        brick.setReflectance( &brick_tex );
        
        legion::Lambertian perlin( &ctx );
        perlin.setReflectance( &perlin_tex );
       
        legion::DiffuseEmitter emitter( &ctx );
        emitter.setRadiance( legion::Color(  4.0f, 4.0f, 4.0f ) );
        
        //
        // Geometry 
        //
        legion::Sphere sphere0( &ctx );
        sphere0.setCenter( legion::Vector3( 1.0f, 0.0f, 0.0f ) );
        sphere0.setSurface( &brick );
        ctx.addGeometry( &sphere0 );
        
        legion::Sphere sphere1( &ctx );
        sphere1.setCenter( legion::Vector3( -1.0f, 0.0f, 0.0f ) );
        sphere1.setSurface( &perlin );
        ctx.addGeometry( &sphere1 );
        
        legion::Parallelogram floor( &ctx );
        floor.setAnchorUV( 
            legion::Vector3( -10.0f, -1.0f,  10.0f ),
            legion::Vector3(  20.0f,  0.0f,   0.0f ),
            legion::Vector3(   0.0f,  0.0f, -20.0f )
            );
        floor.setSurface( &white );
        ctx.addGeometry( &floor);

        legion::Parallelogram light( &ctx );
        light.setAnchorUV( 
            legion::Vector3( -2.0f, 4.0f, -2.0f ),
            legion::Vector3(  0.0f, 0.0f, -4.0f ),
            legion::Vector3(  4.0f, 0.0f,  0.0f )
            );
        light.setSurface( &emitter );
        ctx.addGeometry( &light);


        legion::ThinLens cam( &ctx );
        legion::Matrix cam_to_world = 
            legion::Matrix::translate( legion::Vector3( 0.0f, 0.0f, -5.0f) ) *
            legion::Matrix::rotate( legion::PI, legion::Vector3( 0.0f, 1.0f, 0.0f) );
        cam.setCameraToWorld( cam_to_world );
        ctx.setCamera( &cam );

        legion::ProgressiveRenderer renderer( &ctx );
        renderer.setSamplesPerPixel( 128 );
        renderer.setDisplay( &display );
        ctx.setRenderer( &renderer );

        ctx.render();
    }
    catch( std::exception& e )
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        LLOG_ERROR << e.what();
    }
}

