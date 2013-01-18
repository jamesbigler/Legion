
#include <Legion/Legion.hpp>
#include <vector>

int main( int , char** )
{
    try
    {
        legion::Context ctx;

        // TODO: ctx.log.setReportingLevel( legion::Log::INFO );
        //     : each api class can have log func which calls m_ctx.log() 
        legion::Log::setReportingLevel( legion::Log::INFO );

        // TODO: can be defined as ctx.log.info()
        LLOG_INFO << "simple scene ...";
        
        // Parameters params;
        // params.add( "Kd", legion::Color( 0.5f, 0.5f, 0.5f ) );
        // legion::createSurfaceShader( "Lambertian", "material", params );
        legion::Lambertian mtl( &ctx );
        mtl.setReflectance( legion::Color(  0.9f, 0.5f, 0.5f ) );
       
        legion::Sphere sphere( &ctx );
        sphere.setCenter( legion::Vector3( 0.0f, 0.0f, -4.0f ) );
        sphere.setSurface( &mtl );
        ctx.addGeometry( &sphere );
        
        legion::Parallelogram pgram( &ctx );
        pgram.setAnchorUV( 
            legion::Vector3( -10.0f, -1.0f,   6.0f ),
            legion::Vector3(  20.0f,  0.0f,   0.0f ),
            legion::Vector3(   0.0f,  0.0f, -20.0f )
            );
        pgram.setSurface( &mtl );
        ctx.addGeometry( &pgram );

        legion::PointLight light( &ctx );
        light.setPosition( legion::Vector3( 1.0f, 1.0f, 1.0f ) );
        light.setIntensity( legion::Color( 1.0f, 1.0f, 1.0f ) );

        legion::ThinLens cam( &ctx );
        ctx.setCamera( &cam );

        legion::ProgressiveRenderer renderer( &ctx );
        renderer.setSamplesPerPixel( 64 );
        renderer.setSamplesPerPass( 8 );
        ctx.setRenderer( &renderer );

        legion::ImageFileDisplay display( &ctx, "simple.exr" );
        renderer.setDisplay( &display );
        
        ctx.render();
    }
    catch( legion::Exception& e )
    {
        LLOG_ERROR << e.what();
    }
}

