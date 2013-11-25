
#include <Legion/Legion.hpp>
#include <vector>

int main( int , char** )
{
    try
    {
        legion::Context ctx;

        legion::ImageFileDisplay display( &ctx, "white.exr" );
        display.beginScene( "simple" );
        

        // TODO: ctx.log.setReportingLevel( legion::Log::INFO );
        //     : each api class can have log func which calls m_ctx.log() 
        legion::Log::setReportingLevel( legion::Log::INFO );

        // TODO: can be defined as ctx.log.info()
        LLOG_INFO << "simple scene ...";
        
        legion::Parameters params;
        params.set( "value", legion::Color( 1.0f, 1.0f, 1.0f ) );
        //params.set( "value", legion::Color( 0.5f, 0.5f, 0.5f ) );
        legion::ITexture* white_tex = 
            ctx.createTexture( "ConstantTexture", params);

        params.clear();
        params.set( "reflectance", white_tex );
        legion::ISurface* lambertian =
            ctx.createSurface( "Lambertian", params);

        legion::Sphere sphere( &ctx );
        sphere.setCenter( legion::Vector3( 0.0f, 0.0f, -5.0f ) );
        sphere.setRadius( 1.0f ); 
        sphere.setSurface( lambertian );
        ctx.addGeometry( &sphere );
        
        legion::ThinLens cam( &ctx );
        ctx.setCamera( &cam );

        legion::ConstantEnvironment env( &ctx );
        //env.setRadiance( legion::Color( 0.529f, 0.808f, 0.922f )*0.5f );
        env.setRadiance( legion::Color( 1.0f, 1.0f, 1.0f ) );
        ctx.setEnvironment( &env );

        legion::ProgressiveRenderer renderer( &ctx );
        renderer.setSamplesPerPixel( 32*32 );
        renderer.setDisplay( &display );
        ctx.setRenderer( &renderer );

        ctx.render();
    }
    catch( legion::Exception& e )
    {
        std::cerr << e.what();
        LLOG_ERROR << e.what();
    }
}

