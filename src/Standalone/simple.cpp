
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
        legion::Lambertian lambertian( &ctx );
        lambertian.setReflectance( legion::Color(  1.0f, 1.0f, 1.0f ) );
       
        legion::Sphere sphere( &ctx );
        sphere.setCenter( legion::Vector3( 1.0f, 0.0f, -5.0f ) );
        sphere.setSurface( &lambertian );
        ctx.addGeometry( &sphere );
        
        legion::Parallelogram pgram( &ctx );
        pgram.setAnchorUV( 
            legion::Vector3( -10.0f, -1.0f,   6.0f ),
            legion::Vector3(  20.0f,  0.0f,   0.0f ),
            legion::Vector3(   0.0f,  0.0f, -20.0f )
            );
        pgram.setSurface( &lambertian );
        ctx.addGeometry( &pgram );

        legion::DiffuseEmitter emitter( &ctx );
        emitter.setRadiance( legion::Color(  0.9f, 0.7f, 0.2f ) );

        legion::Sphere light( &ctx );
        light.setCenter( legion::Vector3( -1.0f, 0.0f, -5.0f ) );
        light.setSurface( &emitter);
        ctx.addGeometry( &light );

        legion::ThinLens cam( &ctx );
        ctx.setCamera( &cam );

        legion::ConstantEnvironment env( &ctx );
        //env.setRadiance( legion::Color( 0.529f, 0.808f, 0.922f )*0.5f );
        env.setRadiance( legion::Color( 0.0f, 0.0f, 0.0f ) );
        ctx.setEnvironment( &env );

        legion::ProgressiveRenderer renderer( &ctx );
        //renderer.setSamplesPerPixel( 1 );
        //renderer.setSamplesPerPass( 1 );
        renderer.setSamplesPerPixel( 32 );
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

