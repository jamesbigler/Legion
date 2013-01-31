
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
        LLOG_INFO << "galileo ward scene ...";
        
        // Parameters params;
        // params.add( "Kd", legion::Color( 0.5f, 0.5f, 0.5f ) );
        // legion::createSurfaceShader( "Lambertian", "material", params );
        legion::Ward yellow_yogurt( &ctx );
        yellow_yogurt.setReflectance( legion::Color( 0.9f, 0.83f, 0.46f ) );
       
        legion::Ward blue_yogurt( &ctx );
        blue_yogurt.setReflectance( legion::Color( 0.53f, 0.64f, 0.81f ) );
        
        legion::Ward chrome( &ctx );
        chrome.setReflectance( legion::Color( 0.0f, 0.0f, 0.0f ) );

        legion::Lambertian matte_white( &ctx );
        matte_white.setReflectance( legion::Color( 1.0f, 1.0f, 1.0f ) );

        legion::Sphere chrome_sphere( &ctx );
        chrome_sphere.setCenter( legion::Vector3( -0.25f, 0.0f, 0.0f ) );
        chrome_sphere.setSurface( &chrome );
        ctx.addGeometry( &chrome_sphere );
        
        legion::Sphere yellow_sphere( &ctx );
        yellow_sphere.setCenter( legion::Vector3( -0.5f, 0.0f, 2.5f ) );
        yellow_sphere.setSurface( &yellow_yogurt );
        ctx.addGeometry( &yellow_sphere );
        
        legion::Sphere blue_sphere( &ctx );
        blue_sphere.setCenter( legion::Vector3( 2.0f, 0.0f, 1.5f ) );
        blue_sphere.setSurface( &blue_yogurt );
        ctx.addGeometry( &blue_sphere );
        
        
        legion::Parallelogram pgram( &ctx );
        pgram.setAnchorUV( 
            legion::Vector3(   50.0f, -1.0f,   50.0f ),
            legion::Vector3(    0.0f,  0.0f, -100.0f ),
            legion::Vector3( -100.0f,  0.0f,    0.0f )
            );
        pgram.setSurface( &matte_white );
        ctx.addGeometry( &pgram );

        legion::DiffuseEmitter emitter( &ctx );
        emitter.setRadiance( legion::Color(  4.0f, 4.0f, 4.0f ) );

        /*
        legion::Sphere light( &ctx );
        light.setCenter( legion::Vector3( -0.5f, 5.0f, 3.5f ) );
        */

        legion::Parallelogram light( &ctx );
        light.setAnchorUV( 
            legion::Vector3(  2.0f, 5.0f, 2.0f ),
            legion::Vector3(  0.0f, 0.0f, 3.0f ),
            legion::Vector3( -3.0f, 0.0f, 0.0f )
            );
        light.setSurface( &emitter);
        ctx.addGeometry( &light );

        legion::ThinLens cam( &ctx );
        cam.setFocalDistance( 2.75f );
        cam.setViewPlane( -1.0f, 1.0f, -0.75f, 0.75f );
        cam.setCameraToWorld( 
            legion::Matrix::lookAt( 
              legion::Vector3( 5.0f,  5.0f, 5.0f ),
              legion::Vector3( 0.0f, -1.0f, 1.0f ),
              legion::Vector3( 0.0f,  1.0f, 0.0f ) ) );
        ctx.setCamera( &cam );

        legion::ConstantEnvironment env( &ctx );
        env.setRadiance( legion::Color( 0.0f, 0.0f, 0.0f ) );
        ctx.setEnvironment( &env );

        legion::ProgressiveRenderer renderer( &ctx );
        renderer.setResolution( legion::Index2( 800u, 600u ) );
        //renderer.setSamplesPerPixel( 1 );
        //renderer.setSamplesPerPass( 1 );
        renderer.setSamplesPerPixel( 32*4 );
        renderer.setSamplesPerPass( 8 );
        ctx.setRenderer( &renderer );

        legion::ImageFileDisplay display( &ctx, "galileo_ward.exr" );
        renderer.setDisplay( &display );
        
        ctx.render();
    }
    catch( legion::Exception& e )
    {
        LLOG_ERROR << e.what();
    }
}

