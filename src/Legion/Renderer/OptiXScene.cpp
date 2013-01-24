
// Copyright (C) 2011 R. Keith Morley
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// (MIT/X11 License)

#include <Legion/Common/Util/Image.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Core/Context.hpp>
#include <Legion/Core/Exception.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Objects/Camera/ICamera.hpp>
#include <Legion/Objects/Geometry/IGeometry.hpp>
#include <Legion/Objects/Renderer/IRenderer.hpp>
#include <Legion/Objects/Surface/ISurface.hpp>
#include <Legion/Renderer/OptiXScene.hpp>
#include <config.hpp>

using namespace legion;

#define OPTIX_CATCH_RETHROW                                                    \
    catch ( optix::Exception& e )                                              \
    {                                                                          \
        throw legion::Exception( std::string("OPTIX_EXCEPTION: ")+e.what() );  \
    }                                                                          \
    catch ( std::exception& e )                                                \
    {                                                                          \
        throw legion::Exception( std::string("OPTIX_EXCEPTION: ")+e.what() );  \
    }                                                                          \
    catch (...)                                                                \
    {                                                                          \
        throw legion::Exception( std::string("OPTIX_EXCEPTION: unknown") );    \
    }

#define OPTIX_CATCH_WARN                                                       \
    catch ( optix::Exception& e )                                              \
    {                                                                          \
        LLOG_WARN << "OPTIX_EXCEPTION: " << e.what();                          \
    }                                                                          \
    catch ( std::exception& e )                                                \
    {                                                                          \
        LLOG_WARN << "OPTIX_EXCEPTION: " << e.what();                          \
    }                                                                          \
    catch (...)                                                                \
    {                                                                          \
        LLOG_WARN << "OPTIX_EXCEPTION: Unknown";                               \
    }


OptiXScene::OptiXScene()
    : m_optix_context( optix::Context::create() ),
      m_program_mgr( m_optix_context ),
      m_renderer( 0u ),
      m_camera( 0u ),
      m_film( 0u )
{
    m_program_mgr.addPath( PTX_DIR );
    initializeOptixContext();
}




OptiXScene::~OptiXScene()
{
    m_optix_context->destroy();
}
    

void OptiXScene::setRenderer( IRenderer* renderer )
{
    m_raygen_program = 
            m_program_mgr.get( renderer->name(),
                               std::string( renderer->name() ) + ".ptx",
                               renderer->rayGenProgramName() );

    m_optix_context->setRayGenerationProgram( 0, m_raygen_program );

    m_renderer = renderer;
}


void OptiXScene::setCamera( ICamera* camera )
{
    m_camera = camera;
    try
    {
        m_create_ray_program = 
            m_program_mgr.get( camera->name(),
                               std::string( camera->name() ) + ".ptx",
                               camera->createRayFunctionName() );

        m_optix_context[ "legionCameraCreateRay" ]->set( m_create_ray_program );
    }
    OPTIX_CATCH_RETHROW;
}


void OptiXScene::setFilm( IFilm* film )
{
    LLOG_INFO << "Film: " << film;
    LEGION_TODO();
}


void OptiXScene::addLight( ILight* light )
{
    LLOG_INFO << "Light: " << light;
    LEGION_TODO();
}



void OptiXScene::addGeometry( IGeometry* geometry )
{
    // TODO: break this into separate creation functions, eg, createMaterial,
    //       createGeometry, createLight
    try
    {
        // Load the geometry programs
        optix::Program intersect = 
            m_program_mgr.get( geometry->name(),
                               std::string( geometry->name() ) + ".ptx",
                               geometry->intersectionFunctionName() );

        optix::Program bbox = 
            m_program_mgr.get( geometry->name(),
                               std::string( geometry->name() ) + ".ptx",
                               geometry->boundingBoxFunctionName() );

        //
        // Create optix Geometry 
        //
        optix::Geometry optix_geometry = m_optix_context->createGeometry();
        optix_geometry->setPrimitiveCount( geometry->numPrimitives() );
        optix_geometry->setIntersectionProgram( intersect ); 
        optix_geometry->setBoundingBoxProgram( bbox ); 

        //
        // Create optix GeometryInstance
        //
        optix::GeometryInstance optix_geometry_instance =
            m_optix_context->createGeometryInstance();
        optix_geometry_instance->setGeometry( optix_geometry );
        optix_geometry_instance->setMaterialCount( 1u );
        optix_geometry_instance->setMaterial( 0, m_default_mtl );
        optix_geometry_instance->setGeometry( optix_geometry );
        
        //
        // Create optix Material
        //
        ISurface* surface = geometry->getSurface();
        const std::string surface_name( surface->name() );
        const std::string surface_ptx( surface_name + ".ptx" );
       
        optix::Program evaluate_bsdf = 
            m_program_mgr.get( surface_name,
                               surface_ptx,
                               surface->evaluateBSDFFunctionName() );
        optix_geometry_instance[ "legionSurfaceEvaluateBSDF" ]->set( 
                evaluate_bsdf );
        
        optix::Program sample_bsdf = 
            m_program_mgr.get( surface_name,
                               surface_ptx,
                               surface->sampleBSDFFunctionName() );
        optix_geometry_instance[ "legionSurfaceSampleBSDF" ]->set( 
                sample_bsdf );
        
        optix::Program pdf = 
            m_program_mgr.get( surface_name,
                               surface_ptx,
                               surface->pdfFunctionName() );
        optix_geometry_instance[ "legionSurfacePDF" ]->set( pdf );
        
        const std::string emission_func_name = surface->emissionFunctionName();
        optix::Program emission = 
            m_program_mgr.get( surface_name,
                               surface_ptx,
                               emission_func_name );
        optix_geometry_instance[ "legionSurfaceEmission" ]->set( emission );

        //
        // create Light
        //
        if( emission_func_name != "nullSurfaceEmission" )
        {
            optix::Program sample = 
                m_program_mgr.get( geometry->name(),
                                   std::string( geometry->name() ) + ".ptx",
                                   geometry->sampleFunctionName() );

            // TODO: callable buffer once implemented, move variablecontainer
            //       setting to sync()
            m_optix_context[ "legionLightSample"   ]->set( sample );
            m_optix_context[ "legionLightEmission" ]->set( emission );
            VariableContainer vc0( sample.get() );
            geometry->setVariables( vc0 );
            VariableContainer vc1( emission.get() );
            surface->setVariables( vc1 );
            
        }

        m_top_group->setChildCount( m_top_group->getChildCount()+1u );
        m_top_group->setChild( 
                m_top_group->getChildCount()-1,
                optix_geometry_instance
                );

        m_geometry.insert(std::make_pair( geometry, optix_geometry_instance ) );
    }
    OPTIX_CATCH_RETHROW;
}


void OptiXScene::sync()
{
    // TODO: have defaults objects for all of these
    LEGION_ASSERT( m_renderer );
    LEGION_ASSERT( m_camera );
   
    //
    // Update optix variables for all objects
    //
    VariableContainer renderer_vc( m_raygen_program.get() );
    m_renderer->setVariables( renderer_vc );
        
    VariableContainer vc( m_create_ray_program.get() );
    m_camera->setVariables( vc );
        
    for( GeometryMap::iterator it = m_geometry.begin();
         it != m_geometry.end();
         ++it )
    {
        IGeometry*              geometry          = it->first; 
        ISurface*               surface           = geometry->getSurface();
        optix::GeometryInstance geometry_instance = it->second; 

        VariableContainer vc( geometry_instance.get() );
        geometry->setVariables( vc );
        surface->setVariables( vc );

    }

    /*
    try
    {
        m_output_buffer->setSize( m_resolution.x(), m_resolution.y() );
        m_optix_context->launch( 0, m_resolution.x(), m_resolution.y() );
        const std::string filename = "test.exr";
        writeOpenEXR( filename, m_resolution.x(), m_resolution.y(), 4,
                      static_cast<float*>( m_output_buffer->map() ) );
        m_output_buffer->unmap();
    }
    OPTIX_CATCH_RETHROW;
    */
}


void OptiXScene::initializeOptixContext()
{
    try
    {
        m_optix_context->setEntryPointCount( 1u ); // Single raygen
        m_optix_context->setRayTypeCount   ( 2u ); // Radiance, shadow query

        m_optix_context[ "legion_radiance_ray_type" ]->setUint( RADIANCE_TYPE );
        m_optix_context[ "legion_shadow_ray_type"   ]->setUint( SHADOW_TYPE );

        // Create the top level scene group
        m_top_group = m_optix_context->createGeometryGroup();
        m_top_group->setAcceleration( 
                m_optix_context->createAcceleration( "Sbvh", "Bvh" ) ); 
        m_optix_context[ "legion_top_group" ]->set( m_top_group );
        
        m_default_mtl = m_optix_context->createMaterial();
        m_default_mtl->setClosestHitProgram(
                RADIANCE_TYPE,
                m_program_mgr.get( 
                    "Surface", 
                    "Surface.ptx", 
                    "legionClosestHit" )
                );
        m_default_mtl->setAnyHitProgram(
                SHADOW_TYPE, 
                m_program_mgr.get( 
                    "Surface", 
                    "Surface.ptx", 
                    "legionAnyHit" )
                );
    }
    OPTIX_CATCH_RETHROW;
}
