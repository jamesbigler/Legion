
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

#include <Legion/Renderer/OptiXScene.hpp>
#include <Legion/Common/Util/Image.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Core/Exception.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Scene/Camera/ICamera.hpp>
#include <Legion/Scene/Geometry/IGeometry.hpp>
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



namespace
{
    /*
    std::string ptxFilename( const std::string& cuda_filename )
    {
        return "cuda_compile_ptx_generated_" + cuda_filename + ".ptx";
        //return PTX_DIR + "/cuda_compile_ptx_generated_" + cuda_filename + ".ptx";
    }
    */
}


OptiXScene::OptiXScene()
    : m_optix_context( optix::Context::create() ),
      m_program_mgr( m_optix_context ),
      m_camera( 0 )
{
    m_program_mgr.addPath( PTX_DIR );

    try
    {
        m_optix_context->setEntryPointCount( 1u );
        m_optix_context->setRayTypeCount( 2u );

        m_output_buffer = m_optix_context->createBuffer( 
                              RT_BUFFER_OUTPUT,
                              RT_FORMAT_FLOAT4,
                              512u, 512u );
        m_optix_context[ "legion_output_buffer" ]->set( m_output_buffer );

        m_top_group = m_optix_context->createGeometryGroup();
        m_top_group->setAcceleration( 
                m_optix_context->createAcceleration( "Sbvh", "Bvh" ) ); 
        m_optix_context[ "legion_top_group" ]->set( m_top_group );
        

        // TODO: create optix initialization func (default material, 
        // context, etc)
        m_camera_program = m_program_mgr.get( "Camera" );
        m_optix_context->setRayGenerationProgram( 0, m_camera_program );

        // Create default material TODO: from Normal class object
        m_default_mtl = m_optix_context->createMaterial();
        m_default_mtl->setClosestHitProgram(
                RADIANCE_RAY_TYPE,
                m_program_mgr.get( "Normal", "Normal.ptx", "normalClosestHit" )
                );
        m_default_mtl->setAnyHitProgram(
                SHADOW_RAY_TYPE, 
                m_program_mgr.get( "Normal", "Normal.ptx", "normalAnyHit" )
                );


    }
    OPTIX_CATCH_RETHROW;

}


OptiXScene::~OptiXScene()
{

}


void OptiXScene::renderPass( const Index2& min,
                             const Index2& max,
                             unsigned spp )
{
   
    try
    {
        m_optix_context->launch( 0, 512, 512 );
        const std::string filename = "test.exr";
        writeOpenEXR( filename, 512, 512, 4,
                      static_cast<float*>( m_output_buffer->map() ) );
        m_output_buffer->unmap();
    }
    OPTIX_CATCH_RETHROW;
}


optix::Buffer OptiXScene::getOutputBuffer()
{
    return m_output_buffer;
}


void OptiXScene::setCamera( ICamera* camera )
{
    m_camera = camera;
    try
    {
        optix::Program create_ray = 
            m_program_mgr.get( camera->name(),
                               std::string( camera->name() ) + ".ptx",
                               camera->createRayFunctionName() );

        VariableContainer vc( create_ray.get() );
        camera->setVariables( vc );
        m_camera_program[ "legionCameraCreateRay" ]->set( create_ray );
    }
    OPTIX_CATCH_RETHROW;
}


void OptiXScene::setFilm( IFilm* film )
{
    LEGION_TODO();
}


void OptiXScene::addGeometry( IGeometry* geometry )
{
    m_geometry.push_back( geometry );
    try
    {
        // Load the geometry programs
        optix::Program intersect = 
            m_program_mgr.get( geometry->name(),
                               std::string( geometry->name() ) + ".ptx",
                               geometry->intersectionName() );

        optix::Program bbox = 
            m_program_mgr.get( geometry->name(),
                               std::string( geometry->name() ) + ".ptx",
                               geometry->boundingBoxName() );

        // Create optix Geometry 
        optix::Geometry optix_geometry = m_optix_context->createGeometry();
        optix_geometry->setPrimitiveCount( geometry->numPrimitives() );
        optix_geometry->setIntersectionProgram( intersect ); 
        optix_geometry->setBoundingBoxProgram( bbox ); 

        VariableContainer vc( optix_geometry.get() );
        geometry->setVariables( vc );

        // Create optix Material
    
        // Create optix GeometryInstance
        optix::GeometryInstance optix_geometry_instance =
            m_optix_context->createGeometryInstance();
        optix_geometry_instance->setGeometry( optix_geometry );
        optix_geometry_instance->setMaterialCount( 1u );
        optix_geometry_instance->setMaterial( 0, m_default_mtl );
        optix_geometry_instance->setGeometry( optix_geometry );

        m_top_group->setChildCount( m_top_group->getChildCount()+1u );
        m_top_group->setChild( 
                m_top_group->getChildCount()-1,
                optix_geometry_instance
                );

    }
    OPTIX_CATCH_RETHROW;
}


void OptiXScene::addLight( ILight* light )
{
    LEGION_TODO();
}



void OptiXScene::clearScene()
{
    LEGION_TODO();
}
