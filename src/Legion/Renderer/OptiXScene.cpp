
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
#include <Legion/Objects/Environment/IEnvironment.hpp>
#include <Legion/Objects/Renderer/IRenderer.hpp>
#include <Legion/Objects/Surface/ISurface.hpp>
#include <Legion/Objects/Texture/ITexture.hpp>
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
      m_film( 0u ),
      m_num_lights( 0u )
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
            m_program_mgr.get( std::string( renderer->name() ) + ".ptx",
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
            m_program_mgr.get( std::string( camera->name() ) + ".ptx",
                               camera->createRayFunctionName() );

        m_optix_context[ "legionCameraCreateRay" ]->set( m_create_ray_program );
    }
    OPTIX_CATCH_RETHROW;
}


void OptiXScene::setEnvironment( IEnvironment* environment )
{
    m_environment = environment;

    try
    {
        // If we have not created a miss program, create now
        if( !m_environment_program )
        {
            m_environment_program = 
                m_program_mgr.get( "Environment.ptx", "legionEnvironment" );
            m_optix_context->setMissProgram( RADIANCE_TYPE, m_environment_program );
        }

        // Get environment programs
        m_environment_miss_evaluate = 
            m_program_mgr.get( std::string( environment->name() ) + ".ptx",
                               environment->missEvaluateFunctionName() );

        m_environment_light_evaluate = 
            m_program_mgr.get( std::string( environment->name() ) + ".ptx",
                               environment->lightEvaluateFunctionName() );

        m_environment_sample = 
            m_program_mgr.get( std::string( environment->name() ) + ".ptx",
                    environment->sampleFunctionName() );

        // Set miss program evaluator
        m_optix_context[ "legionEnvironmentMissEvaluate" ]->set( 
                m_environment_miss_evaluate 
                );

        // Add light
        if( m_num_lights <= MAX_LIGHTS )
        {
            std::stringstream ss;
            ss << "legionLightSample_" << m_num_lights;
            m_optix_context[ ss.str() ]->set( m_environment_sample);

            ss.str( "" );
            ss << "legionLightEvaluate_" << m_num_lights;
            m_optix_context[ ss.str() ]->set( m_environment_light_evaluate );

            ++m_num_lights;
        }
    }
    OPTIX_CATCH_RETHROW;
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
            m_program_mgr.get( std::string( geometry->name() ) + ".ptx",
                               geometry->intersectionFunctionName() );

        optix::Program bbox = 
            m_program_mgr.get( std::string( geometry->name() ) + ".ptx",
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
        optix_geometry_instance->setGeometry( optix_geometry );
        
        //
        // Create optix Material
        //
        optix::Material material = m_optix_context->createMaterial();
        optix_geometry_instance->setMaterial( 0, material );
        material->setClosestHitProgram( RADIANCE_TYPE, m_closest_hit_program );
        material->setAnyHitProgram( SHADOW_TYPE, m_any_hit_program );

        ISurface* surface = geometry->getSurface();
        const std::string surface_name( surface->name() );
        const std::string surface_ptx( surface_name + ".ptx" );
       
        optix::Program evaluate_bsdf = 
            m_program_mgr.get( surface_ptx,
                               surface->evaluateBSDFFunctionName(),
                               false );
        material[ "legionSurfaceEvaluateBSDF" ]->set( evaluate_bsdf );
        
        optix::Program sample_bsdf = 
            m_program_mgr.get( surface_ptx,
                               surface->sampleBSDFFunctionName(),
                               false );
        material[ "legionSurfaceSampleBSDF" ]->set( sample_bsdf );
        
        optix::Program pdf = 
            m_program_mgr.get( surface_ptx,
                               surface->pdfFunctionName(),
                               false );
        material[ "legionSurfacePDF" ]->set( pdf );
        
        const std::string emission_func_name = surface->emissionFunctionName();
        optix::Program emission = 
            m_program_mgr.get( 
                    surface_ptx, 
                    emission_func_name,
                    false
                    );
        material[ "legionSurfaceEmission" ]->set( emission );

        optix::Program light_pdf = 
          m_program_mgr.get( std::string( geometry->name() ) + ".ptx",
                             geometry->pdfFunctionName() );
        optix_geometry_instance[ "legionLightPDF" ]->set( light_pdf );


        //
        // Add Light
        //
        if( emission_func_name != "nullSurfaceEmission" && 
            m_num_lights <= MAX_LIGHTS )
        {
            optix::Program sample = 
                m_program_mgr.get( std::string( geometry->name() ) + ".ptx",
                                   geometry->sampleFunctionName(),
                                   false );

            std::stringstream ss;
            ss << "legionLightSample_" << m_num_lights;
            m_optix_context[ ss.str() ]->set( sample );

            ss.str( "" );
            ss << "legionLightEvaluate_" << m_num_lights;
            m_optix_context[ ss.str() ]->set( emission );

            // TODO: move to sync
            VariableContainer vc0( sample.get() );
            geometry->setVariables( vc0 );

            VariableContainer vc1( emission.get() );
            surface->setVariables( vc1 );

            ++m_num_lights;
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


void OptiXScene::addAssetPath( const std::string& path )
{
    m_program_mgr.addPath( path );
}


void OptiXScene::sync()
{
    // TODO: have defaults objects for all of these
    LEGION_ASSERT( m_renderer );
    LEGION_ASSERT( m_camera );
  
    m_optix_context[ "legionLightCount" ]->setUint( m_num_lights );
    //
    // Update optix variables for all objects
    //
    {
        VariableContainer vc( m_raygen_program.get() );
        m_renderer->setVariables( vc );
    }
        
    {
        VariableContainer vc( m_create_ray_program.get() );
        m_camera->setVariables( vc );
    }
        
    if( m_environment_sample )
    {
        VariableContainer vc( m_environment_sample.get() );
        m_environment->setVariables( vc );
    }

    if( m_environment_miss_evaluate )
    {
        VariableContainer vc( m_environment_miss_evaluate.get() );
        m_environment->setVariables( vc );
    }
    
    if( m_environment_light_evaluate )
    {
        VariableContainer vc( m_environment_light_evaluate.get() );
        m_environment->setVariables( vc ); 
    }


    // Set up null functions for unused light variables
    for( unsigned i = m_num_lights; i < MAX_LIGHTS; ++i )
    {
        optix::Program null_light_sample = 
          m_program_mgr.get( "Light.ptx", "nullLightSample" );

        optix::Program null_light_evaluate = 
          m_program_mgr.get( "Light.ptx", "nullLightEvaluate" );

        std::stringstream ss;
        ss << "legionLightSample_" << i;
        m_optix_context[ ss.str() ]->set( null_light_sample );

        ss.str( "" );
        ss << "legionLightEvaluate_" << i;
        m_optix_context[ ss.str() ]->set( null_light_evaluate );
    }


    for( GeometryMap::iterator geom_it = m_geometry.begin();
         geom_it != m_geometry.end();
         ++geom_it )
    {
        IGeometry*              geometry          = geom_it->first; 
        ISurface*               surface           = geometry->getSurface();
        optix::GeometryInstance geometry_instance = geom_it->second; 

        //optix::Geometry optix_geometry = geometry_instance->getGeometry();
        VariableContainer geom_vc( geometry_instance.get() );
        geometry->setVariables( geom_vc );

        optix::Material optix_material = geometry_instance->getMaterial( 0u );

        optix::Program eval_bsdf =
          optix_material[ "legionSurfaceEvaluateBSDF" ]->getProgram();
        optix::Program sample_bsdf =
          optix_material[ "legionSurfaceSampleBSDF"   ]->getProgram();
        optix::Program pdf =
          optix_material[ "legionSurfacePDF"          ]->getProgram();
        //setSurfaceVariables( optix_material,   surface );
        setSurfaceVariables( eval_bsdf,   surface );
        setSurfaceVariables( sample_bsdf, surface );
        setSurfaceVariables( pdf,         surface );
    }
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
        
        m_closest_hit_program = 
            m_program_mgr.get( 
                    "Surface.ptx", 
                    "legionClosestHit"
                    );
                
        m_any_hit_program = 
                m_program_mgr.get( 
                    "Surface.ptx", 
                    "legionAnyHit"
                    );

        m_default_texture_proc1 =
                m_program_mgr.get( 
                    "Texture.ptx", 
                    "legionDefaultTextureProc1"
                    );
        
        m_default_texture_proc2 =
                m_program_mgr.get( 
                    "Texture.ptx", 
                    "legionDefaultTextureProc2"
                    );
        
        m_default_texture_proc4 =
                m_program_mgr.get( 
                    "Texture.ptx", 
                    "legionDefaultTextureProc4"
                    );

    }
    OPTIX_CATCH_RETHROW;
}


template <typename OptiXNode>
void OptiXScene::setSurfaceVariables(
    OptiXNode       node,
    const ISurface* surface
    )
{
    VariableContainer vc( node.get() );
    surface->setVariables( vc );

    const VariableContainer::Textures& textures = vc.getTextures();
    for( VariableContainer::Textures::const_iterator it = textures.begin();
            it != textures.end();
            ++it )
    {
        setTextureVariables( node, it->first, it->second );
    }

    const VariableContainer::Surfaces& surfaces = vc.getSurfaces();
    for( VariableContainer::Surfaces::const_iterator it = surfaces.begin();
            it != surfaces.end();
            ++it )
    {
        setNestedSurfaceVariables( node, it->first, it->second );
    }
}


template <typename OptiXNode>
void OptiXScene::setNestedSurfaceVariables( OptiXNode          node, 
                                            const std::string& name,
                                            const ISurface*    surf )
{
    // TODO: be smarter about nested program assignments -- only assign PDF
    //       callableFunction to parent PDF function

    {
        optix::Program proc =  
            m_program_mgr.get( std::string( surf->name() ) + ".ptx",
                    surf->sampleBSDFFunctionName(),
                    false
                    );
        VariableContainer vc( proc.get() );
        surf->setVariables( vc );
        const VariableContainer::Textures& textures = vc.getTextures();
        for( VariableContainer::Textures::const_iterator it = textures.begin();
             it != textures.end();
             ++it )
        {
            setTextureVariables( proc, it->first, it->second );
        }

        const std::string sample_bsdf_name = name + "_sampleBSDF__";  
        node[ sample_bsdf_name ]->set( proc );
    }

    {
        optix::Program proc =  
            m_program_mgr.get( std::string( surf->name() ) + ".ptx",
                    surf->evaluateBSDFFunctionName(),
                    false
                    );
        VariableContainer vc( proc.get() );
        surf->setVariables( vc );
        const VariableContainer::Textures& textures = vc.getTextures();
        for( VariableContainer::Textures::const_iterator it = textures.begin();
             it != textures.end();
             ++it )
        {
            setTextureVariables( proc, it->first, it->second );
        }
        
        const std::string eval_bsdf_name = name + "_evaluateBSDF__";
        node[ eval_bsdf_name ]->set( proc );
    }

    {
        optix::Program proc =  
            m_program_mgr.get( std::string( surf->name() ) + ".ptx",
                    surf->pdfFunctionName(),
                    false
                    );
        VariableContainer vc( proc.get() );
        surf->setVariables( vc );
        const VariableContainer::Textures& textures = vc.getTextures();
        for( VariableContainer::Textures::const_iterator it = textures.begin();
             it != textures.end();
             ++it )
        {
            setTextureVariables( proc, it->first, it->second );
        }

        const std::string pdf_name = name + "_PDF__";
        node[ pdf_name ]->set( proc );
    }

    {
        optix::Program proc =  
            m_program_mgr.get( std::string( surf->name() ) + ".ptx",
                    surf->emissionFunctionName(),
                    false
                    );
        VariableContainer vc( proc.get() );
        surf->setVariables( vc );
        const VariableContainer::Textures& textures = vc.getTextures();
        for( VariableContainer::Textures::const_iterator it = textures.begin();
             it != textures.end();
             ++it )
        {
            setTextureVariables( proc, it->first, it->second );
        }
        
        const std::string emission_name = name + "_emission__";
        node[ emission_name ]->set( proc );
    }
}


template <typename OptiXNode>
void OptiXScene::setTextureVariables( OptiXNode          node,
                                      const std::string& name,
                                      const ITexture*    tex )
{
    const ITexture::Type type = tex->getType();
    const unsigned val_dim = tex->getValueDim();

    node[ name + "_type__" ]->setUint( tex->getType() );

    const std::string cname = name + "_const__";
    const std::string tname = name + "_texid__";
    const std::string pname = name + "_proc__";

    // TODO: combine some of this logic
    if( type == ITexture::TYPE_CONSTANT )
    {
        float    v[4];
        tex->getConstValue( v );
        switch( val_dim )
        {
            case 1: 
                {
                    node[ cname ]->set1fv( v );
                    node[ pname ]->set( m_default_texture_proc1 );
                    node[ tname ]->setInt( -1 );
                    break;
                }
            case 2: 
                {
                    node[ cname ]->set2fv( v );
                    node[ pname ]->set( m_default_texture_proc2 );
                    node[ tname ]->setInt( -1 );
                    break;
                }
            case 4:
                {
                    node[ cname ]->set4fv( v );
                    node[ pname ]->set( m_default_texture_proc4 );
                    node[ tname ]->setInt( -1 );
                    break;
                }
            default: 
                throw Exception( "Invalid texture result dim: " + name );
        };
    }
    else if( type == ITexture::TYPE_IMAGE )
    {
        switch( val_dim )
        {
            case 1: 
                {
                    node[ cname ]->setFloat( 0.0f );
                    node[ pname ]->set( m_default_texture_proc1 );
                    node[ tname ]->setInt( tex->getTexID() );
                    break;
                }
            case 2: 
                {
                    node[ cname ]->setFloat( 0.0f, 0.0f );
                    node[ pname ]->set( m_default_texture_proc2 );
                    node[ tname ]->setInt( tex->getTexID() );
                    break;
                }
            case 4:
                {
                    node[ cname ]->setFloat( 0.0f, 0.0f, 0.0f, 0.0f );
                    node[ pname ]->set( m_default_texture_proc4 );
                    node[ tname ]->setInt( tex->getTexID() );
                    break;
                }
            default: 
                throw Exception( "Invalid texture result dim: " + name );
        };
    }
    else // ( type == ITexture::TYPE_PROCEDURAL )
    {
        optix::Program proc =  
            m_program_mgr.get( std::string( tex->name() ) + ".ptx",
                    tex->proceduralFunctionName(),
                    false
                    );

        VariableContainer vc( proc.get() );
        tex->setVariables( vc );

        switch( val_dim )
        {
            case 1: 
                {
                    node[ cname ]->setFloat( 0.0f );
                    node[ pname ]->set( proc );
                    node[ tname ]->setInt( -1 );
                    break;
                }
            case 2: 
                {
                    node[ cname ]->setFloat( 0.0f, 0.0f );
                    node[ pname ]->set( proc );
                    node[ tname ]->setInt( -1 );
                    break;
                }
            case 4:
                {
                    node[ cname ]->setFloat( 0.0f, 0.0f, 0.0f, 0.0f );
                    node[ pname ]->set( proc );
                    node[ tname ]->setInt( -1 );
                    break;
                }
            default: 
                throw Exception( "Invalid texture result dim: " + name );
        };
    }
}
