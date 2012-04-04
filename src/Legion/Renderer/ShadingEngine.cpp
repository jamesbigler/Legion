 

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

#include <Legion/Common/Math/Math.hpp>
#include <Legion/Common/Math/Sobol.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Common/Util/Stream.hpp>
#include <Legion/Common/Util/Timer.hpp>
#include <Legion/Common/Util/TypeConversion.hpp>
#include <Legion/Core/Color.hpp>
#include <Legion/Core/Ray.hpp>
#include <Legion/Renderer/Cuda/Shared.hpp>
#include <Legion/Renderer/RayTracer.hpp>
#include <Legion/Renderer/ShadingEngine.hpp>
#include <Legion/Scene/SurfaceShader/ISurfaceShader.hpp>
#include <Legion/Scene/LightShader/ILightShader.hpp>

#include <iomanip>


using namespace legion;

ShadingEngine::ShadingEngine( RayTracer& ray_tracer )
    : m_ray_tracer( ray_tracer ),
      m_shadow_ray_gen( "        Shadow ray gen     " ),
      m_shadow_trace  ( "        Shadow ray trace   " ),
      m_radiance_trace( "        Radiance ray trace " ),
      m_light_loop    ( "        Light loop         " ),
      m_result_copies ( "        Result copies      " ),
      m_max_ray_depth( 2u ),
      m_pass_number( 0u ),
      m_spp( 0u, 0u ),
      m_rnd( 1234321u )
{
}
    

void ShadingEngine::reset()
{
    m_pass_number = 0u;
    m_shadow_ray_gen.reset();
    m_shadow_trace.reset();
    m_radiance_trace.reset();
    m_light_loop.reset();
    m_result_copies.reset();
}


void ShadingEngine::logTimerInfo()
{
    m_shadow_ray_gen.log();
    m_radiance_trace.log();
    m_shadow_trace.log();
    m_light_loop.log();
    m_result_copies.log();
}


void ShadingEngine::shade( std::vector<Ray>& rays,
                           const std::vector<RayScheduler::PixelID>& pixel_ids )
{
    const unsigned num_rays = rays.size();
    m_results.assign( num_rays, Color( 0.0f ) );
    std::vector<Color> ray_attenuation( num_rays, Color( 1.0f ) );

    std::vector<LocalGeometry> lgeom;
    for( unsigned i = 0u; i < m_max_ray_depth; ++i )
    {
        {
            AutoTimerRef<LoopTimerInfo> radiance_ray_timer( m_radiance_trace );
            m_ray_tracer.trace( RayTracer::CLOSEST_HIT, rays );
            m_ray_tracer.join();
        }
        {
            AutoTimerRef<LoopTimerInfo> result_copy( m_result_copies );
            m_ray_tracer.getResults( lgeom );
        }

        shade( rays, lgeom, ray_attenuation, pixel_ids, i );
    }

    m_pass_number++;
}


void ShadingEngine::shade( std::vector<Ray>&           rays,
                           std::vector<LocalGeometry>& local_geom,
                           std::vector<Color>&         ray_attenuation,
                           const std::vector<RayScheduler::PixelID>& pixel_ids,
                           unsigned                    depth )
{
    
    const unsigned num_rays   = rays.size();
    const unsigned dim_offset = depth * Sobol::NUM_DIMS;

    bool have_lights = m_light_set.numLights() != 0;

    if( have_lights )
    {
        // 
        // Calculate shadow rays for all valid intersections 
        //
        {
            AutoTimerRef<LoopTimerInfo> ray_gen_timer( m_shadow_ray_gen );
            m_closures.resize( num_rays );
            m_secondary_rays.resize( num_rays );

            // TODO: make light selection happen at pixel level (not pass)
            const ILightShader* light;
            float               light_select_pdf;
            m_light_set.selectLight( m_rnd(), light, light_select_pdf );

            optix::Context optix_context = m_ray_tracer.getOptixContext();
            optix_context[ "light_id" ]->setUint( light->getID() );

            for( unsigned i = 0; i < num_rays; ++i )
            {
                if( ray_attenuation[i].sum() < 0.0001f ||
                    !local_geom[i].isValidHit() ) 
                {
                    m_secondary_rays[i].setDirection( Vector3( 0.0f ) );
                    continue;
                }
                    
                // Choose a light
                                                                
                // Choose a point on the light by sampling light Area
                const unsigned sobol_idx = pixel_ids[ i ].sobol_index;
                Vector2 seed( 
                        Sobol::gen( sobol_idx, Sobol::DIM_SHADOW_X+dim_offset ),
                        Sobol::gen( sobol_idx, Sobol::DIM_SHADOW_Y+dim_offset )
                        );

                float light_sample_pdf;
                Vector3 on_light;
                light->sample( seed, local_geom[i], on_light, light_sample_pdf );

                // Create ray
                const Vector3 p = local_geom[i].position;
                Vector3 l( on_light - p );
                float dist_to_light = l.normalize();

                const float max_dist = light->isSingular() ?
                                       dist_to_light       :
                                       1e25f; 

                m_secondary_rays[i] = Ray( p, l, max_dist, rays[i].time());
                m_closures[i]       = Closure( light_select_pdf,
                                               light_sample_pdf,
                                               on_light,
                                               light );
            }
        }

        //
        // Trace shadow rays
        //
        {
            AutoTimerRef<LoopTimerInfo> ray_trace_timer( m_shadow_trace );
            m_ray_tracer.trace( RayTracer::ANY_HIT, m_secondary_rays );
            m_ray_tracer.join();
        }

        {
            AutoTimerRef<LoopTimerInfo> result_copy( m_result_copies );
            m_ray_tracer.getResults( m_shadow_results );
        }
    }
     
    //
    // Shade all rays 
    //
    {
        AutoTimerRef<LoopTimerInfo> light_loop_timer( m_light_loop ) ;
        for( unsigned i = 0; i < num_rays; ++i )
        {
            //
            // Light from environment
            //
            if( !local_geom[i].isValidHit() )
            {
                // TODO: Env map queries
                m_results[i] += ray_attenuation[i]*Color( 0.0f );
                // TODO: more efficient to mask out these cases?
                ray_attenuation[i] = Color( 0.0f ); 
                continue;
            }

            const LocalGeometry& lgeom = local_geom[ i ];
            const Vector3 surface_p( lgeom.position );
            const Vector3 w_out = -rays[ i ].direction();

            //
            // Emission
            //

            Color emission( 0.0f ); 
            {
                const ILightShader* lshader = 
                    m_light_set.lookupLight( lgeom.light_id );
                if( lshader && depth == 0 )
                    emission = lshader->emittance( lgeom, -w_out );
            }
            
            //
            // Determine occlusion
            //
            // TODO: handle case where have_lights == false
            Color direct_light( 0.0f );
            Vector3 w_in( 0.0f );

            const ILightShader* light = m_closures[i].light;
            const bool singular       = light->isSingular();
            const bool shadow_hit     = m_shadow_results[i].isValidHit();
            if( singular && !shadow_hit )
            {
                const Vector3 light_p( m_closures[i].light_point );
                float select_pdf = m_closures[i].light_select_pdf;
                
                w_in = light_p - surface_p;
                float dist = w_in.normalize();

                direct_light = light->emittance( LocalGeometry(), w_in ) /
                               ( select_pdf * dist * dist ); 
            }
            else if( shadow_hit && m_shadow_results[i].light_id == (int)light->getID() ) 
            {
                const Vector3 light_p( m_shadow_results[i].position );
                float select_pdf = m_closures[i].light_select_pdf;
                float sample_pdf = m_shadow_results[i].light_pdf; 
                
                w_in = normalize( light_p - surface_p );

                direct_light = light->emittance( LocalGeometry(), w_in )  /
                               ( select_pdf*sample_pdf ); 

                /*
                LLOG_INFO << "light_p: " << light_p << " selpdf: " << select_pdf << " sampdf: " << sample_pdf
                          << " L: " << direct_light;
                          */
            }

            // Evaluate bsdf
            const ISurfaceShader* shader = m_surface_shaders[lgeom.material_id];
            if( !shader )
            {
                LLOG_ERROR << "no shader found for id: " << lgeom.material_id;
                m_results[ i ] = Color( 0.0f, 1.0f, 0.0f ); 
                continue;
            }
            
            float ns_dot_w_in = fmaxf( 0.0f, dot( lgeom.shading_normal, w_in ));

            Color bsdf_val = shader->evaluateBSDF( w_out, lgeom, w_in ) *
                             ns_dot_w_in;

            m_results[i] += emission + direct_light * bsdf_val * ray_attenuation[i];
            
            Vector3 new_w_in;
            Color f_over_pdf;
            
            Vector2 bsdf_seed =
              Vector2( Sobol::gen( pixel_ids[ i ].sobol_index, Sobol::DIM_BSDF_X + depth ),
                       Sobol::gen( pixel_ids[ i ].sobol_index, Sobol::DIM_BSDF_Y + depth) );

            shader->sampleBSDF( bsdf_seed, w_out, lgeom, new_w_in, f_over_pdf);
            ray_attenuation[i] *= f_over_pdf * fabs( dot(lgeom.geometric_normal,
                                                         new_w_in ) );
            rays[i] = Ray( surface_p,
                           new_w_in, 
                           1e15f, 
                           rays[i].time() );
        }
    }
}


const ShadingEngine::Results& ShadingEngine::getResults()const
{
    return m_results;
}


void ShadingEngine::addSurfaceShader( const ISurfaceShader* shader )
{
    LLOG_INFO << __func__ << ": adding surface shader " << shader->getID();
    m_surface_shaders[ shader->getID() ] = shader;
}


void ShadingEngine::addLight( const ILightShader* shader )
{
    LLOG_INFO << __func__ << ": adding light shader " << shader->getID();
    m_light_set.addLight( shader );
}


void ShadingEngine::setMaxRayDepth( unsigned max_depth )
{
    m_max_ray_depth = max_depth;
}


unsigned ShadingEngine::maxRayDepth()const
{
    return m_max_ray_depth;
}



void ShadingEngine::setSamplesPerPixel( const Index2& spp  )
{
    m_spp = spp;
}

