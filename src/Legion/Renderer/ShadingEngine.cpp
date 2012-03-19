 

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
      m_max_ray_depth( 2 ),
      m_rnd( 1234321u )
{
}
    

void ShadingEngine::reset()
{
    m_shadow_ray_gen.reset();
    m_shadow_trace.reset();
    m_radiance_trace.reset();
    m_light_loop.reset();
}


void ShadingEngine::logTimerInfo()
{
    m_shadow_ray_gen.log();
    m_shadow_trace.log();
    m_radiance_trace.log();
    m_light_loop.log();
}


void ShadingEngine::shade( std::vector<Ray>& rays )
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
            m_ray_tracer.getResults( lgeom );
        }

        shade( rays, lgeom, ray_attenuation, i == 0u );
    }
}


void ShadingEngine::shade( std::vector<Ray>&           rays,
                           std::vector<LocalGeometry>& local_geom,
                           std::vector<Color>&         ray_attenuation,
                           bool                        count_emission )
{
    
    const unsigned num_rays = rays.size();

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

            for( unsigned i = 0; i < num_rays; ++i )
            {
                if( ray_attenuation[i].sum() < 0.0001f ||
                    !local_geom[i].isValidHit() ) 
                {
                    m_secondary_rays[i].setDirection( Vector3( 0.0f ) );
                    continue;
                }
                    
                // Choose a light
                const ILightShader* light;
                float               light_select_pdf;
                float               sample_seed = static_cast<float>( m_rnd() );
                m_light_set.selectLight( sample_seed,
                                         light,
                                         light_select_pdf );
                                                                
                // Choose a point on the light by sampling light Area
                float light_sample_pdf;
                Vector3 on_light;
                light->sample( Vector2( m_rnd(), m_rnd() ),
                               on_light,
                               light_sample_pdf );

                // Create ray
                const Vector3 p = local_geom[i].position;
                Vector3 l( on_light - p );
                float dist = l.normalize();

                const float max_dist = light->isSingular() ? dist : dist + 1.0f;
                m_secondary_rays[i]  = Ray( p, l, max_dist, rays[i].time());
                m_closures[i]        = Closure( on_light, light );
            }
        }

        //
        // Trace shadow rays
        //
        {
            AutoTimerRef<LoopTimerInfo> ray_trace_timer( m_shadow_trace );
            m_ray_tracer.trace( RayTracer::CLOSEST_HIT, m_secondary_rays );
            m_ray_tracer.join();
            m_ray_tracer.getResults( m_shadow_results );
        }
    }
     
    //
    // Shade all rays 
    // TODO: now we need to convert from pdf w/ respect to area to pdf w/
    //       respect to solid angle and shade. see pbr2 page 717
    //
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
                if( lshader && count_emission )
                    emission = lshader->emittance( lgeom, -w_out );
            }
            
            //
            // Determine occlusion
            //
            // TODO: handle case where have_lights == false
            const ILightShader* light = m_closures[i].light;
            const Vector3 light_p( m_closures[i].light_point );
            const Vector3 w_in( normalize( light_p - surface_p ) );
            const bool is_singular_light = light->isSingular();
            Color direct_light( 0.001f );
            if( ( is_singular_light && !m_shadow_results[i].isValidHit() ) || 
                ( !is_singular_light && m_shadow_results[i].light_id ==
                  static_cast<int>( light->getID() ) ) )
            {
                // we have unoccluded light
                direct_light = light->emittance( m_shadow_results[ i ], w_in ); 
            }
            {
                /*
                LLOG_INFO << "singular: " << is_singular_light
                          << " shadow_id: " << m_shadow_results[i].light_id
                          << " new_id: " << light->getID()
                          << " shadow_hit_p: " << m_shadow_results[i].position
                          << " shading_p: " << surface_p; 
                          */
            }


            // Evaluate bsdf
            const ISurfaceShader* shader = m_surface_shaders[ lgeom.material_id ];
            if( !shader )
            {
                LLOG_INFO << "no shader found for id: " << lgeom.material_id;
                m_results[ i ] = Color( 0.0f, 1.0f, 0.0f ); 
                continue;
            }
            
            const float ns_dot_w_in = fmaxf( 0.0f, dot( lgeom.geometric_normal,
                                                        w_in ) );

            const float inv_dist2 = 1.0f / (surface_p-light_p).lengthSquared();
            Color bsdf_val = shader->evaluateBSDF( w_out, lgeom, w_in ) *
                             ns_dot_w_in;

            m_results[i] += emission + direct_light * inv_dist2 * bsdf_val *
                            ray_attenuation[i];
            
            //LLOG_INFO << std::fixed << i/10 << "," << i%10 << " " << m_results[i] << " d: " << direct_light << " dist: " << inv_dist2 << " b: " << bsdf_val << " e: " << emission ;


            Vector3 new_w_in;
            Color f_over_pdf;
            const Vector2 bsdf_seed( m_rnd(), m_rnd() );
            shader->sampleBSDF( bsdf_seed, w_out, lgeom, new_w_in, f_over_pdf);
            ray_attenuation[i] *= f_over_pdf * fabs( dot( lgeom.geometric_normal,
                                                          new_w_in ) );
            rays[i] = Ray( surface_p + 0.0001f*lgeom.geometric_normal,
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

