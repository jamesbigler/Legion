 

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



using namespace legion;

ShadingEngine::ShadingEngine( RayTracer& ray_tracer )
    : m_ray_tracer( ray_tracer ),
      m_shadow_ray_gen  ( "        Shadow ray gen   " ),
      m_shadow_ray_trace( "        Shadow ray trace " ),
      m_light_loop      ( "        Light loop       " )
{
}
    

void ShadingEngine::reset()
{
    m_shadow_ray_gen.reset();
    m_shadow_ray_trace.reset();
    m_light_loop.reset();
}


void ShadingEngine::logTimerInfo()
{
    m_shadow_ray_gen.log();
    m_shadow_ray_trace.log();
    m_light_loop.log();
}


void ShadingEngine::shade( const std::vector<Ray>& rays,
                           const std::vector<LocalGeometry>& local_geom )
{
    assert( rays.size() == local_geom.size() );

    const unsigned num_rays = rays.size();

    m_results.resize( num_rays );

    std::vector<LocalGeometry> shadow_results; // TODO: make persistant 
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
                if( !local_geom[i].isValidHit() )
                {
                    m_secondary_rays[i].setDirection( Vector3( 0.0f ) );
                    continue;
                }
                    
                // Choose a light
                const ILightShader* light;
                float               light_select_pdf;
                float               sample_seed = static_cast<float>( drand48() );
                m_light_set.selectLight( sample_seed,
                                         light,
                                         light_select_pdf );
                                                                
                // Choose a point on the light by sampling light Area
                float light_sample_pdf;
                Vector3 on_light;
                light->sample( Vector2( drand48(), drand48() ),
                               on_light,
                               light_sample_pdf );

                // Create ray
                const Vector3 p = toVector3( local_geom[i].position );
                Vector3 l( on_light - p );
                float dist = l.normalize();

                // TODO: this might be too inefficient to trace to max_dist
                // TODO: can do away with Closure::on_light??
                const float max_dist = light->isSingular() ? dist : 1e15f;
                m_secondary_rays[i]  = Ray( p, l, max_dist, rays[i].time());
                m_closures[i]        = Closure( on_light, light );
            }
        }

        //
        // Trace shdow rays
        //
        {
            AutoTimerRef<LoopTimerInfo> ray_trace_timer( m_shadow_ray_trace );
            m_ray_tracer.trace( RayTracer::CLOSEST_HIT, m_secondary_rays );
            m_ray_tracer.join();  // TODO: REMOVE:  maximize trace/shade overlap
            m_ray_tracer.getResults( shadow_results );
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
                m_results[i] = Color( 0.0f ); // TODO: Env map queries
                continue;
            }

            //
            // Determine occlusion
            //
            const ILightShader* light = m_closures[i].light;
            const bool is_singular_light = light->isSingular();
            if( is_singular_light && !shadow_results[i].isValidHit() )
            {
                // we have unoccluded light
            }
            else if( !is_singular_light &&
                     shadow_results[i].light_id == static_cast<int>( light->getID() ) )
            {
                // we have unoccluded light
            }
            else
            {
                // Light sample is occluded
                m_results[ i ] = Color( 0.0f ); // TODO remove this
                continue;
            }

            const LocalGeometry& lgeom = local_geom[ i ];
            const Vector3 surface_p( toVector3( lgeom.position ) );
            const Vector3 light_p( m_closures[i].light_point );
            const Vector3 w_in( normalize( light_p - surface_p ) );
            const Vector3 w_out = -rays[ i ].direction();

            // 
            Color emittance = light->emittance( shadow_results[ i ], w_in ); 

            // Evaluate bsdf
            const ISurfaceShader* shader = m_shaders[ lgeom.material_id ];
            if( !shader )
            {
                LLOG_INFO << "no shader found for id: " << lgeom.material_id;
                m_results[ i ] = Color( 0.0f, 1.0f, 0.0f ); 
                continue;
            }
            
            float inv_dist2 = 1.0f / ( surface_p - light_p ).lengthSquared();
            Color bsdf_val = shader->evaluateBSDF( w_out, lgeom, w_in );
            m_results[i] = emittance * bsdf_val * inv_dist2;

            //LLOG_INFO << "   bsdf: " << m_results[i] << " light: " << light;
        }
    }
}

const ShadingEngine::Results& ShadingEngine::getResults()const
{
    return m_results;
}


void ShadingEngine::addSurfaceShader( const ISurfaceShader* shader )
{
    LLOG_INFO << __func__ << ": adding shader " << shader->getID();
    m_shaders[ shader->getID() ] = shader;
}

void ShadingEngine::addLight( const ILightShader* shader )
{
    m_light_set.addLight( shader );
}

