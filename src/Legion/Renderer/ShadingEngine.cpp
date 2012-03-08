 

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
#include <Legion/Common/Util/TypeConversion.hpp>
#include <Legion/Core/Color.hpp>
#include <Legion/Core/Ray.hpp>
#include <Legion/Renderer/Cuda/Shared.hpp>
#include <Legion/Renderer/RayTracer.hpp>
#include <Legion/Renderer/ShadingEngine.hpp>
#include <Legion/Scene/SurfaceShader/ISurfaceShader.hpp>


using namespace legion;

ShadingEngine::ShadingEngine( RayTracer& ray_tracer )
    : m_ray_tracer( ray_tracer )
{
}
    

void ShadingEngine::shade( const std::vector<Ray>& rays,
                           const std::vector<LocalGeometry>& local_geom )
{
    assert( rays.size() == local_geom.size() );
    const unsigned num_rays = rays.size();

    m_results.resize( num_rays );
    m_light_points.resize( num_rays );

    // TODO: dont trace shadow rays for rays that hit background!!!!
    // Trace shadow rays
    std::vector<Ray> shadow_rays( rays.size() );
    for( unsigned i = 0; i < num_rays; ++i )
    {
        // Pick point on light
        m_light_points[ i ] = Vector3( 9.0f, 9.0f, -9.0f );

        // Create ray
        const Vector3 p = toVector3( local_geom[i].position );
        shadow_rays[i] = Ray( p,
                              normalize( m_light_points[i] - p ),
                              1e8f,
                              rays[i].time() );
    }

    // Trace shdow rays
    m_ray_tracer.trace( RayTracer::ANY_HIT, shadow_rays );
    m_ray_tracer.join();  // TODO: REMOVE THIS.  maximize overlap of trace/shade
    std::vector<LocalGeometry> shadow_results;
    m_ray_tracer.getResults( shadow_results );
     

    // Shade while shadow rays are tracing
    for( unsigned i = 0; i < num_rays; ++i )
    {
        int material_id = local_geom[i].material_id;
        if( material_id == -1 )
        {
            m_results[i] = Color( 0.0f ); 
            continue;
        }

        // shadow and shade
        bool is_lit = shadow_results[i].material_id == -1;
        if( !is_lit ) 
        {
            m_results[i] = Color( 0.0f ); 
            continue;
        }

        float light = static_cast<float>( is_lit );
        //float lit = 1.0f;
        //m_results[i] = toColor(local_geom[i].geometric_normal) * Color( lit );
        const Vector3 p     = toVector3( local_geom[i].position );
        const Vector3 w_in  = normalize( m_light_points[i] - p );
        const Vector3 w_out = -rays[i].direction();
        //LLOG_INFO << " win : "<< w_in;
        //LLOG_INFO << " wout: "<< w_out;
        const ISurfaceShader* shader = m_shaders[ material_id ];
        if( !shader )
        {
            LLOG_INFO << "no shader found for id: " << material_id;
            m_results[i] = Color( 0.0f ); 
            continue;
        }
        
        m_results[i] = light*shader->evaluateBSDF( w_out, local_geom[i], w_in );
        //LLOG_INFO << "   bsdf: " << m_results[i] << " light: " << light;
    }
    

    // Trace shadow rays

    /*
    const unsigned num_rays = rays.size();
    m_results.resize( num_rays );
    for( unsigned i = 0; i < num_rays; ++i )
    {
        if( local_geom[i].material_id == -1 )
        {
            // TODO: add env map support
            m_results[i] = Color( 0.0f ); 
            continue;
        }

        // queryDirectLighting( lgeom, 
        //const ISurfaceShader* shader = m_shaders[ lgeom[i].material_id ];
        //const Vector3         w_in   = rays[ i ].getDirection();

        m_results[i] = toColor( local_geom[i].geometric_normal ); 
    }
    */
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
    m_lights.push_back( shader );
}

