 

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

    // Trace shadow rays
    std::vector<Ray> shadow_rays( rays.size() );
    for( unsigned i = 0; i < num_rays; ++i )
    {
        // Pick point on light

        // Create ray
        shadow_rays[i] = Ray( toVector3( local_geom[i].position ),
                              Vector3( -1.0f, 0.25f, 0.25f ),
                              1e8f,
                              rays[i].time() );
    }
    m_ray_tracer.trace( RayTracer::ANY_HIT, shadow_rays );
    m_ray_tracer.join();  // TODO: REMOVE THIS.  maximize overlap of trace/shade
    std::vector<LocalGeometry> shadow_results;
    m_ray_tracer.getResults( shadow_results );
     

    // Shade while shadow rays are tracing
    for( unsigned i = 0; i < num_rays; ++i )
    {
        float lit = static_cast<float>( shadow_results[i].material_id != -1);
        //float lit = 1.0f;
        m_results[i] = toColor( local_geom[i].geometric_normal ) * Color( lit );
        // shadow and shade
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
    m_shaders[ shader->getID() ] = shader;
}

void ShadingEngine::addLight( const ILightShader* shader )
{
    m_lights.push_back( shader );
}

