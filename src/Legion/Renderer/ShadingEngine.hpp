 

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

/// \file ShadingEngine.hpp
/// ShadingEngine

#ifndef LEGION_RENDERER_SHADINGENGINE_HPP_
#define LEGION_RENDERER_SHADINGENGINE_HPP_

#include <Legion/Common/Math/MTRand.hpp>
#include <Legion/Common/Util/AutoTimerHelpers.hpp>
#include <Legion/Core/Color.hpp>
#include <Legion/Core/Ray.hpp>
#include <Legion/Renderer/Cuda/Shared.hpp>
#include <Legion/Renderer/LightSet.hpp>

#include <map>
#include <vector>


namespace legion
{

class  ILightShader;
class  ISurfaceShader;
class  RayTracer;

class ShadingEngine
{
public:
    typedef std::vector<Color> Results;

    ShadingEngine( RayTracer& ray_tracer );

    void reset();
    void logTimerInfo();

    void shade( std::vector<Ray>& rays );

    const Results& getResults()const;

    void addSurfaceShader( const ISurfaceShader* shader );

    void addLight( const ILightShader* shader );

    void     setMaxRayDepth( unsigned max_depth );
    unsigned maxRayDepth()const; 

    void     setSamplesPerPixel( const Index2& spp );

private:
    void shade( std::vector<Ray>&           rays,
                std::vector<LocalGeometry>& local_geom,
                std::vector<Color>&         ray_attenuation,
                bool                        count_emission );


    struct Closure
    {
        Closure() {}
        Closure( const Vector3& light_point, const ILightShader* light ) 
          : light_point( light_point ), light( light ) {}

        Vector3             light_point;
        const ILightShader* light;
    };


    typedef std::map<unsigned, const ISurfaceShader*> SurfaceShaderMap;
    typedef std::map<unsigned, const ILightShader*>   LightShaderMap;
    typedef std::vector<Closure>                      Closures;
    typedef std::vector<Ray>                          Rays;
    typedef std::vector<LocalGeometry>                LocalGeometries;

    Results          m_results;
    Closures         m_closures;
    Rays             m_secondary_rays;
    LocalGeometries  m_shadow_results;

    RayTracer&       m_ray_tracer;
    LightSet         m_light_set;
    SurfaceShaderMap m_surface_shaders;

    LoopTimerInfo    m_shadow_ray_gen;
    LoopTimerInfo    m_shadow_trace;
    LoopTimerInfo    m_radiance_trace;
    LoopTimerInfo    m_light_loop;

    unsigned         m_max_ray_depth;
    unsigned         m_pass_number;
    Index2           m_spp;

    MTRand32         m_rnd;



    std::vector<Vector2> m_sample_offsets;
};


}


#endif // LEGION_RENDERER_SHADINGENGINE_HPP_
