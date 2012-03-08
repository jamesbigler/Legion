 

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

#include <vector>
#include <map>
#include <Legion/Core/Color.hpp>

namespace legion
{

class  ILightShader;
class  ISurfaceShader;
class  Ray;
class  RayTracer;
struct LocalGeometry;

class ShadingEngine
{
public:
    ShadingEngine( RayTracer& ray_tracer );

    typedef std::vector< Color > Results;
    void shade( const std::vector<Ray>& rays,
                const std::vector<LocalGeometry>& local_geom );

    const Results& getResults()const;

    void addSurfaceShader( const ISurfaceShader* shader );

    void addLight( const ILightShader* shader );

private:
    typedef std::map<unsigned, const ISurfaceShader*> ShaderMap;
    typedef std::vector<const ILightShader*>          LightList;

    Results          m_results;
    ShaderMap        m_shaders;
    LightList        m_lights;

    RayTracer&       m_ray_tracer;

};


}


#endif // LEGION_RENDERER_SHADINGENGINE_HPP_
