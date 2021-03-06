
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

#ifndef LEGION_RENDERER_OPTIX_SCENE_HPP_
#define LEGION_RENDERER_OPTIX_SCENE_HPP_

#include <optixu/optixpp_namespace.h>
#include <Legion/Renderer/OptiXProgramManager.hpp>

#include <vector>

namespace legion
{

class ICamera;
class IGeometry;
class ILight;
class IRenderer;


class OptiXScene
{
public:
    OptiXScene();
    ~OptiXScene();

    void setRenderer   ( IRenderer*    renderer );
    void setCamera     ( ICamera*      camera );
    void setEnvironment( IEnvironment* environment );
    void addLight      ( ILight*       light );
    void addGeometry   ( IGeometry*    geometry );

    void addAssetPath( const std::string& path );
    void sync();
    void loadAccelCache(const std::string& cache_file);
    void saveAccelCache(const std::string& cache_file);

    optix::Context getOptiXContext() { return m_optix_context; }

private:
    static const unsigned RADIANCE_TYPE = 0u;
    static const unsigned SHADOW_TYPE   = 1u;

    void initializeOptixContext();

    template <typename OptiXNode>
    void setSurfaceVariables( OptiXNode          node,
                              const ISurface*    surf );

    template <typename OptiXNode>
    void setTextureVariables( OptiXNode          node,
                              const std::string& name,
                              const ITexture*    tex );

    template <typename OptiXNode>
    void setNestedSurfaceVariables(
                              OptiXNode          node,
                              const std::string& name,
                              const ISurface*    surf );

    typedef std::map<IGeometry*, optix::GeometryInstance > GeometryMap;

    optix::Context           m_optix_context;
    optix::Program           m_raygen_program;
    optix::Program           m_create_ray_program;
    optix::Program           m_closest_hit_program;
    optix::Program           m_any_hit_program;
    optix::Program           m_environment_sample;
    optix::Program           m_environment_light_evaluate;
    optix::Program           m_environment_miss_evaluate;
    optix::Program           m_environment_program;
    optix::Program           m_default_texture_proc1;
    optix::Program           m_default_texture_proc2;
    optix::Program           m_default_texture_proc4;
    optix::GeometryGroup     m_top_group;

    OptiXProgramManager      m_program_mgr;

    IRenderer*               m_renderer;
    ICamera*                 m_camera;
    IEnvironment*            m_environment;
    std::vector<ILight*>     m_lights;
    GeometryMap              m_geometry;

    optix::Buffer            m_light_sample_buffer;
    std::vector<int>         m_light_sample_ids;

    optix::Buffer            m_light_eval_buffer;
    std::vector<int>         m_light_eval_ids;

    bool                     m_accel_cache_loaded;
};

}

#endif // LEGION_RENDERER_OPTIX_SCENE_HPP_
