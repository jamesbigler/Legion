
// Copyright (C) 2011 R. Keith Morley 
// 
// (MIT/X11 License)
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


#ifndef LEGION_CORE_CONTEXT_IMPL_H_
#define LEGION_CORE_CONTEXT_IMPL_H_

#include <Legion/Core/Context.hpp>
#include <Legion/Common/Util/Plugin.hpp>
#include <Legion/Renderer/OptixScene.hpp>

namespace legion
{

class ICamera;

class Context::Impl
{
public:
    Impl();
    ~Impl();

    void setRenderer   ( IRenderer* renderer );

    void setCamera     ( ICamera* camera );

    void setFilm       ( IFilm* film );

    void addGeometry( const IGeometry* geometry );

    void addLight( const ILight* light );

    void addAssetPath( const std::string& path );

    void render();

private:
    PluginManager   m_plugin_mgr;
    OptiXScene      m_optix_scene;

    ICamera* m_camera;

};

}


#endif // LEGION_CORE_CONTEXT_IMPL_H_
