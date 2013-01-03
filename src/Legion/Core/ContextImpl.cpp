
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


#include <Legion/Core/ContextImpl.hpp>
#include <Legion/Common/Util/Parameters.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Scene/Geometry/Sphere.hpp>

using namespace legion;

Context::Impl::Impl() 
{
    m_plugin_mgr.registerPlugin<IGeometry>( "Sphere", &Sphere::create );

    Parameters params;
    IGeometry* geo = m_plugin_mgr.create<IGeometry>( "Sphere", params );
    LLOG_INFO << "\tSphere: " << geo;
    //Geometry* geo = ctx->create<Geometry>( "Sphere", properties );
}


Context::Impl::~Impl() 
{
}


void Context::Impl::setRenderParameters( const RenderParameters& params )
{
    m_optix_scene.setRenderParameters( params );
}


void Context::Impl::setRenderer( IRenderer* renderer )
{
}


void Context::Impl::setCamera( ICamera* camera )
{
    m_camera = camera;
    m_optix_scene.setCamera( camera );
}


void Context::Impl::setFilm( IFilm* film )
{
}


void Context::Impl::addGeometry( IGeometry* geometry )
{
    m_optix_scene.addGeometry( geometry );
}


void Context::Impl::addLight( ILight* light )
{
}


void Context::Impl::addAssetPath( const std::string& path )
{
}


void Context::Impl::render()
{
    m_optix_scene.renderPass( Index2( 0u, 0u ), Index2( 0u, 0u ), 1u );
}

