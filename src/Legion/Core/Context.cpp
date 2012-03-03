
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


#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>
#include <Legion/Core/Context.hpp>
#include <Legion/Core/config.hpp>
#include <Legion/Core/Exception.hpp>
#include <Legion/Scene/Camera/ICamera.hpp>
#include <Legion/Scene/Film/IFilm.hpp>
#include <Legion/Scene/LightShader/ILightShader.hpp>
#include <Legion/Scene/SurfaceShader/ISurfaceShader.hpp>

using namespace legion;

#define CHECK_NULL( f, ptr )  \
    if( !ptr ) throw Exception( std::string( f ) + ": " #ptr " param is NULL" );


Context::Context( const std::string& name ) 
    : APIBase( this, name )
{
    LLOG_INFO << "Creating legion::Context( " << name << " )";
}


Context::~Context()
{
}


optix::Buffer Context::createOptixBuffer()
{
    return m_renderer.createBuffer();
}


void Context::addMesh( Mesh* mesh )
{
    CHECK_NULL( LFUNC, mesh ); 

    LLOG_INFO << "Adding mesh <" << mesh->getName() << ">";
    m_renderer.addMesh( mesh );
}


void Context::addLight( const ILightShader* shader )
{
    CHECK_NULL( LFUNC, shader ); 

    LLOG_INFO << "Adding light <" << shader->getName() << ">";
    m_renderer.addLight( shader );
}


void Context::setActiveCamera( ICamera* camera )
{
    CHECK_NULL( LFUNC, camera ); 
  
    LLOG_INFO << "Adding camera <" << camera->getName() << ">";
    m_renderer.setCamera( camera );
}


void Context::setActiveFilm( IFilm* film )
{
    CHECK_NULL( LFUNC, film ); 

    LLOG_INFO << "Adding film <" << film->getName() << ">";
    m_renderer.setFilm( film );
}


void Context::render()
{
    LLOG_INFO << "rendering ....";
    m_renderer.render();
}


