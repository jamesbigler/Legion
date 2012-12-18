
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
#include <Legion/Core/ContextImpl.hpp>
#include <Legion/Core/Exception.hpp>
//#include <config.hpp>

using namespace legion;

#define CHECK_NULL( f, ptr )  \
    if( !ptr ) throw Exception( std::string( f ) + ": " #ptr " param is NULL" );



Context::Context() 
    : m_impl( new Impl )
{
    LLOG_INFO << "Creating legion::Context";
}


Context::~Context()
{
    LLOG_INFO << "Destroying legion::Context";
}


void Context::setRenderer( IRenderer* renderer )
{
    m_impl->setRenderer( renderer );
}


void Context::setCamera( ICamera* camera )
{
    m_impl->setCamera( camera );
}


void Context::setFilm( IFilm* film )
{
    m_impl->setFilm( film );
}


void Context::addGeometry( const IGeometry* geometry )
{
    m_impl->addGeometry( geometry );
}


void Context::addLight( const ILight* light )
{
    m_impl->addLight( light );
}


void Context::addAssetPath( const std::string& path )
{
    m_impl->addAssetPath( path );
}


void Context::render()
{
    m_impl->render();
}




