
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

#include <Legion/Common/Util/Plugin.hpp>

using namespace legion;

class PluginManager::Impl
{
public:
    Impl() {}
    ~Impl() {}
};


//-----------------------------------------------------------------------------
//
// PluginManager 
//
//-----------------------------------------------------------------------------
PluginManager::PluginManager()
{
}


PluginManager::~PluginManager()
{
}


template <>
bool PluginManager::registerPlugin<ICamera>( 
        const std::string& name,
        ICamera* (*create)( const Parameters& params ) )
{
    return false;
}


template <>
bool PluginManager::registerPlugin<IFilm>(
        const std::string& name,
        IFilm* (*create)( const Parameters& params ) )
{
    return false;
}


template <>
bool PluginManager::registerPlugin<IGeometry>(
        const std::string& name,
        IGeometry* (*create)( const Parameters& params ) )
{
    return false;
}


template <>
bool PluginManager::registerPlugin<ILight>(
        const std::string& name,
        ILight* (*create)( const Parameters& params ) )
{
    return false;
}


template <>
bool PluginManager::registerPlugin<ISurfaceShader>(
        const std::string& name,
        ISurfaceShader* (*create)( const Parameters& params )
        )
{
    return false;
}


template <>
ICamera* PluginManager::create<ICamera>(
        const std::string& plugin_name,
        const Parameters& params )
{
    return 0;
}

template <>
IFilm* PluginManager::create<IFilm>(
        const std::string& plugin_name,
        const Parameters& params )
{
    return 0;
}



template <>
IGeometry* PluginManager::create<IGeometry>(
        const std::string& plugin_name,
        const Parameters& params )
{
    return 0;
}



template <>
ILight* PluginManager::create<ILight>(
        const std::string& plugin_name,
        const Parameters& params )
{
    return 0;
}



template <>
ISurfaceShader* PluginManager::create<ISurfaceShader>(
        const std::string& plugin_name,
        const Parameters& params )
{
    return 0;
}
